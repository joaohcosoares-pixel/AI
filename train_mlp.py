#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp.py (REFATORADO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Módulo de Treinamento, Isolamento Estatístico & Pipeline Híbrido -- v2.

Mudanças estruturais em relação à v1 (motivadas por auditoria de vazamento de
dados e de simulação aritmética não-empírica):

1. Seleção de estratégia (class_weights vs oversampling) e checkpoint de
   Early Stopping usam SOMENTE métricas de VALIDAÇÃO -- nunca o Teste.
   Critério: maximizar Recall da classe não-trivial (Cn != 0) na validação,
   com desempate por menor FPR. O Teste (15%) é tocado exatamente 1 vez, no
   fim do script, já com estratégia e limiar fixados.
2. Calibração de Alta Sensibilidade: limiar de decisão ancorado no k-ésimo
   menor score de P(Cn != 0) entre os positivos da VALIDAÇÃO (k_tolerance=0
   = score mínimo, comportamento estrito; k_tolerance=k>0 descarta os k
   positivos mais ruidosos em troca de queda de FPR). Esse limiar -- não o
   argmax do softmax -- é o que efetivamente opera o filtro de triagem. O
   Recall pontual no Teste é sempre reportado com IC 95% de Clopper-Pearson
   (N de positivos é pequeno; ponto isolado sem margem não é honesto).
3. Pipeline híbrido roda de fato sobre 10^5 pontos novos e reais em R^4:
   forward pass da MLP cronometrado com time.perf_counter(); custo do FHS
   medido com amostragem dupla e independente (região crítica sinalizada vs
   região trivial não-sinalizada), evitando sobrestimação por viés de borda.
   Reconstrução ponderada do controle FHS puro e destaque primário para a
   métrica invariante de Redução de Chamadas ao Oráculo.
4. A prevalência real (pi) usada na Projeção Bayesiana é lida diretamente da
   distribuição bruta do oráculo FHS (data_generator.generate_dataset), antes
   de qualquer rebalanceamento -- nunca hardcoded.
"""

import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint
from torch.utils.data import DataLoader, Dataset

from data_generator import _BOUNDS, compute_chern_rigorous

warnings.filterwarnings("ignore")

CSV_PATH = Path("topological_dataset.csv")
FEATURES = ["Ko", "h", "eps2", "eps3"]

# ══════════════════════════════════════════════════════════════════════════════
# PASSO 1.1 — ISOLAMENTO DO MOTOR ALEATÓRIO (reprodutibilidade estrita)
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    """Trava TODAS as fontes de estocasticidade do treino (Python, NumPy, PyTorch
    CPU/CUDA e o autotuner do cuDNN). Deve ser a primeira chamada de train_and_ablate."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """worker_init_fn do DataLoader: re-semeia numpy/random em cada worker a partir
    do gerador do PyTorch, para o caso de num_workers>0 (hoje 0, mas protege o futuro)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET & REDE NEURAL (MLP TOPOLÓGICA) -- inalterados
# ══════════════════════════════════════════════════════════════════════════════

class ChernDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class TopoPhaseMLP(nn.Module):
    def __init__(self, n_classes: int, p: float = 0.25):
        super().__init__()

        def _block(d_in: int, d_out: int):
            return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(p))

        self.net = nn.Sequential(
            _block(4, 128),
            _block(128, 256),
            _block(256, 128),
            _block(128, 64),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS: avaliação em ARGMAX (diagnóstico) + probabilidades para calibração
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, loader, device, trivial_idx: int, criterion=None) -> dict:
    """
    Roda o modelo sobre `loader` e retorna um dict com:
      - metricas macro (argmax) para referencia/diagnostico
      - fpr/tpr binarios (Cn=0 vs Cn!=0) no operating point ARGMAX
      - preds, targets, e as probabilidades P(Cn != 0) por amostra (probs_nt),
        necessarias para a calibracao de limiar (Tarefa 2) em outro lugar.
    """
    model.eval()
    all_preds, all_targets, all_probs_nt = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            if criterion is not None:
                total_loss += criterion(logits, yb).item() * len(yb)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(-1)
            probs_nt = 1.0 - probs[:, trivial_idx]
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_probs_nt.append(probs_nt.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    probs_nt = np.concatenate(all_probs_nt)
    loss = total_loss / len(loader.dataset) if criterion is not None else 0.0

    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)

    bin_targets = (targets != trivial_idx).astype(int)
    bin_preds_argmax = (preds != trivial_idx).astype(int)
    tn, fp, fn, tp = confusion_matrix(bin_targets, bin_preds_argmax, labels=[0, 1]).ravel()
    fpr_argmax = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_argmax = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "loss": loss, "prec_macro": prec, "rec_macro": rec, "f1_macro": f1,
        "fpr_argmax": fpr_argmax, "tpr_argmax": tpr_argmax,
        "preds": preds, "targets": targets,
        "probs_nt": probs_nt, "bin_targets": bin_targets,
    }


def binary_confusion_metrics(bin_targets: np.ndarray, bin_preds: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(bin_targets, bin_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "fpr": fpr, "tpr": tpr, "precision": prec}


# ══════════════════════════════════════════════════════════════════════════════
# TAREFA 2 — CALIBRAÇÃO DE ALTA SENSIBILIDADE (k-ésimo menor score, VALIDAÇÃO)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_threshold(bin_targets: np.ndarray, probs_nt: np.ndarray, k_tolerance: int = 0):
    """
    Calibração de Alta Sensibilidade: ancora o limiar de decisão no k-ésimo
    menor score de P(Cn != 0) entre os VERDADEIROS POSITIVOS da validação
    (indexação 0-based: k_tolerance=0 == score mínimo == comportamento
    estrito anterior). k_tolerance=k>0 descarta deliberadamente os k
    positivos mais ruidosos (menor confiança) da validação, trocando recall
    nominal de validação — que passa a ser (n_pos - k) / n_pos — por queda de
    FPR. Retorna (limiar, precisao_no_limiar).

    Se não houver nenhuma amostra positiva na validação, não há o que
    calibrar: retorna limiar=0.0 (sinaliza tudo) como fallback seguro.
    """
    if k_tolerance < 0:
        raise ValueError(f"k_tolerance deve ser >= 0 (recebido: {k_tolerance})")

    n_pos = int(bin_targets.sum())
    if n_pos == 0:
        return 0.0, 0.0
    if k_tolerance >= n_pos:
        raise ValueError(
            f"k_tolerance={k_tolerance} >= n_pos={n_pos}: positivos insuficientes "
            f"na validação para descartar essa quantidade sem esvaziar a garantia."
        )

    pos_scores_sorted = np.sort(probs_nt[bin_targets == 1])  # ascendente
    threshold = float(pos_scores_sorted[k_tolerance])

    bin_preds = apply_calibrated_threshold(probs_nt, threshold)
    precision_at_thr = binary_confusion_metrics(bin_targets, bin_preds)["precision"]
    return threshold, precision_at_thr


def apply_calibrated_threshold(probs_nt: np.ndarray, threshold: float) -> np.ndarray:
    return (probs_nt >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# TAREFA 1 — TREINO + SELEÇÃO DE ESTRATÉGIA ESTRITA (só validação) + TESTE 1x
# ══════════════════════════════════════════════════════════════════════════════

def train_and_ablate(csv_path=CSV_PATH, epochs=120, batch_size=256, lr=1e-3, patience=15,
                      seed: int = 42):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treinamento: {device}")

    df = pd.read_csv(csv_path)

    # --- Tarefa 4: prevalencia real, medida ANTES de qualquer rebalanceamento ---
    real_prevalence = float((df["chern"] != 0).mean())
    print(f"Prevalencia real (pi), medida do oraculo FHS antes do rebalanceamento: "
          f"{real_prevalence:.4%}  (N={len(df)}, nao-triviais={int((df['chern'] != 0).sum())})")

    X_raw = df[FEATURES].values.astype(np.float32)
    y_raw = df["chern"].values

    classes = np.sort(np.unique(y_raw))
    c2i = {int(c): i for i, c in enumerate(classes)}
    y = np.array([c2i[int(c)] for c in y_raw], dtype=np.int64)
    n_classes = len(classes)
    trivial_idx = int(np.where(classes == 0)[0][0])

    # Triplo split -- inalterado (ja era correto: 70/15/15 estratificado)
    X_tr_val, X_te, y_tr_val, y_te = train_test_split(
        X_raw, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_val, y_tr_val, test_size=0.1765, random_state=seed, stratify=y_tr_val
    )

    print(f"\nDivisao do dataset:")
    print(f"  - Treino:    {len(y_tr):>5} amostras (70%)")
    print(f"  - Validacao: {len(y_va):>5} amostras (15%) -> selecao de estrategia + "
          f"calibracao de limiar + early stopping")
    print(f"  - Teste:     {len(y_te):>5} amostras (15%) -> tocado 1 (uma) vez, no final")

    # Padronizacao ajustada estritamente apenas no treino (ja era correto)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    va_loader = DataLoader(ChernDataset(X_va_s, y_va), batch_size=512, shuffle=False)

    candidates = {}

    for strategy in ["class_weights", "oversampling"]:
        print(f"\n{'-' * 60}\n Treinando estrategia: {strategy.upper()}\n{'-' * 60}")

        if strategy == "oversampling":
            ros = RandomOverSampler(random_state=seed)
            X_tr_proc, y_tr_proc = ros.fit_resample(X_tr_s, y_tr)
            criterion = nn.CrossEntropyLoss()
        else:
            X_tr_proc, y_tr_proc = X_tr_s, y_tr
            counts = np.bincount(y_tr, minlength=n_classes)
            counts = np.where(counts == 0, 1, counts)  # evita divisao por zero
            weights = torch.tensor((len(y_tr) / (n_classes * counts)).astype(np.float32), device=device)
            criterion = nn.CrossEntropyLoss(weight=weights)

        g = torch.Generator()
        g.manual_seed(seed)
        tr_loader = DataLoader(ChernDataset(X_tr_proc, y_tr_proc), batch_size=batch_size,
                                shuffle=True, generator=g, worker_init_fn=seed_worker)

        model = TopoPhaseMLP(n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Tarefa 1: criterio de checkpoint = (recall_val, -fpr_val, -loss_val), MAXIMIZADO.
        # Nao e' mais a perda de validacao pura. Empates em (recall, fpr) sao
        # desempatados por menor loss (afinamento adicional, mesmo espirito do
        # criterio pedido: nunca usa o Teste).
        best_key = (-1.0, -1.0, -np.inf)
        best_state = None
        epochs_no_improve = 0
        epochs_run = 0

        for ep in range(1, epochs + 1):
            model.train()
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()

            va = evaluate_model(model, va_loader, device, trivial_idx, criterion)
            scheduler.step()
            epochs_run = ep

            key = (va["tpr_argmax"], -va["fpr_argmax"], -va["loss"])
            if key > best_key:
                best_key = key
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        model.load_state_dict(best_state)
        va_final = evaluate_model(model, va_loader, device, trivial_idx, criterion)

        # Tarefa 2: limiar calibrado NA VALIDACAO (nunca no teste)
        threshold, val_prec_at_thr = calibrate_threshold(
            va_final["bin_targets"], va_final["probs_nt"]
        )
        val_pred_at_thr = apply_calibrated_threshold(va_final["probs_nt"], threshold)
        val_metrics_at_thr = binary_confusion_metrics(va_final["bin_targets"], val_pred_at_thr)

        candidates[strategy] = {
            "model": model,
            "threshold": threshold,
            "epochs_run": epochs_run,
            # chave de SELECAO DE ESTRATEGIA: (recall_val_no_limiar, -fpr_val_no_limiar) -- so validacao
            "selection_key": (val_metrics_at_thr["tpr"], -val_metrics_at_thr["fpr"]),
            "val_f1_argmax": va_final["f1_macro"],
            "val_tpr_argmax": va_final["tpr_argmax"],
            "val_fpr_argmax": va_final["fpr_argmax"],
            "val_tpr_at_thr": val_metrics_at_thr["tpr"],
            "val_fpr_at_thr": val_metrics_at_thr["fpr"],
            "val_precision_at_thr": val_metrics_at_thr["precision"],
        }

        print(f"  [{strategy}] epocas ate' early-stop: {epochs_run}")
        print(f"  [{strategy}] Validacao (argmax):     F1-macro={va_final['f1_macro']:.4f}  "
              f"Recall={va_final['tpr_argmax']:.4f}  FPR={va_final['fpr_argmax']:.4f}")
        print(f"  [{strategy}] Limiar calibrado p/ Recall=1.0 (val): {threshold:.4f}  ->  "
              f"Recall={val_metrics_at_thr['tpr']:.4f}  FPR={val_metrics_at_thr['fpr']:.4f}  "
              f"Precisao={val_metrics_at_thr['precision']:.4f}")

    # --- Tarefa 1: selecao de estrategia usando SOMENTE metricas de VALIDACAO ---
    best_strategy = max(candidates.keys(), key=lambda k: candidates[k]["selection_key"])
    chosen = candidates[best_strategy]

    print(f"\n{'=' * 65}\n SELECAO DE ESTRATEGIA (por VALIDACAO -- Teste ainda nao foi tocado)\n{'=' * 65}")
    print(f"{'Estrategia':<16} | {'Recall@thr(val)':<16} | {'FPR@thr(val)':<14} | {'Precisao@thr(val)':<18}")
    print("-" * 68)
    for k, v in candidates.items():
        marker = "  <-- escolhida" if k == best_strategy else ""
        print(f"{k:<16} | {v['val_tpr_at_thr']:<16.4f} | {v['val_fpr_at_thr']:<14.4f} | "
              f"{v['val_precision_at_thr']:<18.4f}{marker}")

    # --- Teste tocado agora, UMA UNICA VEZ, com estrategia e limiar ja fixados ---
    te_loader = DataLoader(ChernDataset(X_te_s, y_te), batch_size=512, shuffle=False)
    te = evaluate_model(chosen["model"], te_loader, device, trivial_idx, criterion=None)
    te_pred_at_thr = apply_calibrated_threshold(te["probs_nt"], chosen["threshold"])
    te_metrics_at_thr = binary_confusion_metrics(te["bin_targets"], te_pred_at_thr)

    # Calibração de Alta Sensibilidade: IC exato (Clopper-Pearson) para o
    # recall pontual -- reportar Recall sem margem de erro, com N positivo
    # pequeno (~72 no teste), é uma alegação de precisão que os dados não
    # sustentam.
    ci_low, ci_high = proportion_confint(
        count=te_metrics_at_thr["tp"],
        nobs=te_metrics_at_thr["tp"] + te_metrics_at_thr["fn"],
        alpha=0.05,
        method="beta",
    )

    print(f"\n{'=' * 65}\n RESULTADO NO TESTE CEGO (tocado 1x) -- Estrategia: {best_strategy.upper()}\n{'=' * 65}")
    print(f" Operating point ARGMAX (diagnostico, nao usado no pipeline):")
    print(f"   F1-macro={te['f1_macro']:.4f}  Recall={te['tpr_argmax']:.4f}  FPR={te['fpr_argmax']:.4f}")
    print(f" Operating point CALIBRADO (limiar={chosen['threshold']:.4f}; este e' o usado no pipeline):")
    print(f"   Recall={te_metrics_at_thr['tpr']:.4f} (IC 95%: [{ci_low:.4f}, {ci_high:.4f}])  "
          f"FPR={te_metrics_at_thr['fpr']:.4f}  Precisao={te_metrics_at_thr['precision']:.4f}")
    print(f"   Matriz de confusao (binaria, Cn!=0 = positivo): "
          f"TP={te_metrics_at_thr['tp']}  FN={te_metrics_at_thr['fn']}  "
          f"FP={te_metrics_at_thr['fp']}  TN={te_metrics_at_thr['tn']}")
    print(f"\n Relatorio multi-classe no teste (argmax, diagnostico):")
    print(classification_report(te["targets"], te["preds"],
                                 target_names=[str(c) for c in classes], zero_division=0))

    result = {
        "strategy": best_strategy,
        "model": chosen["model"],
        "scaler": scaler,
        "classes": classes,
        "trivial_idx": trivial_idx,
        "threshold": chosen["threshold"],
        "real_prevalence": real_prevalence,
        "test_tpr": te_metrics_at_thr["tpr"],
        "test_tpr_ci95": (float(ci_low), float(ci_high)),
        "test_fpr": te_metrics_at_thr["fpr"],
        "test_precision_raw": te_metrics_at_thr["precision"],
        "test_confusion": te_metrics_at_thr,
        "test_f1_macro_argmax": te["f1_macro"],
        # expõe os splits já computados p/ benchmark_classical_baselines()
        "X_tr_s": X_tr_s, "y_tr": y_tr, "X_va_s": X_va_s, "y_va": y_va,
        "X_te_s": X_te_s, "y_te": y_te, "n_classes": n_classes,
    }
    return result, candidates


# ══════════════════════════════════════════════════════════════════════════════
# PASSO 1.3 + 1.4 — HARNESS DE AVALIAÇÃO EM MÚLTIPLAS SEMENTES + AGREGAÇÃO ESTATÍSTICA
# ══════════════════════════════════════════════════════════════════════════════

def run_multiseed_evaluation(n_trials: int = 20, base_seed: int = 0) -> pd.DataFrame:
    """
    Repete train_and_ablate() do zero em n_trials sementes independentes
    (split + init de pesos + shuffle do DataLoader, todos controlados pela
    MESMA seed por rodada, via seed_everything) e agrega Limiar/Recall/FPR/
    F1-Macro no teste. Substitui a alegacao de uma unica rodada por uma
    distribuicao com media, desvio-padrao, minimo e maximo.
    """
    resultados = []

    for seed in range(base_seed, base_seed + n_trials):
        result, _ = train_and_ablate(seed=seed)
        resultados.append({
            "seed": seed,
            "limiar": result["threshold"],
            "recall": result["test_tpr"],
            "fpr": result["test_fpr"],
            "f1_macro": result["test_f1_macro_argmax"],
        })
        print(f"[seed={seed}] Limiar={result['threshold']:.4f}  "
              f"Recall={result['test_tpr']:.4f}  FPR={result['test_fpr']:.4f}")

    df = pd.DataFrame(resultados)

    print(f"\n{'=' * 65}\n RELATORIO AGREGADO — {n_trials} SEMENTES INDEPENDENTES "
          f"(seeds {base_seed}..{base_seed + n_trials - 1})\n{'=' * 65}")
    for col in ["limiar", "recall", "fpr", "f1_macro"]:
        print(f" {col:10s}: mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
              f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    frac_full_recall = (df["recall"] >= 0.999).mean()
    print(f"\n Fracao de rodadas com Recall >= 0.999: {frac_full_recall:.4f} "
          f"({int((df['recall'] >= 0.999).sum())}/{n_trials})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — BENCHMARKING CLÁSSICO (RandomForest balanceada + Regressão Logística)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_classical_baselines(X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
                                   trivial_idx: int, n_classes: int) -> pd.DataFrame:
    """
    RandomForest(class_weight='balanced') e LogisticRegression(class_weight=
    'balanced'), no MESMO split/scaler da MLP, com a MESMA regra de calibração
    de limiar (maior limiar que garante Recall=1.0 na validação).
    """
    bin_va = (y_va != trivial_idx).astype(int)
    bin_te = (y_te != trivial_idx).astype(int)

    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }

    rows = []
    for name, model in candidates.items():
        t0 = time.perf_counter()
        model.fit(X_tr_s, y_tr)
        t_fit = time.perf_counter() - t0

        probs_va = model.predict_proba(X_va_s)
        thr, _ = calibrate_threshold(bin_va, 1.0 - probs_va[:, trivial_idx])

        probs_te = model.predict_proba(X_te_s)
        pred_te = apply_calibrated_threshold(1.0 - probs_te[:, trivial_idx], thr)
        m = binary_confusion_metrics(bin_te, pred_te)
        f1 = precision_recall_fscore_support(y_te, model.predict(X_te_s),
                                              average="macro", zero_division=0)[2]

        rows.append({"modelo": name, "f1_macro": f1, "limiar": thr,
                     "recall_teste": m["tpr"], "fpr_teste": m["fpr"], "fit_time_s": t_fit})

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 78}\n BLOCO 2 — BASELINES CLASSICOS (mesmo split/limiar da MLP)\n{'=' * 78}")
    print(df.to_string(index=False))
    df.to_csv("classical_baselines.csv", index=False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TAREFA 4 — PROJEÇÃO BAYESIANA (π dinâmico, nunca hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_precision_projection(test_tpr: float, test_fpr: float, real_prevalence: float):
    """
    Recalcula a Precisao Projetada em deployment (prevalencia real, `real_prevalence`,
    OBRIGATORIAMENTE fornecida pelo chamador -- sem default, para impedir reuso
    silencioso de um valor hardcoded desatualizado) via Teorema de Bayes / VPP.
    """
    pi = real_prevalence
    numerator = test_tpr * pi
    denominator = numerator + test_fpr * (1.0 - pi)
    projected_precision = numerator / denominator if denominator > 0 else 0.0

    # Comparativo dinamico: qual seria a precisao se o teste fosse avaliado num
    # cenario artificialmente balanceado 50:50 (pi=0.5) -- calculado com a MESMA
    # formula e os MESMOS tpr/fpr medidos, nao um numero de narrativa hardcoded.
    denom_5050 = test_tpr + test_fpr
    precision_5050 = (test_tpr / denom_5050) if denom_5050 > 0 else 0.0

    print(f"\n{'=' * 65}\n PROJECAO BAYESIANA DA PRECISAO REAL (pi dinamico)\n{'=' * 65}")
    print(f" Prevalencia real medida do oraculo FHS (pi): {pi:.4%}")
    print(f" Recall/TPR medido no teste (limiar calibrado): {test_tpr:.4f} ({test_tpr:.2%})")
    print(f" FPR medido no teste (limiar calibrado):        {test_fpr:.4f} ({test_fpr:.2%})")
    print(f" ---------------------------------------------------------------")
    print(f" Precisao se avaliado em cenario 50:50 (mesma formula, pi=0.5): {precision_5050:.4%}")
    print(f" Precisao projetada em deployment real (pi={pi:.4%}):            {projected_precision:.4%}")
    print(f" ---------------------------------------------------------------")

    return projected_precision


# ══════════════════════════════════════════════════════════════════════════════
# TAREFA 3 — PIPELINE HÍBRIDO: EXECUÇÃO EMPÍRICA REAL (perf_counter, sem aritmética fixa)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# GUARDA DE DOMÍNIO OOD (bbox exato + Mahalanobis)
# ══════════════════════════════════════════════════════════════════════════════

class TopoDomainGuard:
    """
    Softmax NÃO é usado como sinal de OOD (auditoria anterior: confiança
    satura em ~100% tanto no centro do domínio quanto 3x fora de _BOUNDS).
    Dois sinais independentes, calculados sobre o RAW (Ko,h,eps2,eps3):
      1. bbox exato contra `bounds` (ex.: data_generator._BOUNDS).
      2. Distância de Mahalanobis no espaço padronizado (`scaler`), calibrada
         no percentil `maha_percentile` da distribuição do próprio treino.
    """
    def __init__(self, bounds: dict, scaler, X_train_scaled: np.ndarray, maha_percentile: float = 99.5):
        self.bounds = bounds
        self.scaler = scaler
        self.mu = X_train_scaled.mean(axis=0)
        cov = np.cov(X_train_scaled, rowvar=False)
        self.cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        self.maha_threshold = float(np.percentile(self._mahalanobis(X_train_scaled), maha_percentile))

    def _mahalanobis(self, X_scaled: np.ndarray) -> np.ndarray:
        d = X_scaled - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.cov_inv, d))

    def check(self, X_raw: np.ndarray, feature_order=("Ko", "h", "eps2", "eps3")) -> dict:
        in_bbox = np.ones(len(X_raw), dtype=bool)
        for j, name in enumerate(feature_order):
            lo, hi = self.bounds[name]
            in_bbox &= (X_raw[:, j] >= lo) & (X_raw[:, j] <= hi)
        maha_dist = self._mahalanobis(self.scaler.transform(X_raw))
        in_density = maha_dist <= self.maha_threshold
        return {"trusted": in_bbox & in_density, "in_bbox": in_bbox,
                "in_density": in_density, "maha_dist": maha_dist}


def empirical_hybrid_pipeline(model, scaler, classes, trivial_idx: int, threshold: float,
                               X_train_scaled: np.ndarray,
                               n_eval: int = 100_000,
                               fhs_calibration_sample: int = 40,
                               fhs_full_audit: bool = False,
                               maha_percentile: float = 99.5,
                               seed: int = 7) -> dict | None:
    """
    Demonstracao EMPIRICA (sem aritmetica fixa) do pipeline hibrido:

      1. Sorteia n_eval pontos NOVOS em R^4, dentro dos limites fisicos do
         Hamiltoniano (data_generator._BOUNDS) -- nunca vistos em treino/val/teste.
      2. Mede com time.perf_counter() o tempo REAL do forward pass em lote da
         MLP sobre esses n_eval pontos.
      3. Aplica o limiar calibrado (definido na VALIDACAO) e guarda OOD para
         sinalizar candidatos criticos a Cn != 0 ou fora de dominio.
      4. Amostragem Dupla Independente de Custo FHS (data_generator.compute_chern_rigorous):
         - Amostra 1 (Sinalizados / Criticos): mede tempo medio real (t_fhs_flagged_ms)
           em pontos proximos a fronteiras de fase e fechamento de gap.
         - Amostra 2 (Nao-Sinalizados / Triviais): mede tempo medio real (t_fhs_unflagged_ms)
           em pontos do bulk isolante trivial.
      5. Reconstrucao ponderada honesta do custo de controle puro e avaliacao
         do speedup de parede empírico desacoplado, com destaque primario para
         a Reducao de Chamadas ao Oraculo.

    Limitacao explicita: a garantia de Recall=1.0 vem da calibracao na
    validacao (rotulos conhecidos). Para este novo lote de n_eval pontos SEM
    rotulo, nao recomputamos TPR/FPR reais -- isso exigiria rodar FHS nos
    n_eval pontos inteiros, o que anularia o proprio objetivo do pipeline. A
    garantia e', portanto, uma extrapolacao da validacao para a nova amostra,
    valida na medida em que a validacao for representativa do dominio D subset
    R^4 amostrado por data_generator.generate_dataset.
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    model.eval()

    print(f"\n{'=' * 65}\n PIPELINE HIBRIDO -- EXECUCAO EMPIRICA ({n_eval:,} pontos novos)\n{'=' * 65}")

    # 1. pontos novos, reais, nunca vistos por treino/val/teste
    Ko = rng.uniform(*_BOUNDS["Ko"], n_eval)
    h = rng.uniform(*_BOUNDS["h"], n_eval)
    eps2 = rng.uniform(*_BOUNDS["eps2"], n_eval)
    eps3 = rng.uniform(*_BOUNDS["eps3"], n_eval)
    X_new = np.stack([Ko, h, eps2, eps3], axis=1).astype(np.float32)
    X_new_s = scaler.transform(X_new).astype(np.float32)

    # 2. forward pass REAL, tempo REAL (perf_counter)
    X_tensor = torch.from_numpy(X_new_s).to(device)
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_mlp_total = time.perf_counter() - t0
    probs_nt = (1.0 - probs[:, trivial_idx]).cpu().numpy()
    t_mlp_ms_per_point = (t_mlp_total / n_eval) * 1000.0

    # 3. GATE DE DOMÍNIO -- instanciado com X_tr_s da rede de produção
    guard = TopoDomainGuard(_BOUNDS, scaler, X_train_scaled=X_train_scaled, maha_percentile=maha_percentile)
    gate = guard.check(X_new)
    ood_mask = ~gate["trusted"]

    # 4. limiar calibrado (Tarefa 2) OU fora-de-domínio -- qualquer ponto fora
    # do domínio confiável (bbox OU Mahalanobis) é interceptado e roteado para
    # o FHS diretamente, IGNORANDO a predição da MLP nesse ponto
    threshold_mask = probs_nt >= threshold
    flagged_mask = threshold_mask | ood_mask
    flagged_idx = np.where(flagged_mask)[0]
    unflagged_idx = np.where(~flagged_mask)[0]

    n_flagged = int(len(flagged_idx))
    n_unflagged = int(len(unflagged_idx))
    n_ood = int(ood_mask.sum())
    n_ood_only = int((ood_mask & ~threshold_mask).sum())
    oracle_reduction_pct = (1.0 - n_flagged / n_eval) * 100.0

    print(f" Pontos avaliados pela MLP: {n_eval:,}")
    print(f" Tempo REAL do forward pass (perf_counter): {t_mlp_total * 1000:.2f} ms total "
          f"({t_mlp_ms_per_point:.6f} ms/ponto)")
    print(f" Fora do domínio confiável (bbox ou Mahalanobis p{maha_percentile:g}): {n_ood:,} "
          f"({n_ood / n_eval:.2%}); destes, {n_ood_only:,} não teriam sido sinalizados pelo limiar sozinho")
    print(f" Sinalizados p/ auditoria FHS (limiar={threshold:.4f} OU fora-de-domínio): {n_flagged:,} "
          f"({n_flagged / n_eval:.2%} do lote)")
    print(f" [MÉTRICA PRIMÁRIA] Redução de Chamadas ao Oráculo FHS: {oracle_reduction_pct:.2f}% "
          f"({n_eval:,} -> {n_flagged:,} avaliações numéricas)")

    if n_flagged == 0:
        print(" Nenhum ponto sinalizado -- nada a auditar. Pipeline híbrido não aplicável neste lote.")
        return None

    # 5. AMOSTRAGEM DUPLA E INDEPENDENTE DE CUSTO FHS
    # Amostra 1 (Sinalizados / Críticos: fronteira de transição, gap estreito)
    if fhs_full_audit:
        sample_flagged_idx = flagged_idx
        print(f"\n [Amostra 1 - Crítica] fhs_full_audit=True: rodando FHS real em TODOS os {n_flagged:,} sinalizados...")
    else:
        k1 = min(fhs_calibration_sample, n_flagged)
        sample_flagged_idx = rng.choice(flagged_idx, size=k1, replace=False)
        print(f"\n [Amostra 1 - Crítica] Medindo custo FHS em k1={k1} pontos sinalizados (região de transição)...")

    t0 = time.perf_counter()
    fhs_flagged_labels = []
    for i in sample_flagged_idx:
        c = compute_chern_rigorous(float(Ko[i]), float(h[i]), float(eps2[i]), float(eps3[i]),
                                    N_init=60, n_occ=3)
        fhs_flagged_labels.append(c)
    t_fhs_flagged_sample_total = time.perf_counter() - t0
    t_fhs_flagged_ms = (t_fhs_flagged_sample_total / len(sample_flagged_idx)) * 1000.0 if len(sample_flagged_idx) > 0 else 0.0

    # Amostra 2 (Não-Sinalizados / Triviais: bulk isolante trivial)
    if n_unflagged > 0:
        k2 = min(fhs_calibration_sample, n_unflagged)
        sample_unflagged_idx = rng.choice(unflagged_idx, size=k2, replace=False)
        print(f" [Amostra 2 - Trivial] Medindo custo FHS em k2={k2} pontos não-sinalizados (bulk trivial)...")
        t0 = time.perf_counter()
        fhs_unflagged_labels = []
        for i in sample_unflagged_idx:
            c = compute_chern_rigorous(float(Ko[i]), float(h[i]), float(eps2[i]), float(eps3[i]),
                                        N_init=60, n_occ=3)
            fhs_unflagged_labels.append(c)
        t_fhs_unflagged_sample_total = time.perf_counter() - t0
        t_fhs_unflagged_ms = (t_fhs_unflagged_sample_total / len(sample_unflagged_idx)) * 1000.0
    else:
        sample_unflagged_idx = np.array([], dtype=int)
        t_fhs_unflagged_sample_total = 0.0
        t_fhs_unflagged_ms = t_fhs_flagged_ms

    # 6. CÁLCULO REALISTA DO CUSTO FHS PURO E HÍBRIDO
    if fhs_full_audit:
        t_fhs_audit_total_sec = t_fhs_flagged_sample_total
    else:
        t_fhs_audit_total_sec = (n_flagged * t_fhs_flagged_ms) / 1000.0

    t_pure_fhs_total_sec = (n_flagged * t_fhs_flagged_ms + n_unflagged * t_fhs_unflagged_ms) / 1000.0
    t_hybrid_total_sec = t_mlp_total + t_fhs_audit_total_sec
    speedup = t_pure_fhs_total_sec / t_hybrid_total_sec if t_hybrid_total_sec > 0 else float("inf")

    frac_nontrivial_in_sample = float(np.mean([c is not None and c != 0 for c in fhs_flagged_labels]))
    frac_none_in_sample = float(np.mean([c is None for c in fhs_flagged_labels]))

    print(f"\n --- CONTRASTE DE LATÊNCIAS FHS MEDIDAS (perf_counter) ---")
    print(f"  * Tempo FHS Região Crítica/Sinalizada:     {t_fhs_flagged_ms:.4f} ms/ponto (N_amostra={len(sample_flagged_idx)})")
    print(f"  * Tempo FHS Região Trivial/Não-Sinalizada: {t_fhs_unflagged_ms:.4f} ms/ponto (N_amostra={len(sample_unflagged_idx)})")
    if t_fhs_unflagged_ms > 0:
        print(f"  * Razão de lentidão relativa (Crítica/Trivial): {t_fhs_flagged_ms / t_fhs_unflagged_ms:.2f}x")

    print(f"\n --- CUSTOS TEMPORAIS & SPEEDUP (ESTIMATIVA SECUNDÁRIA) ---")
    print(f" Custo FHS puro ponderado p/ {n_eval:,} pts: {t_pure_fhs_total_sec:.2f} s ({t_pure_fhs_total_sec / 60:.1f} min)")
    print(f" Custo híbrido MEDIDO: MLP({t_mlp_total:.2f}s) + FHS-sinalizados({t_fhs_audit_total_sec:.2f}s) = "
          f"{t_hybrid_total_sec:.2f} s ({t_hybrid_total_sec / 60:.1f} min)")
    print(f" Speedup de parede medido: {speedup:.2f}x (secundário à redução de chamadas)")
    print(f" (Diagnóstico amostra crítica) Fração confirmada Cn!=0: {frac_nontrivial_in_sample:.2%} | "
          f"Gap fechado (None): {frac_none_in_sample:.2%}")

    return {
        "n_eval": n_eval,
        "n_flagged": n_flagged,
        "n_unflagged": n_unflagged,
        "oracle_reduction_pct": oracle_reduction_pct,
        "threshold": threshold,
        "n_ood": n_ood,
        "n_ood_only": n_ood_only,
        "ood_fraction": n_ood / n_eval,
        "t_mlp_total_s": t_mlp_total,
        "t_mlp_ms_per_point": t_mlp_ms_per_point,
        "t_fhs_flagged_ms": t_fhs_flagged_ms,
        "t_fhs_unflagged_ms": t_fhs_unflagged_ms,
        "t_fhs_audit_total_s": t_fhs_audit_total_sec,
        "t_hybrid_total_s": t_hybrid_total_sec,
        "t_pure_fhs_total_s": t_pure_fhs_total_sec,
        "speedup_measured": speedup,
        "fhs_flagged_sample_size": len(sample_flagged_idx),
        "fhs_unflagged_sample_size": len(sample_unflagged_idx),
        "fraction_confirmed_nontrivial_in_sample": frac_nontrivial_in_sample,
        "fraction_gap_closed_in_sample": frac_none_in_sample,
        "full_audit": fhs_full_audit,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"ERRO: Dataset {CSV_PATH} nao encontrado. Execute data_generator.py primeiro.")
    else:
        # 1. AUDITORIA ESTATÍSTICA (Comprovação de estabilidade exigida pelo Bloco 1)
        print(f"\n{'=' * 65}\n FASE 1: AUDITORIA ESTOCÁSTICA (ENSEMBLE DE SEMENTES)\n{'=' * 65}")
        df_stats = run_multiseed_evaluation(n_trials=30, base_seed=0)

        # 2. TREINAMENTO DE PRODUÇÃO
        # ERRO CORRIGIDO: seed=42 (fixo, arbitrário) NÃO pertence a range(0,15)
        # auditado na FASE 1 -- a estatística do ensemble não certificava a
        # semente que ia para produção. Produção agora usa a MEDIANA do
        # próprio ensemble já auditado (por recall, desempate por -fpr).
        seed_producao = int(
            df_stats.sort_values(["recall", "fpr"], ascending=[True, True])
            .iloc[len(df_stats) // 2]["seed"]
        )
        print(f"\n{'=' * 65}\n FASE 2: TREINAMENTO DO MODELO DE PRODUÇÃO "
              f"(SEED={seed_producao}, mediana do ensemble da FASE 1)\n{'=' * 65}")
        result, all_candidates = train_and_ablate(epochs=120, batch_size=256, lr=1e-3, seed=seed_producao)

        # 2b. BENCHMARK CLÁSSICO -- mesmo split/limiar da MLP escolhida na FASE 2
        baseline_df = benchmark_classical_baselines(
            result["X_tr_s"], result["y_tr"], result["X_va_s"], result["y_va"],
            result["X_te_s"], result["y_te"], result["trivial_idx"], result["n_classes"],
        )

        # 3. PROJEÇÃO BAYESIANA
        proj_prec = bayesian_precision_projection(
            test_tpr=result["test_tpr"],
            test_fpr=result["test_fpr"],
            real_prevalence=result["real_prevalence"],
        )

        # 4. SIMULAÇÃO EMPÍRICA DO PIPELINE HÍBRIDO (Em 10^5 pontos reais)
        hybrid_report = empirical_hybrid_pipeline(
            model=result["model"],
            scaler=result["scaler"],
            classes=result["classes"],
            trivial_idx=result["trivial_idx"],
            threshold=result["threshold"],
            X_train_scaled=result["X_tr_s"],
            n_eval=100_000,
        )

        # 5. RESUMO EXECUTIVO
        print(f"\n{'=' * 65}\n RESUMO EXECUTIVO\n{'=' * 65}")
        print(f" Estrategia escolhida (por validacao):      {result['strategy']}")
        print(f" Limiar calibrado (Recall=1.0 na validacao): {result['threshold']:.4f}")
        print(f" Teste cego (1x), no limiar calibrado:       Recall={result['test_tpr']:.4f}  "
              f"FPR={result['test_fpr']:.4f}")
        print(f" Prevalencia real medida (pi):                {result['real_prevalence']:.4%}")
        print(f" Precisao projetada em deployment (Bayes):    {proj_prec:.4%}")
        if hybrid_report is not None:
            print(f"\n [METRICA PRIMARIA] Reducao de Chamadas FHS:  {hybrid_report['oracle_reduction_pct']:.2f}% "
                  f"({hybrid_report['n_eval']:,} -> {hybrid_report['n_flagged']:,} chamadas)")
            print(f" Contraste FHS (Critico vs Trivial):          {hybrid_report['t_fhs_flagged_ms']:.4f} ms/pt vs {hybrid_report['t_fhs_unflagged_ms']:.4f} ms/pt")
            print(f" Speedup de parede ponderado (secundario):    {hybrid_report['speedup_measured']:.2f}x "
                  f"({hybrid_report['t_pure_fhs_total_s']:.1f}s puro vs {hybrid_report['t_hybrid_total_s']:.1f}s hibrido)")
