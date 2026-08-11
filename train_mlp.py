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
2. Limiar de decisão calibrado via curva Precisão-Recall NA VALIDAÇÃO, para
   o maior limiar que ainda garante Recall = 1.0 (0 falsos negativos) nesse
   conjunto. Esse limiar -- não o argmax do softmax -- é o que efetivamente
   opera o filtro de triagem.
3. Pipeline híbrido roda de fato sobre 10^5 pontos novos e reais em R^4:
   forward pass da MLP cronometrado com time.perf_counter(); custo do FHS
   medido também com perf_counter() (não mais uma constante importada do
   artigo antigo) numa amostra dos pontos sinalizados, e extrapolado
   linearmente para o total sinalizado (ou, opcionalmente, auditado por
   inteiro com fhs_full_audit=True). Nenhuma aritmética fixa.
4. A prevalência real (pi) usada na Projeção Bayesiana é lida diretamente da
   distribuição bruta do oráculo FHS (data_generator.generate_dataset), antes
   de qualquer rebalanceamento -- nunca hardcoded.
"""

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data_generator import _BOUNDS, compute_chern_rigorous

warnings.filterwarnings("ignore")

CSV_PATH = Path("topological_dataset.csv")
FEATURES = ["Ko", "h", "eps2", "eps3"]

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
# TAREFA 2 — CALIBRAÇÃO DE LIMIAR (Precision-Recall na VALIDAÇÃO, Recall=1.0)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_threshold_for_full_recall(bin_targets: np.ndarray, probs_nt: np.ndarray):
    """
    Usa precision_recall_curve (sklearn) sobre P(Cn != 0) na VALIDAÇÃO para achar
    o MAIOR limiar de probabilidade que ainda garante Recall = 1.0 (0 falsos
    negativos) nesse conjunto. Retorna (limiar, precisao_no_limiar).

    Se nao houver nenhuma amostra positiva na validacao, nao ha o que calibrar:
    retorna limiar=0.0 (sinaliza tudo) como fallback seguro.
    """
    if bin_targets.sum() == 0:
        return 0.0, 0.0

    precision, recall, thresholds = precision_recall_curve(bin_targets, probs_nt)
    # precision/recall tem 1 elemento a mais que thresholds (ponto sintetico
    # threshold=+inf); alinhamos descartando esse ultimo ponto.
    idx_full_recall = np.where(recall[:-1] >= 1.0 - 1e-9)[0]
    if len(idx_full_recall) == 0:
        # nem o limiar mais baixo (0) recupera 100% dos positivos na validacao
        # -- nao deveria acontecer (threshold=0 sinaliza tudo), mas por
        # seguranca retornamos o ponto de maior recall disponivel.
        idx_best = int(np.argmax(recall[:-1])) if len(recall) > 1 else 0
        thr = float(thresholds[idx_best]) if len(thresholds) > 0 else 0.0
        return thr, float(precision[idx_best])

    idx = idx_full_recall[-1]  # maior limiar que ainda preserva recall=1.0
    return float(thresholds[idx]), float(precision[idx])


def apply_calibrated_threshold(probs_nt: np.ndarray, threshold: float) -> np.ndarray:
    return (probs_nt >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# TAREFA 1 — TREINO + SELEÇÃO DE ESTRATÉGIA ESTRITA (só validação) + TESTE 1x
# ══════════════════════════════════════════════════════════════════════════════

def train_and_ablate(csv_path=CSV_PATH, epochs=120, batch_size=256, lr=1e-3, patience=15):
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
        X_raw, y, test_size=0.15, random_state=42, stratify=y
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_val, y_tr_val, test_size=0.1765, random_state=42, stratify=y_tr_val
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
            ros = RandomOverSampler(random_state=42)
            X_tr_proc, y_tr_proc = ros.fit_resample(X_tr_s, y_tr)
            criterion = nn.CrossEntropyLoss()
        else:
            X_tr_proc, y_tr_proc = X_tr_s, y_tr
            counts = np.bincount(y_tr, minlength=n_classes)
            counts = np.where(counts == 0, 1, counts)  # evita divisao por zero
            weights = torch.tensor((len(y_tr) / (n_classes * counts)).astype(np.float32), device=device)
            criterion = nn.CrossEntropyLoss(weight=weights)

        tr_loader = DataLoader(ChernDataset(X_tr_proc, y_tr_proc), batch_size=batch_size, shuffle=True)

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
        threshold, val_prec_at_thr = calibrate_threshold_for_full_recall(
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

    print(f"\n{'=' * 65}\n RESULTADO NO TESTE CEGO (tocado 1x) -- Estrategia: {best_strategy.upper()}\n{'=' * 65}")
    print(f" Operating point ARGMAX (diagnostico, nao usado no pipeline):")
    print(f"   F1-macro={te['f1_macro']:.4f}  Recall={te['tpr_argmax']:.4f}  FPR={te['fpr_argmax']:.4f}")
    print(f" Operating point CALIBRADO (limiar={chosen['threshold']:.4f}; este e' o usado no pipeline):")
    print(f"   Recall={te_metrics_at_thr['tpr']:.4f}  FPR={te_metrics_at_thr['fpr']:.4f}  "
          f"Precisao={te_metrics_at_thr['precision']:.4f}")
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
        "test_fpr": te_metrics_at_thr["fpr"],
        "test_precision_raw": te_metrics_at_thr["precision"],
        "test_confusion": te_metrics_at_thr,
        "test_f1_macro_argmax": te["f1_macro"],
    }
    return result, candidates


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

def empirical_hybrid_pipeline(model, scaler, classes, trivial_idx: int, threshold: float,
                               n_eval: int = 100_000,
                               fhs_calibration_sample: int = 40,
                               fhs_full_audit: bool = False,
                               seed: int = 7) -> dict | None:
    """
    Demonstracao EMPIRICA (nao mais aritmetica fixa) do pipeline hibrido:

      1. Sorteia n_eval pontos NOVOS em R^4, dentro dos limites fisicos do
         Hamiltoniano (data_generator._BOUNDS) -- nunca vistos em treino/val/teste.
      2. Mede com time.perf_counter() o tempo REAL do forward pass em lote da
         MLP sobre esses n_eval pontos.
      3. Aplica o limiar calibrado (Tarefa 2, definido na VALIDACAO) para
         sinalizar candidatos a Cn != 0.
      4. Mede com time.perf_counter() o custo REAL do integrador FHS
         (data_generator.compute_chern_rigorous) numa amostra dos pontos
         sinalizados -- a amostra e' extraida dos PROPRIOS sinalizados (nao de
         pontos aleatorios do dominio inteiro), porque sao eles que carregam o
         viés para perto da fronteira de fechamento de gap, onde o FHS pode
         ficar mais lento (reavaliacao adaptativa) -- e extrapola linearmente
         para o total sinalizado. Com fhs_full_audit=True, audita TODOS os
         sinalizados (sem extrapolacao; mais lento).

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

    # 3. aplica limiar calibrado (Tarefa 2)
    flagged_mask = probs_nt >= threshold
    flagged_idx = np.where(flagged_mask)[0]
    n_flagged = int(len(flagged_idx))

    print(f" Pontos avaliados pela MLP: {n_eval:,}")
    print(f" Tempo REAL do forward pass (perf_counter): {t_mlp_total * 1000:.2f} ms total "
          f"({t_mlp_ms_per_point:.6f} ms/ponto)")
    print(f" Sinalizados p/ auditoria FHS (limiar={threshold:.4f}): {n_flagged:,} "
          f"({n_flagged / n_eval:.2%} do lote)")

    if n_flagged == 0:
        print(" Nenhum ponto sinalizado -- nada a auditar. Pipeline hibrido nao aplicavel neste lote.")
        return None

    # 4. custo do FHS: medido agora, neste hardware, nunca importado de outra fonte
    if fhs_full_audit:
        sample_idx = flagged_idx
        print(f" fhs_full_audit=True: rodando FHS real em TODOS os {n_flagged:,} sinalizados "
              f"(sem extrapolacao; pode demorar).")
    else:
        k = min(fhs_calibration_sample, n_flagged)
        sample_idx = rng.choice(flagged_idx, size=k, replace=False)
        print(f" Medindo custo do FHS numa amostra de {k} pontos sinalizados "
              f"(nao os {n_flagged:,} inteiros, por custo computacional -- ver docstring); "
              f"extrapolacao linear a partir dessa medicao real.")

    t0 = time.perf_counter()
    fhs_labels = []
    for i in sample_idx:
        c = compute_chern_rigorous(float(Ko[i]), float(h[i]), float(eps2[i]), float(eps3[i]),
                                    N_init=60, n_occ=3)
        fhs_labels.append(c)
    t_fhs_sample_total = time.perf_counter() - t0
    t_fhs_ms_per_point_measured = (t_fhs_sample_total / len(sample_idx)) * 1000.0

    print(f" Tempo REAL do FHS medido agora (perf_counter), media sobre {len(sample_idx)} "
          f"avaliacoes reais: {t_fhs_ms_per_point_measured:.4f} ms/ponto")

    if fhs_full_audit:
        t_fhs_audit_total_sec = t_fhs_sample_total
    else:
        t_fhs_audit_total_sec = (n_flagged * t_fhs_ms_per_point_measured) / 1000.0

    t_hybrid_total_sec = t_mlp_total + t_fhs_audit_total_sec
    t_pure_fhs_total_sec = (n_eval * t_fhs_ms_per_point_measured) / 1000.0
    speedup = t_pure_fhs_total_sec / t_hybrid_total_sec if t_hybrid_total_sec > 0 else float("inf")

    frac_nontrivial_in_sample = float(np.mean([c is not None and c != 0 for c in fhs_labels]))
    frac_none_in_sample = float(np.mean([c is None for c in fhs_labels]))

    print(f"\n Custo FHS puro estimado p/ {n_eval:,} pts (mesma medicao, mesmo hardware): "
          f"{t_pure_fhs_total_sec:.2f} s ({t_pure_fhs_total_sec / 60:.1f} min)")
    print(f" Custo hibrido MEDIDO: MLP({t_mlp_total:.2f}s) + FHS-nos-sinalizados"
          f"({t_fhs_audit_total_sec:.2f}s) = {t_hybrid_total_sec:.2f} s "
          f"({t_hybrid_total_sec / 60:.1f} min)")
    print(f" SPEEDUP MEDIDO: {speedup:.2f}x")
    print(f" (diagnostico, amostra auditada) fracao confirmada Cn!=0 pelo FHS: "
          f"{frac_nontrivial_in_sample:.2%}  |  fracao com gap fechado/indefinido (None): "
          f"{frac_none_in_sample:.2%}")

    return {
        "n_eval": n_eval, "n_flagged": n_flagged, "threshold": threshold,
        "t_mlp_total_s": t_mlp_total, "t_mlp_ms_per_point": t_mlp_ms_per_point,
        "t_fhs_ms_per_point_measured": t_fhs_ms_per_point_measured,
        "t_fhs_audit_total_s": t_fhs_audit_total_sec,
        "t_hybrid_total_s": t_hybrid_total_sec,
        "t_pure_fhs_total_s": t_pure_fhs_total_sec,
        "speedup_measured": speedup,
        "fhs_sample_size": len(sample_idx),
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
        result, all_candidates = train_and_ablate(epochs=120, batch_size=256, lr=1e-3)

        proj_prec = bayesian_precision_projection(
            test_tpr=result["test_tpr"],
            test_fpr=result["test_fpr"],
            real_prevalence=result["real_prevalence"],  # Tarefa 4: pi dinamico
        )

        hybrid_report = empirical_hybrid_pipeline(
            model=result["model"],
            scaler=result["scaler"],
            classes=result["classes"],
            trivial_idx=result["trivial_idx"],
            threshold=result["threshold"],
            n_eval=100_000,
        )

        print(f"\n{'=' * 65}\n RESUMO EXECUTIVO\n{'=' * 65}")
        print(f" Estrategia escolhida (por validacao):      {result['strategy']}")
        print(f" Limiar calibrado (Recall=1.0 na validacao): {result['threshold']:.4f}")
        print(f" Teste cego (1x), no limiar calibrado:       Recall={result['test_tpr']:.4f}  "
              f"FPR={result['test_fpr']:.4f}")
        print(f" Prevalencia real medida (pi):                {result['real_prevalence']:.4%}")
        print(f" Precisao projetada em deployment (Bayes):    {proj_prec:.4%}")
        if hybrid_report is not None:
            print(f" Speedup medido (empirico, {hybrid_report['n_eval']:,} pts novos, "
                  f"perf_counter): {hybrid_report['speedup_measured']:.2f}x")
