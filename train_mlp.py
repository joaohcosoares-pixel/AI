#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp.py — Pipeline de Avaliação Estatística Rigorosa com Isolamento Out-of-Sample Estrito
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Classificação de Fases Topológicas com Rigor Metodológico para Periódicos de Alto Impacto.

Princípios Metodológicos e Estruturais Implementados:
1. ISOLAMENTO OUT-OF-SAMPLE ESTRITO:
   - Partição global inicial (85% Desenvolvimento / 15% Holdout Externo)
     sob semente fixa e estratificação estrita.
   - O Holdout Externo é colocado em quarentena estrita e permanece intocado
     durante toda a fase de exploração, busca de hiperparâmetros, avaliação 5x2 CV,
     estimativa de prevalência e fixação de limiares.
   - A escolha do modelo de produção é conduzida exclusivamente a partir do
     desempenho na validação interna do conjunto de desenvolvimento de forma simétrica e independente.
   - O conjunto Holdout Externo é avaliado exatamente 1 (uma) única vez no final do
     pipeline, garantindo estimativas de generalização não enviesadas.

2. PROTOCOLO 5x2 CROSS-VALIDATION (Dietterich / Nadeau-Bengio):
   - Executa 5 iterações com partição estratificada 50%/50% (treino/validação) sobre X_dev,
     invertendo os papéis para gerar 10 dobras (folds) controladas.
   - MLP, Random Forest Balanceado e Regressão Logística Balanceada operam exatamente
     nas mesmas 10 dobras e sob a mesma padronização ajustada estritamente no treino de cada dobra.

3. INFERÊNCIA ESTATÍSTICA PAREADA (Testes 5x2 CV para ΔFPR e ΔF1-Macro):
   - Cálculo da distribuição das diferenças pareadas:
       ΔFPR = FPR_MLP - FPR_Baseline
       ΔF1_macro = F1_MLP - F1_Baseline
   - Reporte da média pareada, erro padrão, IC de 95% e teste de hipótese 5x2 CV de Dietterich (df = 5).
   - Omitida a comparação tautológica de ΔRecall na validação (fixada em 100% pelo limiar k=0).

4. HEURÍSTICA FIXA DE ALTA SENSIBILIDADE (k=0):
   - O limiar de corte operacional é fixado determinística e estritamente no
     menor escore P(Cn != 0) observado entre os verdadeiros positivos da validação
     interna (k=0).
   - Documentado explicitamente como escolha de política arquitetural de alta
     sensibilidade para triagem física de fases raras (não constituindo alegação
     de controle de risco calibrado conformalmente).
"""

from __future__ import annotations

import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from data_generator import _BOUNDS, compute_chern_rigorous
from domain_guard import TopoDomainGuard

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBAIS E CONFIGURAÇÃO EXPERIMENTAL
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH: Path = Path("topological_dataset.csv")
FEATURES: List[str] = ["M", "p2", "p3", "p4"]
GLOBAL_HOLDOUT_SEED: int = 42
HOLDOUT_TEST_SIZE: float = 0.15
N_5X2CV_ITERATIONS: int = 5
N_SEEDS_EVALUATION: int = 10


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — CONTROLE DETERMINÍSTICO E ISOLAMENTO DO MOTOR ALEATÓRIO
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    """
    Trava rigorosamente todas as fontes de estocasticidade do pipeline:
    Python random, NumPy RNG, PyTorch CPU/CUDA e autotuner determinístico do cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """
    Garante sementes estatisticamente reprodutíveis para subprocessos/workers
    de DataLoader PyTorch.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — DATASET E REDE NEURAL (MLP TOPOLÓGICA — ARQUITETURA INALTERADA)
# ══════════════════════════════════════════════════════════════════════════════

class ChernDataset(Dataset):
    """Encapsulador PyTorch para tensores de features contínuas e rótulos de classe de Chern."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X: torch.Tensor = torch.from_numpy(X.astype(np.float32))
        self.y: torch.Tensor = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class TopoPhaseMLP(nn.Module):
    """
    Perceptron Multicamadas (MLP) Profundo para Classificação de Fases Topológicas.

    Arquitetura estritamente preservada:
      - Entrada: 4 features (Ko, h, eps2, eps3)
      - Camadas: Linear(4, 128) -> GELU -> Dropout(p=0.25)
                 Linear(128, 256) -> GELU -> Dropout(p=0.25)
                 Linear(256, 128) -> GELU -> Dropout(p=0.25)
                 Linear(128, 64) -> GELU -> Dropout(p=0.25)
                 Linear(64, n_classes)
    """

    def __init__(self, n_classes: int, p: float = 0.25) -> None:
        super().__init__()

        def _block(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(p))

        self.net = nn.Sequential(
            _block(4, 128),
            _block(128, 256),
            _block(256, 128),
            _block(128, 64),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — MÉTRICAS ESTATÍSTICAS E HEURÍSTICA DE ALTA SENSIBILIDADE
# ══════════════════════════════════════════════════════════════════════════════

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calcula o Intervalo de Confiança Exato de Clopper-Pearson (distribuição Beta)
    para uma proporção binomial k/n com nível de confiança (1 - alpha).
    """
    if n == 0:
        return 0.0, 1.0
    low = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    high = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low, high


def binary_confusion_metrics(
    bin_targets: np.ndarray, bin_preds: np.ndarray
) -> Dict[str, Union[int, float]]:
    """
    Calcula matriz de confusão binária e métricas operacionais para triagem
    da fase não-trivial (Cn != 0 definido como classe positiva = 1).
    """
    tn, fp, fn, tp = confusion_matrix(bin_targets, bin_preds, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fpr,
        "tpr": tpr,
        "recall": tpr,
        "precision": prec,
        "specificity": spec,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    trivial_idx: int,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Avalia a MLP sobre um DataLoader e retorna predições, métricas macro, perda
    e o vetor contínuo de probabilidades não-triviais P(Cn != 0).
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_probs_nt: List[np.ndarray] = []
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
    loss = (
        total_loss / len(loader.dataset)
        if criterion is not None and len(loader.dataset) > 0
        else 0.0
    )

    prec, rec, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    bin_targets = (targets != trivial_idx).astype(np.int64)
    bin_preds_argmax = (preds != trivial_idx).astype(np.int64)
    cm_argmax = binary_confusion_metrics(bin_targets, bin_preds_argmax)

    return {
        "loss": float(loss),
        "prec_macro": float(prec),
        "rec_macro": float(rec),
        "f1_macro": float(f1),
        "fpr_argmax": cm_argmax["fpr"],
        "tpr_argmax": cm_argmax["tpr"],
        "preds": preds,
        "targets": targets,
        "probs_nt": probs_nt,
        "bin_targets": bin_targets,
    }


def compute_high_sensitivity_threshold(
    bin_targets: np.ndarray, probs_nt: np.ndarray
) -> float:
    """
    Heurística Fixa de Alta Sensibilidade (Fixed High-Sensitivity Heuristic, k=0):
    Define o limiar de decisão operacional como o menor escore predito P(Cn != 0) entre
    os verdadeiros positivos observados no conjunto de validação interna:
        tau = min { P(Cn != 0 | x_i) : i in Validação, y_i != 0 }

    NOTA DE RIGOR METODOLÓGICO:
    Este limiar é uma heurística determinística imposta arquiteturalmente para priorizar
    sensibilidade máxima em triagem e evitar a perda de fases topológicas raras. Não
    constitui garantia matemática formal de calibração nem controle de risco conformalizado
    com cobertura exata assegurada fora da amostra.

    Retorna:
        threshold (float): Menor escore positivo de validação (ou 0.0 caso não haja positivos).
    """
    pos_mask = bin_targets == 1
    if not np.any(pos_mask):
        return 0.0
    pos_scores = probs_nt[pos_mask]
    return float(np.min(pos_scores))


def apply_high_sensitivity_threshold(
    probs_nt: np.ndarray, threshold: float
) -> np.ndarray:
    """Aplica o limiar da heurística fixa de alta sensibilidade para binarização (1 se >= limiar, 0 caso contrário)."""
    return (probs_nt >= threshold).astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — MOTOR DE INFERÊNCIA ESTATÍSTICA PAREADA (5x2 Cross-Validation)
# ══════════════════════════════════════════════════════════════════════════════

def compute_paired_differences(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Calcula estatísticas pareadas sob o protocolo 5x2 Cross-Validation (Dietterich, 1998):
      - 5 replicações com 2 dobras controladas (total de 10 dobras).
      - Para cada replicação r, variância s_r^2 = (p_{r,0} - p_bar_r)^2 + (p_{r,1} - p_bar_r)^2.
      - Estimativa de variância de Dietterich: s_5x2^2 = (1/5) * sum_{r=0}^4 s_r^2.
      - Estatística t_5x2 = p_{0,0} / sqrt(s_5x2^2) com 5 graus de liberdade (df = 5).
      - Média pareada (mu_Delta), desvio padrão amostral (s_Delta) e IC 95% sobre as 10 dobras.
    """
    diffs = np.asarray(metric_a, dtype=np.float64) - np.asarray(metric_b, dtype=np.float64)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    sem_diff = std_diff / np.sqrt(n) if n > 0 else 0.0

    alpha = 1.0 - confidence_level
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)) if n > 1 else 0.0
    ci_lower = mean_diff - t_crit * sem_diff
    ci_upper = mean_diff + t_crit * sem_diff

    # Protocolo 5x2 CV de Dietterich (df = 5) se n == 10
    if n == 10:
        diff_pairs = diffs.reshape(5, 2)
        p0_1 = float(diff_pairs[0, 0])
        r_means = np.mean(diff_pairs, axis=1)
        s_r_sq = np.sum((diff_pairs - r_means[:, None]) ** 2, axis=1)
        s_pooled_sq = float(np.mean(s_r_sq))  # (1/5) * sum(s_r_sq)
        denom = np.sqrt(s_pooled_sq) if s_pooled_sq > 1e-15 else 1e-15
        t_stat = float(p0_1 / denom)
        p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=5)))
    else:
        t_stat, p_value = stats.ttest_1samp(diffs, 0.0) if n > 1 else (0.0, 1.0)

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "sem_diff": sem_diff,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "diffs": diffs,
    }


def print_comparative_statistical_report(
    df_runs: pd.DataFrame, n_seeds: int
) -> None:
    """
    Imprime relatório formatado de auditoria estatística 5x2 CV agregada e análise pareada
    dobra a dobra (MLP vs Random Forest e MLP vs Regressão Logística).
    Exclui comparações tautológicas de Recall (fixadas em 100% pelo limiar k=0).
    """
    print(f"\n{'=' * 86}")
    print(f" RELATÓRIO ESTATÍSTICO 5x2 CROSS-VALIDATION ({n_seeds} DOBRAS NO DESENVOLVIMENTO)")
    print(f"{'=' * 86}")

    # 1. Tabela de Desempenho Agregado Absoluto (mu +- sigma, min, max)
    models = ["MLP", "RandomForest", "LogisticRegression"]
    summary_rows = []
    for m in models:
        sub = df_runs[df_runs["modelo"] == m]
        rec_mean, rec_std = float(sub["recall_val"].mean()), float(sub["recall_val"].std())
        fpr_mean, fpr_std = float(sub["fpr_val"].mean()), float(sub["fpr_val"].std())
        f1_mean, f1_std = float(sub["f1_macro_val"].mean()), float(sub["f1_macro_val"].std())
        thr_mean, thr_std = float(sub["limiar"].mean()), float(sub["limiar"].std())
        summary_rows.append({
            "Modelo": m,
            "Recall_Val (μ ± σ)": f"{rec_mean:.4f} ± {rec_std:.4f}",
            "FPR_Val (μ ± σ)": f"{fpr_mean:.4f} ± {fpr_std:.4f}",
            "F1_Macro (μ ± σ)": f"{f1_mean:.4f} ± {f1_std:.4f}",
            "Limiar (μ ± σ)": f"{thr_mean:.4f} ± {thr_std:.4f}",
        })

    df_summary_abs = pd.DataFrame(summary_rows)
    print("\n--- 1. DESEMPENHO AGREGADO ABSOLUTO (Validação Interna — 10 Dobras) ---")
    print(df_summary_abs.to_string(index=False))

    # 2. Estatística Pareada de Diferença 5x2 CV (Δ = MLP - Baseline)
    sub_mlp = df_runs[df_runs["modelo"] == "MLP"].sort_values("fold_id" if "fold_id" in df_runs.columns else "seed")
    sub_rf = df_runs[df_runs["modelo"] == "RandomForest"].sort_values("fold_id" if "fold_id" in df_runs.columns else "seed")
    sub_lr = df_runs[df_runs["modelo"] == "LogisticRegression"].sort_values("fold_id" if "fold_id" in df_runs.columns else "seed")

    paired_comparisons = [
        ("MLP vs RandomForest", sub_mlp, sub_rf),
        ("MLP vs LogisticRegression", sub_mlp, sub_lr),
    ]

    print(f"\n--- 2. INFERÊNCIA ESTATÍSTICA PAREADA 5x2 CV (Δ = MLP - Baseline, {n_seeds} Dobras) ---")
    print(" (Nota: Comparação de ΔRecall omitida pois a heurística k=0 fixa 100% de Recall na validação)")
    paired_rows = []

    for name, df_a, df_b in paired_comparisons:
        for metric_key, label in [
            ("fpr_val", "Δ FPR"),
            ("f1_macro_val", "Δ F1-Macro"),
        ]:
            stats_p = compute_paired_differences(
                df_a[metric_key].values, df_b[metric_key].values, confidence_level=0.95
            )
            sig_marker = "***" if stats_p["p_value"] < 0.001 else ("**" if stats_p["p_value"] < 0.01 else ("*" if stats_p["p_value"] < 0.05 else "n.s."))
            paired_rows.append({
                "Comparação": name,
                "Métrica": label,
                "Média Δ (μ_Δ)": f"{stats_p['mean_diff']:+.4f}",
                "IC 95% [Inf, Sup]": f"[{stats_p['ci_lower']:+.4f}, {stats_p['ci_upper']:+.4f}]",
                "Desvio (s_Δ)": f"{stats_p['std_diff']:.4f}",
                "Estatística t (5x2)": f"{stats_p['t_stat']:+.3f}",
                "p-valor (5x2 CV)": f"{stats_p['p_value']:.4e} ({sig_marker})",
            })

    df_paired = pd.DataFrame(paired_rows)
    print(df_paired.to_string(index=False))

    # Diagnóstico de Colapso do RandomForest sob a heurística k=0
    rf_collapse_count = int((sub_rf["limiar"] == 0.0).sum())
    print(f"\n--- 3. DIAGNÓSTICO METODOLÓGICO DE COLAPSO DE LIMIAR (Random Forest) ---")
    print(f" Dobras com colapso de limiar no RF (limiar = 0.0 -> FPR = 100%): "
          f"{rf_collapse_count}/{n_seeds} ({rf_collapse_count / n_seeds:.1%})")
    print(f" Nota: Árvores de decisão puras geram folhas onde a probabilidade empírica "
          f"atribuída a um ponto positivo pode atingir 0.0 exato, colapsando a heurística k=0.")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5 — TREINAMENTO DE SEMENTE INDIVIDUAL DA MLP NO SUB-FOLD DE DESENVOLVIMENTO
# ══════════════════════════════════════════════════════════════════════════════

def train_mlp_single_seed(
    X_tr_s: np.ndarray,
    y_tr: np.ndarray,
    X_va_s: np.ndarray,
    y_va: np.ndarray,
    n_classes: int,
    trivial_idx: int,
    device: torch.device,
    seed: int,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
) -> Dict[str, Any]:
    """
    Treina a MLP em um único sub-fold de treino/validação interna.
    Avalia as estratégias de ponderação de perda e oversampling, selecionando
    o melhor checkpoint exclusivamente a partir do desempenho de validação.
    """
    seed_everything(seed)
    va_loader = DataLoader(ChernDataset(X_va_s, y_va), batch_size=512, shuffle=False)
    candidates: Dict[str, Any] = {}

    for strategy in ["class_weights", "oversampling"]:
        if strategy == "oversampling":
            ros = RandomOverSampler(random_state=seed)
            X_tr_proc, y_tr_proc = ros.fit_resample(X_tr_s, y_tr)
            criterion = nn.CrossEntropyLoss()
        else:
            X_tr_proc, y_tr_proc = X_tr_s, y_tr
            counts = np.bincount(y_tr, minlength=n_classes)
            counts = np.where(counts == 0, 1, counts)
            weights = torch.tensor(
                (len(y_tr) / (n_classes * counts)).astype(np.float32), device=device
            )
            criterion = nn.CrossEntropyLoss(weight=weights)

        g = torch.Generator()
        g.manual_seed(seed)
        tr_loader = DataLoader(
            ChernDataset(X_tr_proc, y_tr_proc),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
        )

        model = TopoPhaseMLP(n_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

            # Critério de seleção de checkpoint: maximizar TPR argmax, minimizar FPR argmax, minimizar loss
            key = (va["tpr_argmax"], -va["fpr_argmax"], -va["loss"])
            if key > best_key:
                best_key = key
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        va_final = evaluate_model(model, va_loader, device, trivial_idx, criterion)

        # Heurística Fixa de Alta Sensibilidade (k=0) na validação interna
        threshold = compute_high_sensitivity_threshold(
            va_final["bin_targets"], va_final["probs_nt"]
        )
        val_pred_at_thr = apply_high_sensitivity_threshold(va_final["probs_nt"], threshold)
        val_metrics_at_thr = binary_confusion_metrics(va_final["bin_targets"], val_pred_at_thr)

        candidates[strategy] = {
            "model": model,
            "strategy": strategy,
            "threshold": threshold,
            "epochs_run": epochs_run,
            "selection_key": (
                val_metrics_at_thr["tpr"],
                -val_metrics_at_thr["fpr"],
                va_final["f1_macro"],
                -va_final["loss"],
            ),
            "val_f1_macro": va_final["f1_macro"],
            "val_tpr_at_thr": val_metrics_at_thr["tpr"],
            "val_fpr_at_thr": val_metrics_at_thr["fpr"],
            "val_precision_at_thr": val_metrics_at_thr["precision"],
            "val_loss": va_final["loss"],
        }

    # Seleção da melhor estratégia com base estrita no desempenho de validação
    best_strat = max(candidates.keys(), key=lambda k: candidates[k]["selection_key"])
    return candidates[best_strat]


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 6 — PROTOCOLO 5x2 CROSS-VALIDATION NO DESENVOLVIMENTO (10 DOBRAS)
# ══════════════════════════════════════════════════════════════════════════════

def run_symmetric_multiseed_evaluation(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    trivial_idx: int,
    n_classes: int,
    n_iterations: int = 5,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executa o protocolo 5x2 Cross-Validation (Dietterich / Nadeau-Bengio) estritamente sobre o
    conjunto de Desenvolvimento.
    Gera 5 iterações com divisão 50%/50% (treino/validação), invertendo os papéis para gerar
    10 dobras (folds) controladas para MLP, Random Forest e Regressão Logística.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_folds_total = n_iterations * 2
    print(f"\n{'=' * 86}")
    print(f" EXECUÇÃO DO PROTOCOLO 5x2 CROSS-VALIDATION ({n_iterations} ITERAÇÕES / {n_folds_total} DOBRAS NO DEV)")
    print(f" Dispositivo: {device} | Holdout Externo mantido 100% isolado em quarentena")
    print(f"{'=' * 86}")

    all_runs: List[Dict[str, Any]] = []
    mlp_registry: Dict[int, Dict[str, Any]] = {}
    rf_registry: Dict[int, Any] = {}
    lr_registry: Dict[int, Any] = {}

    idx_all = np.arange(len(y_dev))

    for iter_idx in range(n_iterations):
        # Partição estratificada 50% / 50% para a iteração corrente (Dietterich 5x2 CV)
        idx_a, idx_b = train_test_split(
            idx_all,
            test_size=0.5,
            random_state=iter_idx,
            stratify=y_dev,
        )

        fold_pairs = [
            (0, idx_a, idx_b),  # Dobra 0 da iteração: Treina em A, Valida em B
            (1, idx_b, idx_a),  # Dobra 1 da iteração: Treina em B, Valida em A
        ]

        for fold_in_iter, tr_idx, va_idx in fold_pairs:
            fold_id = iter_idx * 2 + fold_in_iter
            seed_everything(100 + fold_id)

            X_tr, y_tr = X_dev[tr_idx], y_dev[tr_idx]
            X_va, y_va = X_dev[va_idx], y_dev[va_idx]

            # Padronização ajustada EXCLUSIVAMENTE sobre o conjunto de treino desta dobra
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_va_s = scaler.transform(X_va)

            bin_va = (y_va != trivial_idx).astype(np.int64)

            # ── 1. MLP TOPOLÓGICA ──
            t0_mlp = time.perf_counter()
            mlp_res = train_mlp_single_seed(
                X_tr_s=X_tr_s,
                y_tr=y_tr,
                X_va_s=X_va_s,
                y_va=y_va,
                n_classes=n_classes,
                trivial_idx=trivial_idx,
                device=device,
                seed=100 + fold_id,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
            )
            t_mlp = time.perf_counter() - t0_mlp

            mlp_registry[fold_id] = {
                "fold_id": fold_id,
                "iteration": iter_idx,
                "fold_in_iter": fold_in_iter,
                "model": mlp_res["model"],
                "scaler": scaler,
                "X_train_scaled": X_tr_s,
                "threshold": mlp_res["threshold"],
                "strategy": mlp_res["strategy"],
                "selection_key": mlp_res["selection_key"],
                "val_tpr": mlp_res["val_tpr_at_thr"],
                "val_fpr": mlp_res["val_fpr_at_thr"],
                "val_f1": mlp_res["val_f1_macro"],
                "val_loss": mlp_res["val_loss"],
            }

            all_runs.append({
                "modelo": "MLP",
                "iteration": iter_idx,
                "fold_in_iter": fold_in_iter,
                "fold_id": fold_id,
                "seed": fold_id,
                "estrategia": mlp_res["strategy"],
                "limiar": mlp_res["threshold"],
                "recall_val": mlp_res["val_tpr_at_thr"],
                "fpr_val": mlp_res["val_fpr_at_thr"],
                "f1_macro_val": mlp_res["val_f1_macro"],
                "precision_val": mlp_res["val_precision_at_thr"],
                "tempo_treino_s": t_mlp,
            })

            # ── 2. RANDOM FOREST BALANCEADO ──
            rf = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=100 + fold_id,
                n_jobs=-1,
            )
            t0_rf = time.perf_counter()
            rf.fit(X_tr_s, y_tr)
            t_rf = time.perf_counter() - t0_rf

            probs_va_rf = rf.predict_proba(X_va_s)
            probs_nt_va_rf = 1.0 - probs_va_rf[:, trivial_idx]
            thr_rf = compute_high_sensitivity_threshold(bin_va, probs_nt_va_rf)
            pred_va_rf = apply_high_sensitivity_threshold(probs_nt_va_rf, thr_rf)
            m_rf = binary_confusion_metrics(bin_va, pred_va_rf)
            f1_rf = float(precision_recall_fscore_support(
                y_va, rf.predict(X_va_s), average="macro", zero_division=0
            )[2])

            rf_registry[fold_id] = {
                "fold_id": fold_id,
                "iteration": iter_idx,
                "fold_in_iter": fold_in_iter,
                "model": rf,
                "scaler": scaler,
                "threshold": thr_rf,
                "selection_key": (m_rf["tpr"], -m_rf["fpr"], f1_rf),
                "val_tpr": m_rf["tpr"],
                "val_fpr": m_rf["fpr"],
                "val_f1": f1_rf,
            }
            all_runs.append({
                "modelo": "RandomForest",
                "iteration": iter_idx,
                "fold_in_iter": fold_in_iter,
                "fold_id": fold_id,
                "seed": fold_id,
                "estrategia": "balanced",
                "limiar": thr_rf,
                "recall_val": m_rf["tpr"],
                "fpr_val": m_rf["fpr"],
                "f1_macro_val": f1_rf,
                "precision_val": m_rf["precision"],
                "tempo_treino_s": t_rf,
            })

            # ── 3. REGRESSÃO LOGÍSTICA BALANCEADA ──
            lr_model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=100 + fold_id,
            )
            t0_lr = time.perf_counter()
            lr_model.fit(X_tr_s, y_tr)
            t_lr = time.perf_counter() - t0_lr

            probs_va_lr = lr_model.predict_proba(X_va_s)
            probs_nt_va_lr = 1.0 - probs_va_lr[:, trivial_idx]
            thr_lr = compute_high_sensitivity_threshold(bin_va, probs_nt_va_lr)
            pred_va_lr = apply_high_sensitivity_threshold(probs_nt_va_lr, thr_lr)
            m_lr = binary_confusion_metrics(bin_va, pred_va_lr)
            f1_lr = float(precision_recall_fscore_support(
                y_va, lr_model.predict(X_va_s), average="macro", zero_division=0
            )[2])

            lr_registry[fold_id] = {
                "fold_id": fold_id,
                "iteration": iter_idx,
                "fold_in_iter": fold_in_iter,
                "model": lr_model,
                "scaler": scaler,
                "threshold": thr_lr,
                "selection_key": (m_lr["tpr"], -m_lr["fpr"], f1_lr),
                "val_tpr": m_lr["tpr"],
                "val_fpr": m_lr["fpr"],
                "val_f1": f1_lr,
            }
            all_runs.append({
                "modelo": "LogisticRegression",
                "iteration": iter_idx,
                "fold_in_iter": fold_in_iter,
                "fold_id": fold_id,
                "seed": fold_id,
                "estrategia": "balanced",
                "limiar": thr_lr,
                "recall_val": m_lr["tpr"],
                "fpr_val": m_lr["fpr"],
                "f1_macro_val": f1_lr,
                "precision_val": m_lr["precision"],
                "tempo_treino_s": t_lr,
            })

            print(
                f"  [5x2 CV Iter {iter_idx + 1}/5, Dobra {fold_in_iter + 1}/2 (ID {fold_id})] "
                f"MLP (FPR_val={mlp_res['val_fpr_at_thr']:.4f}, F1={mlp_res['val_f1_macro']:.4f}) | "
                f"RF (FPR_val={m_rf['fpr']:.4f}) | LR (FPR_val={m_lr['fpr']:.4f})"
            )

    df_runs = pd.DataFrame(all_runs)
    models_registry = {
        "mlp": mlp_registry,
        "rf": rf_registry,
        "lr": lr_registry,
    }
    return df_runs, models_registry


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 7 — SELEÇÃO DO MODELO DE PRODUÇÃO E AVALIAÇÃO NO HOLDOUT EXTERNO
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_production_on_blind_holdout(
    production_candidate: Dict[str, Any],
    baselines_production: Dict[str, Any],
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    trivial_idx: int,
    classes: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Executa o teste de avaliação exatamente 1 (uma) única vez sobre o conjunto
    Holdout Externo (15%) após o congelamento estrito do modelo de produção, do
    scaler e do limiar heurístico.
    """
    print(f"\n{'=' * 86}")
    print(" AVALIAÇÃO NO CONJUNTO HOLDOUT EXTERNO (15% — ISOLAMENTO OUT-OF-SAMPLE ESTRITO)")
    print(f"{'=' * 86}")

    scaler: StandardScaler = production_candidate["scaler"]
    mlp_model: nn.Module = production_candidate["model"]
    mlp_threshold: float = production_candidate["threshold"]

    # Padronização do Holdout Externo com o scaler congelado do treino da MLP
    X_holdout_s_mlp = scaler.transform(X_holdout)
    bin_holdout = (y_holdout != trivial_idx).astype(np.int64)

    # ── Avaliação da MLP ──
    ho_loader = DataLoader(ChernDataset(X_holdout_s_mlp, y_holdout), batch_size=512, shuffle=False)
    ho_eval = evaluate_model(mlp_model, ho_loader, device, trivial_idx, criterion=None)

    mlp_ho_pred_at_thr = apply_high_sensitivity_threshold(ho_eval["probs_nt"], mlp_threshold)
    mlp_ho_metrics = binary_confusion_metrics(bin_holdout, mlp_ho_pred_at_thr)

    ci_low_mlp, ci_high_mlp = clopper_pearson_ci(
        k=mlp_ho_metrics["tp"],
        n=mlp_ho_metrics["tp"] + mlp_ho_metrics["fn"],
        alpha=0.05,
    )

    # ── Avaliação dos Baselines Congelados no Holdout (com seus scalers independentes) ──
    rf_model = baselines_production["rf"]["model"]
    rf_threshold = baselines_production["rf"]["threshold"]
    rf_scaler: StandardScaler = baselines_production["rf"].get("scaler", scaler)
    X_holdout_s_rf = rf_scaler.transform(X_holdout)
    probs_ho_rf = rf_model.predict_proba(X_holdout_s_rf)
    probs_nt_ho_rf = 1.0 - probs_ho_rf[:, trivial_idx]
    rf_ho_pred = apply_high_sensitivity_threshold(probs_nt_ho_rf, rf_threshold)
    rf_ho_metrics = binary_confusion_metrics(bin_holdout, rf_ho_pred)
    ci_low_rf, ci_high_rf = clopper_pearson_ci(
        k=rf_ho_metrics["tp"], n=rf_ho_metrics["tp"] + rf_ho_metrics["fn"], alpha=0.05
    )

    lr_model = baselines_production["lr"]["model"]
    lr_threshold = baselines_production["lr"]["threshold"]
    lr_scaler: StandardScaler = baselines_production["lr"].get("scaler", scaler)
    X_holdout_s_lr = lr_scaler.transform(X_holdout)
    probs_ho_lr = lr_model.predict_proba(X_holdout_s_lr)
    probs_nt_ho_lr = 1.0 - probs_ho_lr[:, trivial_idx]
    lr_ho_pred = apply_high_sensitivity_threshold(probs_nt_ho_lr, lr_threshold)
    lr_ho_metrics = binary_confusion_metrics(bin_holdout, lr_ho_pred)
    ci_low_lr, ci_high_lr = clopper_pearson_ci(
        k=lr_ho_metrics["tp"], n=lr_ho_metrics["tp"] + lr_ho_metrics["fn"], alpha=0.05
    )

    print("\n--- DESEMPENHO NO HOLDOUT EXTERNO (N = %d amostras, %d não-triviais) ---"
          % (len(y_holdout), int(bin_holdout.sum())))

    holdout_table = [
        {
            "Modelo": "MLP (Produção)",
            "Limiar": f"{mlp_threshold:.4f}",
            "Recall @ Thr": f"{mlp_ho_metrics['tpr']:.4f}",
            "IC 95% Clopper-Pearson": f"[{ci_low_mlp:.4f}, {ci_high_mlp:.4f}]",
            "FPR @ Thr": f"{mlp_ho_metrics['fpr']:.4f}",
            "Precisão": f"{mlp_ho_metrics['precision']:.4f}",
            "TP / FN / FP / TN": f"{mlp_ho_metrics['tp']} / {mlp_ho_metrics['fn']} / {mlp_ho_metrics['fp']} / {mlp_ho_metrics['tn']}",
        },
        {
            "Modelo": "RandomForest",
            "Limiar": f"{rf_threshold:.4f}",
            "Recall @ Thr": f"{rf_ho_metrics['tpr']:.4f}",
            "IC 95% Clopper-Pearson": f"[{ci_low_rf:.4f}, {ci_high_rf:.4f}]",
            "FPR @ Thr": f"{rf_ho_metrics['fpr']:.4f}",
            "Precisão": f"{rf_ho_metrics['precision']:.4f}",
            "TP / FN / FP / TN": f"{rf_ho_metrics['tp']} / {rf_ho_metrics['fn']} / {rf_ho_metrics['fp']} / {rf_ho_metrics['tn']}",
        },
        {
            "Modelo": "LogisticRegression",
            "Limiar": f"{lr_threshold:.4f}",
            "Recall @ Thr": f"{lr_ho_metrics['tpr']:.4f}",
            "IC 95% Clopper-Pearson": f"[{ci_low_lr:.4f}, {ci_high_lr:.4f}]",
            "FPR @ Thr": f"{lr_ho_metrics['fpr']:.4f}",
            "Precisão": f"{lr_ho_metrics['precision']:.4f}",
            "TP / FN / FP / TN": f"{lr_ho_metrics['tp']} / {lr_ho_metrics['fn']} / {lr_ho_metrics['fp']} / {lr_ho_metrics['tn']}",
        },
    ]

    print(pd.DataFrame(holdout_table).to_string(index=False))

    print("\n--- RELATÓRIO MULTI-CLASSE DA MLP NO HOLDOUT (Operating Point Argmax, Diagnóstico) ---")
    print(
        classification_report(
            ho_eval["targets"],
            ho_eval["preds"],
            labels=list(range(len(classes))),
            target_names=[str(c) for c in classes],
            zero_division=0,
        )
    )

    return {
        "mlp_metrics": mlp_ho_metrics,
        "mlp_ci95": (ci_low_mlp, ci_high_mlp),
        "mlp_f1_macro": ho_eval["f1_macro"],
        "rf_metrics": rf_ho_metrics,
        "lr_metrics": lr_ho_metrics,
        "X_holdout_scaled": X_holdout_s_mlp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 8 — PROJEÇÃO BAYESIANA DE PRECISÃO (π medido empiricamente)
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_precision_projection(
    test_tpr: float, test_fpr: float, real_prevalence: float
) -> Dict[str, float]:
    """
    Projeta o Valor Preditivo Positivo (Precisão) em condições de distribuição
    sob prior uniforme prescrito via Teorema de Bayes utilizando a prevalência (pi).
    """
    pi = real_prevalence
    numerator = test_tpr * pi
    denominator = numerator + test_fpr * (1.0 - pi)
    projected_precision = numerator / denominator if denominator > 0 else 0.0

    denom_5050 = test_tpr + test_fpr
    precision_5050 = (test_tpr / denom_5050) if denom_5050 > 0 else 0.0

    print(f"\n{'=' * 86}")
    print(" PROJEÇÃO BAYESIANA DE PRECISÃO EM DEPLOYMENT REAL (π Dinâmico)")
    print(f"{'=' * 86}")
    print(f" Prevalência de Evento Raro induzida pelo prior uniforme de amostragem prescrito (π, calculada em Dev): {pi:.4%}")
    print(f" Sensibilidade / Recall medido no Holdout Externo:  {test_tpr:.4%}")
    print(f" Taxa de Falsos Positivos (FPR) no Holdout Externo: {test_fpr:.4%}")
    print(f" ----------------------------------------------------------------------------------")
    print(f" Precisão em cenário artificialmente balanceado (π = 0.50): {precision_5050:.4%}")
    print(f" Precisão Projetada no espaço contínuo (π = {pi:.2%}):             {projected_precision:.4%}")
    print(f" ----------------------------------------------------------------------------------")

    return {
        "real_prevalence": pi,
        "precision_5050": precision_5050,
        "projected_precision": projected_precision,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 9 — GUARDA DE DOMÍNIO OOD E SIMULAÇÃO EMPÍRICA DO PIPELINE HÍBRIDO
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_domain_guard(
    model: nn.Module,
    scaler: StandardScaler,
    guard: TopoDomainGuard,
    classes: np.ndarray,
    trivial_idx: int,
    M: float,
    p2: float,
    p3: float,
    p4: float,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Predição de fase topológica para uma única amostra com Guarda de Domínio OOD.

    Aplica a política estrita de contenção:
      1. Checagem de Bounding Box Físico e Densidade Espectral (Mahalanobis).
      2. Se ood_mask = True, a predição da MLP é sumariamente descartada
         e o ponto é marcado para cálculo mandatório pelo Oráculo FHS.
      3. Se confiável no domínio, avalia o limiar de triagem P(Cn != 0) >= threshold.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    x_raw = np.array([[M, p2, p3, p4]], dtype=np.float32)
    guard_res = guard.check(x_raw)

    in_bbox = bool(guard_res["in_bbox"])
    in_density = bool(guard_res["in_density"])
    ood_mask = bool(guard_res["ood_mask"])
    maha_dist = float(guard_res["maha_dist"])

    x_scaled = scaler.transform(x_raw).astype(np.float32)
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_scaled).to(device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_chern = int(classes[pred_idx])
    confidence = float(probs[pred_idx])
    prob_nt = float(1.0 - probs[trivial_idx])
    threshold_mask = bool(prob_nt >= threshold)
    flagged_for_oracle = bool(threshold_mask or ood_mask)

    return {
        "pred_chern": pred_chern if not ood_mask else None,
        "raw_mlp_chern": pred_chern,
        "confidence": confidence,
        "prob_nt": prob_nt,
        "in_bbox": in_bbox,
        "in_density": in_density,
        "ood_mask": ood_mask,
        "threshold_mask": threshold_mask,
        "flagged_for_oracle": flagged_for_oracle,
        "maha_dist": maha_dist,
        "action": "ORACLE_FHS_REQUIRED" if flagged_for_oracle else "TRUSTED_MLP_PREDICTION",
        "ood_reason": (
            "OUT_OF_BOUNDING_BOX" if not in_bbox
            else ("OUT_OF_SPECTRAL_DENSITY" if not in_density else "NONE")
        ),
    }


def batch_predict_with_domain_guard(
    model: nn.Module,
    scaler: StandardScaler,
    guard: TopoDomainGuard,
    classes: np.ndarray,
    trivial_idx: int,
    X_raw: np.ndarray,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Inferência em lote 100% vetorizada com contenção OOD e política de fallback ao oráculo.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X_raw_arr = np.asarray(X_raw, dtype=np.float32)
    X_scaled = scaler.transform(X_raw_arr).astype(np.float32)

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_scaled).to(device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    preds_idx = np.argmax(probs, axis=-1)
    preds_chern = classes[preds_idx]
    probs_nt = 1.0 - probs[:, trivial_idx]

    routing = guard.route_inference(X_raw_arr, probs_nt, threshold=threshold)

    # Predições confiáveis: mantidas apenas quando o ponto não é OOD
    trusted_preds = np.where(~routing["ood_mask"], preds_chern, np.nan)

    return {
        "preds_chern": trusted_preds,
        "raw_mlp_preds": preds_chern,
        "probs_nt": probs_nt,
        "in_bbox": routing["in_bbox"],
        "in_density": routing["in_density"],
        "ood_mask": routing["ood_mask"],
        "threshold_mask": routing["threshold_mask"],
        "flagged_for_oracle": routing["flagged_for_oracle"],
        "maha_dist": routing["maha_dist"],
        "n_total": routing["n_total"],
        "n_flagged": routing["n_flagged"],
        "n_ood_intercepted": routing["n_ood_intercepted"],
        "oracle_reduction_pct": routing["oracle_reduction_pct"],
    }


def measure_stratified_fhs_speedup(
    X_clean: np.ndarray,
    flagged_idx: np.ndarray,
    unflagged_idx: np.ndarray,
    t_screening_total_sec: float,
    k_sample: int = 1000,
    rng: Optional[np.random.Generator] = None,
    n_bootstrap: int = 1000,
    fhs_full_audit: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark Estratificado de Eficiência Computacional via Amostragem Dupla Independente.

    Mitiga Thermal Throttling, Clock Jitter e Viés de Cache através de Intercalação Estocástica:
      1. Seleciona subamostras de calibração k_1 e k_2 para os estratos Flagged e Unflagged.
      2. Cria uma lista unificada de tarefas rotuladas [(idx, True), ..., (idx, False)] e aplica rng.shuffle().
      3. Executa o loop do oráculo FHS (compute_chern_rigorous) medindo cada latência individual via time.perf_counter().
      4. Separa os tempos nas listas latencies_flagged e latencies_unflagged, calculando médias aritméticas em ms.
      5. Computa T_pure_fhs (projetado) e T_hybrid (projetado com overhead End-to-End de screening), derivando o Speedup.
      6. Avalia intervalos de incerteza via Bootstrap Não-Paramétrico (IC 95%).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_total = len(X_clean)
    n_flagged = len(flagged_idx)
    n_unflagged = len(unflagged_idx)

    M_clean = X_clean[:, 0]
    p2_clean = X_clean[:, 1]
    p3_clean = X_clean[:, 2]
    p4_clean = X_clean[:, 3]

    # ── 1. Amostragem Estratificada e Intercalação Estocástica de Tarefas ──
    k1 = min(k_sample, n_flagged) if not fhs_full_audit else n_flagged
    sample_flagged_idx = rng.choice(flagged_idx, size=k1, replace=False) if not fhs_full_audit else flagged_idx

    if n_unflagged > 0:
        k2 = min(k_sample, n_unflagged) if not fhs_full_audit else n_unflagged
        sample_unflagged_idx = rng.choice(unflagged_idx, size=k2, replace=False) if not fhs_full_audit else unflagged_idx
    else:
        k2 = 0
        sample_unflagged_idx = np.array([], dtype=int)

    # Lista unificada de tarefas estocasticamente intercaladas para prevenir thermal throttling
    tasks = [(int(idx), True) for idx in sample_flagged_idx] + [(int(idx), False) for idx in sample_unflagged_idx]
    rng.shuffle(tasks)

    # ── 2. Execução Intercalada com Cronometragem Individual ──
    latencies_flagged: List[float] = []
    latencies_unflagged: List[float] = []
    fhs_flagged_labels: List[Optional[int]] = []
    fhs_unflagged_labels: List[Optional[int]] = []

    for idx, is_flagged in tasks:
        t0 = time.perf_counter()
        c = compute_chern_rigorous(
            float(M_clean[idx]), float(p2_clean[idx]), float(p3_clean[idx]), float(p4_clean[idx]),
            N_init=60, n_occ=3
        )
        dt = time.perf_counter() - t0
        if is_flagged:
            latencies_flagged.append(dt)
            fhs_flagged_labels.append(c)
        else:
            latencies_unflagged.append(dt)
            fhs_unflagged_labels.append(c)

    arr_lat_flagged = np.array(latencies_flagged, dtype=np.float64)
    t_fhs_flagged_mean_s = float(np.mean(arr_lat_flagged)) if len(arr_lat_flagged) > 0 else 0.0
    t_fhs_flagged_std_s = float(np.std(arr_lat_flagged, ddof=1)) if len(arr_lat_flagged) > 1 else 0.0

    if len(latencies_unflagged) > 0:
        arr_lat_unflagged = np.array(latencies_unflagged, dtype=np.float64)
        t_fhs_unflagged_mean_s = float(np.mean(arr_lat_unflagged))
        t_fhs_unflagged_std_s = float(np.std(arr_lat_unflagged, ddof=1)) if len(arr_lat_unflagged) > 1 else 0.0
    else:
        arr_lat_unflagged = arr_lat_flagged
        t_fhs_unflagged_mean_s = t_fhs_flagged_mean_s
        t_fhs_unflagged_std_s = t_fhs_flagged_std_s

    t_fhs_flagged_ms = t_fhs_flagged_mean_s * 1000.0
    t_fhs_unflagged_ms = t_fhs_unflagged_mean_s * 1000.0
    latency_ratio = t_fhs_flagged_mean_s / t_fhs_unflagged_mean_s if t_fhs_unflagged_mean_s > 0 else 1.0

    # ── 3. Projeção Despoluída de Custos Totais e Speedup Realista ──
    t_fhs_audit_flagged_total_s = n_flagged * t_fhs_flagged_mean_s
    t_fhs_unflagged_pure_total_s = n_unflagged * t_fhs_unflagged_mean_s
    t_pure_fhs_total_s = t_fhs_audit_flagged_total_s + t_fhs_unflagged_pure_total_s
    t_hybrid_total_s = t_screening_total_sec + t_fhs_audit_flagged_total_s
    speedup = t_pure_fhs_total_s / t_hybrid_total_s if t_hybrid_total_s > 0 else float("inf")

    # ── 4. Incerteza Estatística via Bootstrap Não-Paramétrico (IC 95%) ──
    boot_speedups = []
    boot_ratios = []
    for _ in range(n_bootstrap):
        b_f = rng.choice(arr_lat_flagged, size=len(arr_lat_flagged), replace=True)
        b_u = rng.choice(arr_lat_unflagged, size=len(arr_lat_unflagged), replace=True)
        m_f = float(np.mean(b_f))
        m_u = float(np.mean(b_u))
        b_pure = (n_flagged * m_f) + (n_unflagged * m_u)
        b_hyb = t_screening_total_sec + (n_flagged * m_f)
        boot_speedups.append(b_pure / b_hyb if b_hyb > 0 else 0.0)
        boot_ratios.append(m_f / m_u if m_u > 0 else 1.0)

    ci_speedup = (float(np.percentile(boot_speedups, 2.5)), float(np.percentile(boot_speedups, 97.5)))
    ci_ratio = (float(np.percentile(boot_ratios, 2.5)), float(np.percentile(boot_ratios, 97.5)))

    frac_nontrivial_in_flagged = float(np.mean([c is not None and c != 0 for c in fhs_flagged_labels])) if fhs_flagged_labels else 0.0

    # ── 5. Relatório Formatado de Auditoria HPC com Rigor de Nomenclatura ──
    print(f"\n{'=' * 86}")
    print(" BENCHMARK ESTRATIFICADO DE EFICIÊNCIA COMPUTACIONAL (AMOSTRAGEM DUPLA HPC)")
    print(f"{'=' * 86}")
    print(f" Lote Total Avaliado:                          {n_total:,} pontos")
    print(f" População Flagged (Encaminhada ao Oráculo):   {n_flagged:,} pontos ({n_flagged / n_total:.2%})")
    print(f" População Unflagged (Filtrada pela MLP):      {n_unflagged:,} pontos ({n_unflagged / n_total:.2%})")
    print(f" ----------------------------------------------------------------------------------")
    print(f" [AMOSTRAGEM DUPLA INTERCALADA ESTOCASTICAMENTE (k_flagged={k1}, k_unflagged={k2})]")
    print(f" • Latência Média FHS Flagged (t_fhs_flagged):     {t_fhs_flagged_ms:.3f} ± {t_fhs_flagged_std_s * 1000:.3f} ms/pt")
    print(f" • Latência Média FHS Unflagged (t_fhs_unflagged): {t_fhs_unflagged_ms:.3f} ± {t_fhs_unflagged_std_s * 1000:.3f} ms/pt")
    print(f" • Razão de Latência (Flagged / Unflagged):      {latency_ratio:.2f}x (IC 95%: [{ci_ratio[0]:.2f}x, {ci_ratio[1]:.2f}x])")
    print(f" ----------------------------------------------------------------------------------")
    print(f" [CUSTOS DE TEMPO DE PAREDE PROJETADOS (Wall-Clock Time)]")
    print(f" • Tempo Screening Pipeline (End-to-End Total):{t_screening_total_sec * 1000:.2f} ms ({t_screening_total_sec / n_total * 1000:.6f} ms/pt)")
    print(f" • Custo Oráculo FHS Puro Projetado (T_pure):  {t_pure_fhs_total_s:.2f} s ({t_pure_fhs_total_s / 60:.2f} min)")
    print(f"     -> Parcela Flagged ({n_flagged:,} pts):           {t_fhs_audit_flagged_total_s:.2f} s ({t_fhs_audit_flagged_total_s / t_pure_fhs_total_s:.1%})")
    print(f"     -> Parcela Unflagged ({n_unflagged:,} pts):       {t_fhs_unflagged_pure_total_s:.2f} s ({t_fhs_unflagged_pure_total_s / t_pure_fhs_total_s:.1%})")
    print(f" • Custo Pipeline Híbrido Estimado (Estrat.):  {t_hybrid_total_s:.2f} s ({t_hybrid_total_s / 60:.2f} min)")
    print(f" ----------------------------------------------------------------------------------")
    print(f" [MÉTRICAS FINAIS DE GANHO DE DESEMPENHO]")
    print(f" • Redução de Chamadas ao Oráculo:             {(1.0 - n_flagged / n_total) * 100.0:.2f}%")
    print(f" • Speedup de Parede Estimado (Estratificado): {speedup:.2f}x (IC 95%: [{ci_speedup[0]:.2f}x, {ci_speedup[1]:.2f}x])")
    print(f" • Confirmação de Fases Cn!=0 na Amostra:       {frac_nontrivial_in_flagged:.2%}")
    print(f"{'=' * 86}\n")

    return {
        "n_eval": n_total,
        "n_flagged": n_flagged,
        "n_unflagged": n_unflagged,
        "oracle_reduction_pct": (1.0 - n_flagged / n_total) * 100.0,
        "t_fhs_flagged_ms": t_fhs_flagged_ms,
        "t_fhs_unflagged_ms": t_fhs_unflagged_ms,
        "latency_ratio": latency_ratio,
        "latency_ratio_ci95": ci_ratio,
        "speedup_stratified_estimated": speedup,
        "speedup_measured": speedup,
        "speedup_ci95": ci_speedup,
        "t_screening_total_s": t_screening_total_sec,
        "t_hybrid_total_s": t_hybrid_total_s,
        "t_pure_fhs_total_s": t_pure_fhs_total_s,
        "frac_nontrivial_in_sample": frac_nontrivial_in_flagged,
    }


def empirical_hybrid_pipeline(
    model: nn.Module,
    scaler: StandardScaler,
    trivial_idx: int,
    threshold: float,
    X_train_scaled: np.ndarray,
    n_eval: int = 100_000,
    fhs_calibration_sample: int = 1000,
    fhs_full_audit: bool = False,
    maha_percentile: float = 99.5,
    seed: int = 7,
) -> Optional[Dict[str, Any]]:
    """
    Avaliação empírica do pipeline híbrido (MLP como filtro de triagem + Oráculo FHS
    como auditor de alta fidelidade) em 10^5 pontos reais amostrados no espaço de parâmetros.
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    model.eval()

    print(f"\n{'=' * 86}")
    print(" PIPELINE HÍBRIDO — AVALIAÇÃO EMPÍRICA EM 10^5 PONTOS REAIS COM GUARDA OOD")
    print(f"{'=' * 86}")

    guard = TopoDomainGuard(
        bounds=_BOUNDS,
        scaler=scaler,
        X_train_scaled=X_train_scaled,
        maha_percentile=maha_percentile,
        epsilon=1e-6,
    )

    # ── Experimento A: Robustez Adversarial OOD (BBox + Covariate Shift no BBox) ──
    n_adv = 10_000
    n_adv_in_bbox_shift = 5_000
    n_adv_out_bbox = 5_000

    # 1. Pontos fora do Bounding Box físico
    features_out = []
    for feat in FEATURES:
        lo, hi = _BOUNDS[feat]
        span = hi - lo
        side = rng.choice([0, 1], size=n_adv_out_bbox)
        val_out_low = rng.uniform(lo - 0.25 * span, lo, size=n_adv_out_bbox)
        val_out_high = rng.uniform(hi, hi + 0.25 * span, size=n_adv_out_bbox)
        features_out.append(np.where(side == 0, val_out_low, val_out_high))
    X_adv_out = np.stack(features_out, axis=1).astype(np.float32)

    # 2. Pontos DENTRO do Bounding Box, mas com Covariate Shift / correlação anômala não vista
    #    (amostrados nos vértices e cantos do hipercubo onde a densidade espectral é anômala)
    M_lo, M_hi = _BOUNDS["M"]
    p_lo, p_hi = -1.0, 1.0
    corner_M = rng.choice([M_lo + 0.01, M_hi - 0.01], size=n_adv_in_bbox_shift)
    corner_p2 = rng.choice([p_lo + 0.01, p_hi - 0.01], size=n_adv_in_bbox_shift)
    corner_p3 = rng.choice([p_lo + 0.01, p_hi - 0.01], size=n_adv_in_bbox_shift)
    corner_p4 = rng.choice([p_lo + 0.01, p_hi - 0.01], size=n_adv_in_bbox_shift)
    noise = rng.normal(0, 0.02, size=(n_adv_in_bbox_shift, 4))
    X_adv_in = np.stack([corner_M, corner_p2, corner_p3, corner_p4], axis=1).astype(np.float32) + noise
    # Garante que X_adv_in permaneça estritamente dentro do BBox para testar Mahalanobis
    for j, feat in enumerate(FEATURES):
        lo, hi = _BOUNDS[feat]
        X_adv_in[:, j] = np.clip(X_adv_in[:, j], lo + 1e-4, hi - 1e-4)

    X_adv = np.vstack([X_adv_out, X_adv_in])
    rng.shuffle(X_adv)

    gate_adv = guard.check(X_adv)
    n_adv_ood = int((~gate_adv["trusted"]).sum())
    n_adv_bbox = int((~gate_adv["in_bbox"]).sum())
    n_adv_maha = int((~gate_adv["in_density"]).sum())

    print(f"\n [EXPERIMENTO A: Robustez OOD — Lote Adversarial 10k pts]")
    print(f" Pontos OOD Interceptados: {n_adv_ood:,} ({n_adv_ood / n_adv:.1%})")
    print(f"   -> Barrados por Limites Físicos (BBox): {n_adv_bbox:,}")
    print(f"   -> Barrados por Densidade Espectral (Mahalanobis p{maha_percentile:g}): {n_adv_maha:,}")

    # ── Experimento B: Eficiência Operacional em Lote Limpo (10^5 Pontos) ──
    features_list_clean = []
    for feat in FEATURES:
        lo, hi = _BOUNDS[feat]
        features_list_clean.append(rng.uniform(lo, hi, n_eval))

    X_clean = np.stack(features_list_clean, axis=1).astype(np.float32)

    # CRONOMETRAGEM END-TO-END (Inicia antes do scaler e para após criação das máscaras e vetores de índices)
    t0_screening = time.perf_counter()
    X_clean_s = scaler.transform(X_clean).astype(np.float32)

    X_tensor = torch.from_numpy(X_clean_s).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()

    probs_nt = (1.0 - probs[:, trivial_idx]).cpu().numpy()
    routing_clean = guard.route_inference(X_clean, probs_nt, threshold=threshold)
    flagged_mask_clean = routing_clean["flagged_for_oracle"]

    flagged_idx = np.where(flagged_mask_clean)[0]
    unflagged_idx = np.where(~flagged_mask_clean)[0]
    t_screening_total_s = time.perf_counter() - t0_screening

    n_flagged = int(len(flagged_idx))
    n_unflagged = int(len(unflagged_idx))
    oracle_reduction_pct = routing_clean["oracle_reduction_pct"]
    t_screening_ms_per_point = (t_screening_total_s / n_eval) * 1000.0

    print(f"\n [EXPERIMENTO B: Eficiência em Produção — Lote de {n_eval // 1000}k pts]")
    print(f" Tempo total screening pipeline (End-to-End): {t_screening_total_s * 1000:.2f} ms ({t_screening_ms_per_point:.6f} ms/pt)")
    print(f" Pontos sinalizados para auditoria FHS: {n_flagged:,} ({n_flagged / n_eval:.2%})")
    print(f" [MÉTRICA PRIMÁRIA] Redução de Chamadas ao Oráculo FHS: {oracle_reduction_pct:.2f}%")

    if n_flagged == 0:
        return None

    # Benchmark Estratificado com Amostragem Dupla Independente e Intercalação Estocástica
    strat_report = measure_stratified_fhs_speedup(
        X_clean=X_clean,
        flagged_idx=flagged_idx,
        unflagged_idx=unflagged_idx,
        t_screening_total_sec=t_screening_total_s,
        k_sample=fhs_calibration_sample,
        rng=rng,
        n_bootstrap=1000,
        fhs_full_audit=fhs_full_audit,
    )
    strat_report["adv_n_ood"] = n_adv_ood
    return strat_report


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO PRINCIPAL (ORQUESTRADOR DO PIPELINE COM ISOLAMENTO ESTRITO)
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset {CSV_PATH} não encontrado. Execute data_generator.py para gerar a base física."
        )

    # 1. CARREGAMENTO DO DATASET
    df_raw = pd.read_csv(CSV_PATH)
    X_raw = df_raw[FEATURES].values.astype(np.float32)
    y_raw = df_raw["chern"].values

    classes = np.sort(np.unique(y_raw))
    c2i = {int(c): i for i, c in enumerate(classes)}
    y = np.array([c2i[int(c)] for c in y_raw], dtype=np.int64)
    n_classes = len(classes)
    trivial_idx = int(np.where(classes == 0)[0][0])

    # 2. ISOLAMENTO OUT-OF-SAMPLE ESTRITO DO HOLDOUT EXTERNO (85% Dev / 15% Holdout)
    # O Holdout Externo é colocado em quarentena estrita antes de qualquer estimativa estatística.
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X_raw,
        y,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=GLOBAL_HOLDOUT_SEED,
        stratify=y,
    )

    # PREVALÊNCIA ESTIMADA ESTRITAMENTE SOBRE O CONJUNTO DE DESENVOLVIMENTO (Zero vazamento de Holdout)
    real_prevalence = float((y_dev != trivial_idx).mean())

    print(f"\n{'=' * 86}")
    print(" INICIALIZAÇÃO DO PIPELINE COM ISOLAMENTO OUT-OF-SAMPLE ESTRITO QUALIS A1")
    print(f"{'=' * 86}")
    print(f" Dataset carregado: {CSV_PATH} ({len(df_raw)} amostras totais)")
    print(f" Prevalência de Evento Raro induzida pelo prior uniforme de amostragem prescrito (π, calculada em Dev): {real_prevalence:.4%}")
    print(f"\n Particionamento Inicial de Isolamento Metodológico (Seed Global = {GLOBAL_HOLDOUT_SEED}):")
    print(f"   -> Conjunto de Desenvolvimento (85%): {len(y_dev):>5} amostras (opera 5x2 CV e calibrações)")
    print(f"   -> Conjunto Holdout Externo (15%):     {len(y_holdout):>5} amostras (isolado estritamente até o final)")

    # 3. PROTOCOLO 5x2 CROSS-VALIDATION NO DESENVOLVIMENTO (10 DOBRAS CONTROLADAS)
    df_runs, models_registry = run_symmetric_multiseed_evaluation(
        X_dev=X_dev,
        y_dev=y_dev,
        trivial_idx=trivial_idx,
        n_classes=n_classes,
        n_iterations=N_5X2CV_ITERATIONS,
        epochs=120,
        batch_size=256,
        lr=1e-3,
        patience=15,
    )

    # 4. INFERÊNCIA ESTATÍSTICA PAREADA 5x2 CV (Testes de Diferença Dobra a Dobra)
    print_comparative_statistical_report(df_runs, n_seeds=N_SEEDS_EVALUATION)
    df_runs.to_csv("multiseed_paired_evaluation_dev.csv", index=False)

    # 5. SELEÇÃO INDEPENDENTE E SIMÉTRICA DO MODELO DE PRODUÇÃO NA VALIDAÇÃO INTERNA
    mlp_registry = models_registry["mlp"]
    best_mlp_fold = max(
        mlp_registry.keys(),
        key=lambda s: mlp_registry[s]["selection_key"],
    )
    prod_mlp_candidate = mlp_registry[best_mlp_fold]

    rf_registry = models_registry["rf"]
    best_rf_fold = max(
        rf_registry.keys(),
        key=lambda s: rf_registry[s]["selection_key"],
    )
    prod_rf_candidate = rf_registry[best_rf_fold]

    lr_registry = models_registry["lr"]
    best_lr_fold = max(
        lr_registry.keys(),
        key=lambda s: lr_registry[s]["selection_key"],
    )
    prod_lr_candidate = lr_registry[best_lr_fold]

    baselines_production = {
        "rf": prod_rf_candidate,
        "lr": prod_lr_candidate,
    }

    print(f"\n{'=' * 86}")
    print(" CONGELAMENTO INDEPENDENTE DOS MODELOS DE PRODUÇÃO (Validação Interna 5x2 CV)")
    print(f" • MLP (Dobra {best_mlp_fold}): Estratégia = {prod_mlp_candidate['strategy']}, Limiar = {prod_mlp_candidate['threshold']:.4f}")
    print(f" • Random Forest (Dobra {best_rf_fold}): Limiar = {prod_rf_candidate['threshold']:.4f}")
    print(f" • Regressão Logística (Dobra {best_lr_fold}): Limiar = {prod_lr_candidate['threshold']:.4f}")
    print(f"{'=' * 86}")

    # 6. AVALIAÇÃO ÚNICA NO CONJUNTO HOLDOUT EXTERNO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout_results = evaluate_production_on_blind_holdout(
        production_candidate=prod_mlp_candidate,
        baselines_production=baselines_production,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        trivial_idx=trivial_idx,
        classes=classes,
        device=device,
    )

    # 7. PROJEÇÃO BAYESIANA DE PRECISÃO NO DEPLOYMENT
    mlp_ho_metrics = holdout_results["mlp_metrics"]
    proj_bayes = bayesian_precision_projection(
        test_tpr=mlp_ho_metrics["tpr"],
        test_fpr=mlp_ho_metrics["fpr"],
        real_prevalence=real_prevalence,
    )

    # 8. SIMULAÇÃO DO PIPELINE HÍBRIDO EM PRODUÇÃO (10^5 PONTOS)
    hybrid_report = empirical_hybrid_pipeline(
        model=prod_mlp_candidate["model"],
        scaler=prod_mlp_candidate["scaler"],
        trivial_idx=trivial_idx,
        threshold=prod_mlp_candidate["threshold"],
        X_train_scaled=prod_mlp_candidate["X_train_scaled"],
        n_eval=100_000,
        fhs_calibration_sample=1000,
        fhs_full_audit=False,
        maha_percentile=99.5,
        seed=7,
    )

    # 9. SÍNTESE EXECUTIVA METODOLÓGICA QUALIS A1
    sub_mlp = df_runs[df_runs["modelo"] == "MLP"]
    print(f"\n{'=' * 86}")
    print(" SÍNTESE EXECUTIVA QUALIS A1 — CONFORMIDADE METODOLÓGICA ESTRITA")
    print(f"{'=' * 86}")
    print(f" 1. Isolamento Externo: Holdout externo de 15% avaliado exatamente 1x após congelamento.")
    print(f" 2. Protocolo 5x2 CV: MLP, RF e LR avaliados nas mesmas {N_SEEDS_EVALUATION} dobras controladas.")
    print(f" 3. Estabilidade Global (Dev): Recall = {sub_mlp['recall_val'].mean():.4f} ± {sub_mlp['recall_val'].std():.4f} | "
          f"FPR = {sub_mlp['fpr_val'].mean():.4f} ± {sub_mlp['fpr_val'].std():.4f}")
    print(f" 4. Desempenho Holdout Externo: Recall = {mlp_ho_metrics['tpr']:.4f} "
          f"(IC 95%: [{holdout_results['mlp_ci95'][0]:.4f}, {holdout_results['mlp_ci95'][1]:.4f}]) | "
          f"FPR = {mlp_ho_metrics['fpr']:.4f}")
    print(f" 5. Precisão Projetada (Bayes, π={real_prevalence:.2%}): {proj_bayes['projected_precision']:.4%}")
    if hybrid_report is not None:
        speedup_est = hybrid_report.get("speedup_stratified_estimated", hybrid_report.get("speedup_measured"))
        print(f" 6. Eficiência Operacional Híbrida: Redução de Chamadas FHS = {hybrid_report['oracle_reduction_pct']:.2f}% | "
              f"Speedup Estratificado Estimado = {speedup_est:.2f}x "
              f"(IC 95%: [{hybrid_report['speedup_ci95'][0]:.2f}x, {hybrid_report['speedup_ci95'][1]:.2f}x])")
    print(f"{'=' * 86}\n")


if __name__ == "__main__":
    main()

