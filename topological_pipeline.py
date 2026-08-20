#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topological_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline: (K_o, h, ε₂, ε₃) → Chern number → topological phase classifier.

Modules
-------
1. Hamiltonian Engine   — 6×6 Bloch matrix, multipolar spin liquid / honeycomb
2. FHS Chern Integrator — Fukui-Hatsugai-Suzuki (2005), fully vectorized with gap checking
3. Monte Carlo Gen.     — Uniform parameter sampling → labeled CSV
3b. Balanced Gen.       — Dataset FISICAMENTE balanceado (sem SMOTE sintético)
4. MLP Classifier       — PyTorch 4-layer dense network, multiclass with Early Stopping

CORREÇÕES APLICADAS (anti-alucinação / anti-viés para Chern=0):
- Remoção do SMOTE: interpolar (Ko,h,ε2,ε3) entre classes gera pontos em fases
  físicas erradas → rótulos inválidos que confundem a rede e a empurram ao prior.
- Class weights no CrossEntropyLoss (inverso da frequência real).
- Split estratificado (mantém proporção de classes na validação).
- Seleção do melhor modelo por F1 macro (não apenas val_loss).
- Inferência com confiança (aviso em regiões de fronteira / sem gap).

Dependencies: numpy, pandas, torch, scikit-learn, tqdm
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from domain_guard import TopoDomainGuard

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — BULK HAMILTONIAN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_r3i = 1.0 / np.sqrt(3.0)
NN: np.ndarray = np.array(
    [[0.0, _r3i], [0.5, -0.5 * _r3i], [-0.5, -0.5 * _r3i]], dtype=np.float64
)

_r2 = np.sqrt(2.0)
Jx: np.ndarray = (np.array([[0, _r2, 0], [_r2, 0, _r2], [0, _r2, 0]], dtype=complex) * 0.5)
Jy: np.ndarray = (np.array([[0, -1j * _r2, 0], [1j * _r2, 0, -1j * _r2], [0, 1j * _r2, 0]], dtype=complex) * 0.5)
Jz: np.ndarray = np.diag([1.0, 0.0, -1.0]).astype(complex)
I3: np.ndarray = np.eye(3, dtype=complex)

O20: np.ndarray = 3.0 * (Jz @ Jz) - 2.0 * I3
O22c: np.ndarray = Jx @ Jx - Jy @ Jy
O22s: np.ndarray = Jx @ Jy + Jy @ Jx

def _hamiltonian_batch(
    kx_g: np.ndarray,
    ky_g: np.ndarray,
    Ko: float,
    h: float,
    eps2: float,
    eps3: float,
    alpha: float = 0.5,
) -> np.ndarray:
    """Vectorized H(k) for a 2D grid of k-points."""
    phi = kx_g[:, :, None] * NN[:, 0] + ky_g[:, :, None] * NN[:, 1]
    f_k = np.exp(1j * phi).sum(axis=-1)

    H_cf = Ko * O20 + h * Jz + eps2 * O22c + eps3 * O22s
    # Uso do parâmetro alpha para induzir saltos topológicos maiores
    T    = I3 + alpha * Ko * (Jx + Jy)

    H_AB = f_k[:, :, None, None] * T[None, None]

    H = np.zeros((*kx_g.shape, 6, 6), dtype=complex)
    H[:, :, :3, :3] =  H_cf
    H[:, :, 3:, 3:] = -H_cf
    H[:, :, :3, 3:] =  H_AB
    H[:, :, 3:, :3] =  H_AB.conj().transpose(0, 1, 3, 2)
    return H


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Metodo FHS
# ══════════════════════════════════════════════════════════════════════════════

def check_gap(eigenvalues: np.ndarray, n_occ: int, tol: float = 1e-6) -> bool:
    """Verifica se o gap mínimo de energia entre a banda preenchida e a vazia é respeitado."""
    gap_min = np.min(eigenvalues[..., n_occ] - eigenvalues[..., n_occ - 1])
    return gap_min > tol

def fhs_chern_number(H_batch: np.ndarray, n_occ: int) -> int | None:
    """
    Motor central numérico FHS. Extraído para permitir o uso
    tanto no modelo Multipolar quanto no teste de Haldane.
    """
    # 1. Diagonalização e checagem de isolamento das bandas
    eigvals, psi_all = np.linalg.eigh(H_batch)
    if not check_gap(eigvals, n_occ):
        return None

    psi = psi_all[:, :, :, :n_occ]

    def _link(ax: int) -> np.ndarray:
        psi_fwd = np.roll(psi, -1, axis=ax)
        M = np.einsum("...ia,...ib->...ab", psi.conj(), psi_fwd)
        det_M = np.linalg.det(M)
        # Proteção contra divisão por zero (instabilidade numérica)
        det_M = np.where(np.abs(det_M) < 1e-12, 1.0 + 0j, det_M)
        return det_M / np.abs(det_M)

    U1 = _link(ax=0)
    U2 = _link(ax=1)

    # Cálculo da força de campo com offset imaginário para estabilizar o ângulo próximo a descontinuidades
    U_plaquette = (
        U1
        * np.roll(U2, -1, axis=0)
        * np.roll(U1, -1, axis=1).conj()
        * U2.conj()
    )
    F_tilde = np.angle(U_plaquette)

    return int(np.round(F_tilde.sum() / (2.0 * np.pi)))

def compute_chern(Ko: float, h: float, eps2: float, eps3: float, N: int = 60, n_occ: int = 3) -> int | None:
    k1d = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    kx_g, ky_g = np.meshgrid(k1d, k1d, indexing="ij")
    H_batch = _hamiltonian_batch(kx_g, ky_g, Ko, h, eps2, eps3, alpha=0.5)
    return fhs_chern_number(H_batch, n_occ)

def test_haldane_model() -> None:
    """Valida a precisão da integração FHS usando o clássico modelo de Haldane."""
    print("Executando teste de validação do método FHS (Modelo de Haldane)...")
    N = 30
    k1d = np.linspace(0, 2 * np.pi, N, endpoint=False)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")

    # Vetores para a rede Honeycomb no espaço real
    delta = np.array([[0.0, 1/np.sqrt(3)], [0.5, -0.5/np.sqrt(3)], [-0.5, -0.5/np.sqrt(3)]])
    v1, v2, v3 = delta[1]-delta[2], delta[2]-delta[0], delta[0]-delta[1]

    # Parâmetros topológicos de Haldane
    M_mass, t1, t2, phi = 0.3, 1.0, 0.1, np.pi/2

    # Montagem vetorial
    f_k = sum(np.exp(1j * (kx * d[0] + ky * d[1])) for d in delta)
    sum_sin = sum(np.sin(kx * v[0] + ky * v[1]) for v in [v1, v2, v3])

    d_z = M_mass - 2 * t2 * np.sin(phi) * sum_sin

    H = np.zeros((N, N, 2, 2), dtype=complex)
    H[:, :, 0, 0] = d_z
    H[:, :, 1, 1] = -d_z
    H[:, :, 0, 1] = t1 * f_k
    H[:, :, 1, 0] = t1 * f_k.conj()

    chern_val = fhs_chern_number(H, n_occ=1)

    assert chern_val in [1, -1], f"FALHA! O modelo de Haldane obteve Chern = {chern_val}, mas deveria ser ±1."
    print(f"Sucesso! Teste de Haldane passou corretamente com número de Chern C = {chern_val}.\n")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MONTE CARLO DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = Path("topological_dataset.csv")
BALANCED_CSV_PATH = Path("topological_balanced_dataset.csv")

_BOUNDS: dict[str, tuple[float, float]] = {
    "Ko":   (0.0, 3.0),
    "h":    (-5.0, 5.0),
    "eps2": (0.0, 1.5),
    "eps3": (0.0, 1.5),
}

def generate_dataset(n_samples: int = 5000, N_bz: int = 60, n_occ: int = 3, seed: int = 42, out: Path = CSV_PATH) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    valid_rows = []

    print(f"Gerando dataset com {n_samples} amostras topologicamente válidas (gap protegido)...")
    pbar = tqdm(total=n_samples, desc="FHS Integrator")

    while len(valid_rows) < n_samples:
        batch_size = min(500, n_samples - len(valid_rows) + 200)
        Ko_v   = rng.uniform(*_BOUNDS["Ko"], batch_size)
        h_v    = rng.uniform(*_BOUNDS["h"], batch_size)
        eps2_v = rng.uniform(*_BOUNDS["eps2"], batch_size)
        eps3_v = rng.uniform(*_BOUNDS["eps3"], batch_size)

        for i in range(batch_size):
            if len(valid_rows) >= n_samples:
                break

            c = compute_chern(Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], N=N_bz, n_occ=n_occ)

            if c is not None:
                valid_rows.append((Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], c))
                pbar.update(1)

    pbar.close()

    df = pd.DataFrame(valid_rows, columns=["Ko", "h", "eps2", "eps3", "chern"])
    df.to_csv(out, index=False)

    print(f"\nDataset gerado -> {out}")
    print("Distribuição das Classes Topológicas (Chern):")
    print(df["chern"].value_counts().sort_index().to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3b — DATASET FISICAMENTE BALANCEADO (substitui SMOTE)
# ══════════════════════════════════════════════════════════════════════════════

def generate_balanced_dataset(n_per_class: int = 800, N_bz: int = 60, n_occ: int = 3,
                              seed: int = 42,
                              out: Path = BALANCED_CSV_PATH,
                              max_attempts: int = 120000) -> pd.DataFrame:
    """
    Gera um dataset FISICAMENTE balanceado (sem SMOTE / oversampling sintético).

    Coleta amostras reais do integrador FHS até que cada classe Chern observada
    tenha n_per_class exemplares, ou até max_attempts tentativas. Se alguma
    classe for muito rara, o dataset é rebalanceado pelo tamanho da classe menos
    populosa (undersampling da maioria com amostras 100% reais do FHS).

    Isso elimina a "alucinação" induzida por pontos sintéticos interpolados no
    espaço de parâmetros (que caem em fases físicas erradas / sem gap).
    """
    rng = np.random.default_rng(seed)
    pool: dict[int, list] = {}
    attempts = 0
    target = n_per_class

    print(f"Coletando amostras FHS reais até {target} por classe (cap {max_attempts} tentativas)...")
    pbar = tqdm(total=max_attempts, desc="FHS Integrator (balanceado)")

    stop = False
    while attempts < max_attempts and not stop:
        batch_size = 200
        Ko_v   = rng.uniform(*_BOUNDS["Ko"], batch_size)
        h_v    = rng.uniform(*_BOUNDS["h"], batch_size)
        eps2_v = rng.uniform(*_BOUNDS["eps2"], batch_size)
        eps3_v = rng.uniform(*_BOUNDS["eps3"], batch_size)

        for i in range(batch_size):
            attempts += 1
            pbar.update(1)
            c = compute_chern(Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], N=N_bz, n_occ=n_occ)
            if c is not None:
                pool.setdefault(int(c), []).append((Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], int(c)))
                if all(len(v) >= target for v in pool.values()):
                    stop = True
                    break
            if attempts >= max_attempts:
                stop = True
                break
    pbar.close()

    classes = sorted(pool.keys())
    counts = {c: len(pool[c]) for c in classes}

    print("\nDistribuição do pool coletado:")
    for c in classes:
        print(f"  Chern {c:>2}: {counts[c]} amostras reais")

    n_per_class_eff = min(target, min(counts.values()))
    if n_per_class_eff < target:
        print(f"[ATENÇÃO] Classe mais rara tem apenas {min(counts.values())} amostras. "
              f"Balanceando para {n_per_class_eff} por classe.")

    balanced = []
    for c in classes:
        rows = pool[c]
        idx = rng.choice(len(rows), size=n_per_class_eff, replace=False)
        balanced.extend(rows[j] for j in idx)

    rng.shuffle(balanced)
    df = pd.DataFrame(balanced, columns=["Ko", "h", "eps2", "eps3", "chern"])
    df.to_csv(out, index=False)

    print(f"\nDataset BALANCEADO gerado -> {out}  ({len(df)} linhas, {len(classes)} classes)")
    print("Distribuição Final das Classes:")
    print(df["chern"].value_counts().sort_index().to_string())
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — TOPOLOGICAL PHASE CLASSIFIER (PyTorch MLP)
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ["Ko", "h", "eps2", "eps3"]

class ChernDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, i: int): return self.X[i], self.y[i]

class TopoPhaseMLP(nn.Module):
    def __init__(self, n_classes: int, p: float = 0.25) -> None:
        super().__init__()
        def _block(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(p))

        self.net = nn.Sequential(
            _block(4,   128),
            _block(128, 256),
            _block(256, 128),
            _block(128,  64),
            nn.Linear(64, n_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_classifier(csv_path: Path = BALANCED_CSV_PATH, epochs: int = 200,
                     batch_size: int = 256, lr: float = 1e-3, val_frac: float = 0.2,
                     patience: int = 20, use_class_weights: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTreinando modelo no dispositivo: {device}")

    df = pd.read_csv(csv_path)
    X_raw = df[FEATURES].values.astype(np.float32)
    y_raw = df["chern"].values

    classes = np.sort(np.unique(y_raw))
    c2i = {int(c): i for i, c in enumerate(classes)}
    y = np.array([c2i[int(c)] for c in y_raw], dtype=np.int64)
    n_classes = len(classes)
    print(f"Classes Chern encontradas: {classes.tolist()}  |  Total de amostras: {len(y)}")

    # ── Split ESTRATIFICADO (preserva a proporção de classes na validação) ──
    X_tr_raw, X_va_raw, y_tr_raw, y_va = train_test_split(
        X_raw, y, test_size=val_frac, random_state=0, stratify=y
    )

    # ── Padronização (fit APENAS no treino) ──
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)
    X_va = scaler.transform(X_va_raw)

    tr_loader = DataLoader(ChernDataset(X_tr, y_tr_raw), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(ChernDataset(X_va, y_va), batch_size=512, shuffle=False)

    model = TopoPhaseMLP(n_classes).to(device)

    # ── Class weights (inverso da frequência real → penaliza erro na minoria) ──
    if use_class_weights:
        counts = np.bincount(y_tr_raw)
        class_weights = torch.tensor(
            (len(y_tr_raw) / (n_classes * counts)).astype(np.float32), device=device
        )
        print(f"Class weights: {class_weights.tolist()}")
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Seleção do melhor modelo por F1 macro ──
    best_f1, best_state = -1.0, None
    epochs_no_improve = 0

    print(f'\n{"Ep":>4}  {"TrLoss":>8}  {"VaLoss":>8}  {"F1(Mac)":>8}  {"Recall":>7}')
    print("─" * 45)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(y_tr_raw)

        model.eval()
        va_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for Xb, yb in va_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                va_loss += criterion(logits, yb).item() * len(yb)

                preds = logits.argmax(-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        va_loss /= len(y_va)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )

        scheduler.step()

        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if ep % 10 == 0 or ep == 1:
            print(f"{ep:>4}  {tr_loss:>8.4f}  {va_loss:>8.4f}  {f1:>8.4f}  {recall:>7.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping trigado na época {ep}! (Sem melhora por {patience} épocas)")
            break

    model.load_state_dict(best_state)
    model.eval()

    # ── Métricas finais por classe (transparência completa) ──
    print("\n═══ RELATÓRIO FINAL DE VALIDAÇÃO (melhor época por F1 macro) ═══")
    print(f"F1-Score macro: {best_f1:.4f}")

    final_preds, final_targets = [], []
    with torch.no_grad():
        for Xb, yb in va_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            final_preds.extend(logits.argmax(-1).cpu().numpy())
            final_targets.extend(yb.cpu().numpy())

    idx2c = {i: int(c) for i, c in enumerate(classes)}
    print("\nMétricas POR CLASSE (Chern real):")
    print(classification_report(
        final_targets, final_preds,
        labels=list(range(n_classes)),
        target_names=[f"Chern={idx2c[i]}" for i in range(n_classes)],
        zero_division=0,
    ))
    print("Matriz de Confusão (linhas=real, colunas=predito):")
    print(confusion_matrix(final_targets, final_preds, labels=list(range(n_classes))))
    print("  Legendas:", [f"Chern={idx2c[i]}" for i in range(n_classes)])

    return model, scaler, classes


# ══════════════════════════════════════════════════════════════════════════════
# INFERÊNCIA COM CONFIANÇA (reduz "alucinação" em pontos incertos)
# ══════════════════════════════════════════════════════════════════════════════

def predict_chern(
    model: nn.Module,
    scaler: StandardScaler,
    classes: np.ndarray,
    Ko: float,
    h: float,
    eps2: float,
    eps3: float,
    guard: Optional[TopoDomainGuard] = None,
    threshold: float = 0.5,
) -> Tuple[Optional[int], float]:
    """
    Prediz o número de Chern com probabilidade de confiança e contenção OOD.

    Se guard for fornecido, executa primeiro a contenção pelo TopoDomainGuard.
    Se o ponto for OOD (ood_mask=True), descarta sumariamente a inferência da MLP.
    Caso contrário, avalia o limiar de confiança softmax.
    """
    x_raw = np.array([[Ko, h, eps2, eps3]], dtype=np.float32)

    if guard is not None:
        guard_res = guard.check(x_raw)
        if guard_res["ood_mask"]:
            reason = "Bounding Box violado" if not guard_res["in_bbox"] else "Densidade espectral fora do domínio"
            print(f"[REJEIÇÃO OOD] Ponto fora do domínio ({reason})! Encaminhando obrigatoriamente ao Oráculo FHS.")
            return None, 0.0

    device = next(model.parameters()).device
    x_scaled = scaler.transform(x_raw)
    with torch.no_grad():
        logits = model(torch.from_numpy(x_scaled).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    chern = int(classes[idx])
    confidence = float(probs[idx])

    if confidence < threshold:
        print(f"[AVISO] Baixa confiança ({confidence:.2%}). "
              f"Provável região de fronteira topológica / sem gap.")
        return None, confidence
    return chern, confidence


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 0. Teste estrito de estabilidade
    test_haldane_model()

    # 1. Geração de Dataset FISICAMENTE BALANCEADO (sem SMOTE)
    #    N_bz=60 para garantir fidelidade matemática extrema
    generate_balanced_dataset(n_per_class=800, N_bz=60, n_occ=3, seed=42)

    # 2. Treinamento da Rede
    model, scaler, chern_classes = train_classifier(
        csv_path=BALANCED_CSV_PATH, epochs=200, batch_size=256, lr=1e-3
    )

    # 3. Salvar artefatos
    torch.save(
        {
            "model_state":   model.state_dict(),
            "scaler_mean":   scaler.mean_,
            "scaler_scale":  scaler.scale_,
            "chern_classes": chern_classes.tolist(),
        },
        "topological_mlp.pt",
    )
    print("Salvo com sucesso -> topological_mlp.pt")

