#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_generator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Módulos de Física e Geração de Dados:
1. Hamiltonian Engine
2. FHS Chern Integrator
3. Monte Carlo Dataset Generator
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def _hamiltonian_batch(kx_g, ky_g, Ko, h, eps2, eps3, alpha=0.5):
    phi = kx_g[:, :, None] * NN[:, 0] + ky_g[:, :, None] * NN[:, 1]
    f_k = np.exp(1j * phi).sum(axis=-1)
    H_cf = Ko * O20 + h * Jz + eps2 * O22c + eps3 * O22s
    T = I3 + alpha * Ko * (Jx + Jy)
    H_AB = f_k[:, :, None, None] * T[None, None]
    H = np.zeros((*kx_g.shape, 6, 6), dtype=complex)
    H[:, :, :3, :3] = H_cf
    H[:, :, 3:, 3:] = -H_cf
    H[:, :, :3, 3:] = H_AB
    H[:, :, 3:, :3] = H_AB.conj().transpose(0, 1, 3, 2)
    return H

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — Metodo FHS
# ══════════════════════════════════════════════════════════════════════════════

def check_gap(eigenvalues: np.ndarray, n_occ: int, tol: float = 1e-6) -> bool:
    gap_min = np.min(eigenvalues[..., n_occ] - eigenvalues[..., n_occ - 1])
    return gap_min > tol

def fhs_chern_number(H_batch: np.ndarray, n_occ: int) -> int | None:
    eigvals, psi_all = np.linalg.eigh(H_batch)
    if not check_gap(eigvals, n_occ):
        return None
        
    psi = psi_all[:, :, :, :n_occ]

    def _link(ax: int) -> np.ndarray:
        psi_fwd = np.roll(psi, -1, axis=ax)
        M = np.einsum("...ia,...ib->...ab", psi.conj(), psi_fwd)
        det_M = np.linalg.det(M)
        det_M = np.where(np.abs(det_M) < 1e-12, 1.0 + 0j, det_M)
        return det_M / np.abs(det_M)

    U1, U2 = _link(ax=0), _link(ax=1)
    U_plaquette = U1 * np.roll(U2, -1, axis=0) * np.roll(U1, -1, axis=1).conj() * U2.conj()
    F_tilde = np.angle(U_plaquette + 1e-10j)
    return int(np.round(F_tilde.sum() / (2.0 * np.pi)))

def compute_chern(Ko, h, eps2, eps3, N=60, n_occ=3):
    k1d = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    kx_g, ky_g = np.meshgrid(k1d, k1d, indexing="ij")
    H_batch = _hamiltonian_batch(kx_g, ky_g, Ko, h, eps2, eps3, alpha=0.5)
    return fhs_chern_number(H_batch, n_occ)

def test_haldane_model():
    print("Executando teste de validação do método FHS (Modelo de Haldane)...")
    N, n_occ = 30, 1
    k1d = np.linspace(0, 2 * np.pi, N, endpoint=False)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
    delta = np.array([[0.0, 1/np.sqrt(3)], [0.5, -0.5/np.sqrt(3)], [-0.5, -0.5/np.sqrt(3)]])
    v1, v2, v3 = delta[1]-delta[2], delta[2]-delta[0], delta[0]-delta[1]
    M_mass, t1, t2, phi = 0.3, 1.0, 0.1, np.pi/2
    f_k = sum(np.exp(1j * (kx * d[0] + ky * d[1])) for d in delta)
    sum_sin = sum(np.sin(kx * v[0] + ky * v[1]) for v in [v1, v2, v3])
    d_z = M_mass - 2 * t2 * np.sin(phi) * sum_sin
    H = np.zeros((N, N, 2, 2), dtype=complex)
    H[:, :, 0, 0] = d_z
    H[:, :, 1, 1] = -d_z
    H[:, :, 0, 1] = t1 * f_k
    H[:, :, 1, 0] = t1 * f_k.conj()
    chern_val = fhs_chern_number(H, n_occ)
    assert chern_val in [1, -1], f"FALHA! Obtido C = {chern_val}."
    print(f"Sucesso! C = {chern_val}.\n")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — MONTE CARLO DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = Path("topological_dataset.csv")

_BOUNDS = {"Ko": (0.0, 3.0), "h": (-5.0, 5.0), "eps2": (0.0, 1.5), "eps3": (0.0, 1.5)}

def generate_dataset(n_samples=5000, N_bz=60, n_occ=3, seed=42, out=CSV_PATH):
    rng = np.random.default_rng(seed)
    valid_rows = []
    print(f"Gerando dataset com {n_samples} amostras topologicamente válidas...")
    pbar = tqdm(total=n_samples, desc="FHS Integrator")

    while len(valid_rows) < n_samples:
        batch_size = min(500, n_samples - len(valid_rows) + 200)
        Ko_v = rng.uniform(*_BOUNDS["Ko"], batch_size)
        h_v = rng.uniform(*_BOUNDS["h"], batch_size)
        eps2_v = rng.uniform(*_BOUNDS["eps2"], batch_size)
        eps3_v = rng.uniform(*_BOUNDS["eps3"], batch_size)

        for i in range(batch_size):
            if len(valid_rows) >= n_samples: break
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

if __name__ == "__main__":
    test_haldane_model()
    generate_dataset(n_samples=5000, N_bz=60, n_occ=3, seed=42)