#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_generator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Módulos de Física Numérica e Oráculo FHS (Refatorado & Rigoroso):
1. Bulk Hamiltonian Engine (Qi-Wu-Zhang)
2. FHS Chern Integrator (com verificação adaptativa de gap, log de singularidade e trava inteira)
3. Monte Carlo Dataset Generator (com prevalência física real e balanceamento seguro)
"""

import logging
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuração de Logger para auditoria de singularidades e fechamento de gap
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("FHS_Oracle")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — BULK HAMILTONIAN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _hamiltonian_batch(kx_g, ky_g, M, p2, p3, p4):
    N1, N2 = kx_g.shape
    H = np.zeros((N1, N2, 6, 6), dtype=np.complex128)
    sx = np.sin(kx_g) + p2 * 0.1
    sy = np.sin(ky_g) + p3 * 0.1
    sz = M + np.cos(kx_g) + np.cos(ky_g) + p4 * 0.1
    H[:, :, 2, 2] = sz
    H[:, :, 3, 3] = -sz
    H[:, :, 2, 3] = sx - 1j * sy
    H[:, :, 3, 2] = sx + 1j * sy
    H[:, :, 0, 0] = 10.0
    H[:, :, 1, 1] = 10.0
    H[:, :, 4, 4] = -10.0
    H[:, :, 5, 5] = -10.0
    return H


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — GEOMETRIA DA ZONA DE BRILLOUIN + MÉTODO FHS RIGOROSO
# ══════════════════════════════════════════════════════════════════════════════

GAP_TOL: float = 1e-6

# Limiar absoluto de gap.
# Substitui a janela multiplicativa estreita
#     GAP_TOL < gap_min < 5*GAP_TOL
# porque o mínimo verdadeiro pode ocorrer entre pontos da malha grosseira.
# Calibração registrada na auditoria:
# apenas 2,75% dos pontos da varredura apresentaram gap_min(N=60) < 0.08.
REFINE_GAP_ABS: float = 0.08

# Critério secundário:
# concentração do fluxo de Berry nas plaquetas de maior |F_tilde|.
FLUX_TOPK: int = 4
FLUX_FRAC_THRESH: float = 0.10


def bz_grid(N: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(N, dtype=np.float64)
    f = n * (2.0 * np.pi / N)
    kx, ky = np.meshgrid(f, f, indexing="ij")
    return kx, ky


def fhs_chern_number(
    H_batch: np.ndarray,
    n_occ: int,
    quant_tol: float = 1e-2,
    gap_tol: float = GAP_TOL,
    return_diag: bool = False,
):
    """
    Calcula o número de Chern via método Fukui-Hatsugai-Suzuki (FHS).

    Travas implementadas:
    1. Verificação explícita do gap entre a banda ocupada mais alta e a primeira banda desocupada.
    2. Rejeição do ponto quando: gap_min <= gap_tol
    3. Detecção de matrizes de overlap praticamente singulares: |det(M)| < 1e-12
    4. Validação da quantização inteira: |C_raw - round(C_raw)| <= quant_tol

    Se return_diag=True, retorna: (chern, gap_min, F_tilde)
    Caso contrário, retorna apenas: chern
    ou None caso o ponto não possa receber um rótulo topológico confiável.
    """
    eigvals, psi_all = np.linalg.eigh(H_batch)

    # Gap direto entre a primeira banda não ocupada e a última banda ocupada.
    gap = eigvals[..., n_occ] - eigvals[..., n_occ - 1]
    gap_min = float(np.min(gap))

    if gap_min <= gap_tol:
        if return_diag:
            return None, gap_min, None
        return None

    # Subespaço ocupado.
    psi = psi_all[:, :, :, :n_occ]
    singular_detected = False

    def _link(ax: int) -> np.ndarray:
        """
        Determinante do overlap não-Abeliano entre os subespaços ocupados em
        pontos vizinhos da malha.
        """
        nonlocal singular_detected
        psi_fwd = np.roll(psi, -1, axis=ax)
        M = np.einsum("...ia,...ib->...ab", psi.conj(), psi_fwd)
        det_M = np.linalg.det(M)
        abs_det = np.abs(det_M)

        if np.any(abs_det < 1e-12):
            singular_detected = True
            logger.debug("Singularidade detectada em _link: |det(M)| < 1e-12.")

        return det_M / (abs_det + 1e-15)

    U1 = _link(ax=0)
    U2 = _link(ax=1)

    if singular_detected:
        if return_diag:
            return None, gap_min, None
        return None

    U_plaquette = U1 * np.roll(U2, -1, axis=0) * np.roll(U1, -1, axis=1).conj() * U2.conj()
    F_tilde = np.angle(U_plaquette + 1e-10j)
    
    C_raw = F_tilde.sum() / (2.0 * np.pi)
    C_round = np.round(C_raw)
    quant_error = abs(C_raw - C_round)

    if quant_error > quant_tol:
        logger.debug(f"Violação de quantização inteira: C_raw={C_raw:.4f}, desvio={quant_error:.4e}")
        if return_diag:
            return None, gap_min, F_tilde
        return None

    C = int(C_round)

    if return_diag:
        return C, gap_min, F_tilde
    return C


def _needs_mesh_refinement(gap_min: float, F_tilde: np.ndarray | None) -> tuple[bool, str | None]:
    """
    Critério duplo para decidir se a malha grosseira deve ser substituída pela malha N_high.
    Critério 1: gap_min < REFINE_GAP_ABS
    Critério 2: fração do fluxo absoluto concentrada nas FLUX_TOPK maiores plaquetas superior a FLUX_FRAC_THRESH.
    """
    if gap_min < REFINE_GAP_ABS:
        return (True, "gap_absoluto")

    if F_tilde is not None:
        flat = np.abs(F_tilde).ravel()
        topk_flux = np.sort(flat)[-FLUX_TOPK:].sum()
        total_flux = flat.sum()
        topk_frac = topk_flux / total_flux

        if topk_frac > FLUX_FRAC_THRESH:
            return (True, "concentracao_de_fluxo_berry")

    return (False, None)


def compute_chern_rigorous(M: float, p2: float, p3: float, p4: float, N_init: int = 60, N_high: int = 120, n_occ: int = 3) -> int | None:
    """
    Calcula o número de Chern sobre a geometria correta da zona de Brillouin.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Malha inicial
    # ─────────────────────────────────────────────────────────────────────────
    kx0, ky0 = bz_grid(N_init)
    H0 = _hamiltonian_batch(kx0, ky0, M, p2, p3, p4)
    C0, gap0, F0 = fhs_chern_number(H0, n_occ, return_diag=True)

    refine, _reason = _needs_mesh_refinement(gap0, F0)

    if not refine:
        return C0

    # ─────────────────────────────────────────────────────────────────────────
    # Malha refinada
    # ─────────────────────────────────────────────────────────────────────────
    kx1, ky1 = bz_grid(N_high)
    H1 = _hamiltonian_batch(kx1, ky1, M, p2, p3, p4)
    C1, _gap1, _F1 = fhs_chern_number(H1, n_occ, return_diag=True)

    return C1


def test_haldane_model():
    """
    Teste independente de regressão do integrador FHS usando o modelo de
    Haldane em regime topológico.
    """
    print("Executando teste de validação do método FHS (Modelo de Haldane)...")

    N = 30
    n_occ = 1

    k1d = np.linspace(0, 2 * np.pi, N, endpoint=False)
    kx, ky = np.meshgrid(k1d, k1d, indexing="ij")

    delta = np.array([
        [0.0, 1 / np.sqrt(3)],
        [0.5, -0.5 / np.sqrt(3)],
        [-0.5, -0.5 / np.sqrt(3)],
    ])

    v1 = delta[1] - delta[2]
    v2 = delta[2] - delta[0]
    v3 = delta[0] - delta[1]

    M_mass = 0.3
    t1 = 1.0
    t2 = 0.1
    phi = np.pi / 2

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
# MODULE 3 — GERADOR DE DATASET COM DISTRIBUIÇÃO FÍSICA REAL
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = Path("topological_dataset.csv")

_BOUNDS = {
    "M": (1.95, 6.95),      # Fase topológica (C!=0) ocorre se 0 < M < 2. Prevalência garantida de ~1%.
    "p2": (-1.0, 1.0),      # Perturbações/ruídos locais
    "p3": (-1.0, 1.0),
    "p4": (-1.0, 1.0),
}


def generate_dataset(n_samples=5000, N_bz=60, n_occ=3, seed=42, out=CSV_PATH):
    """
    Gera dataset de fases topológicas amostrando aleatoriamente o espaço R^4.
    A prevalência das classes é determinada diretamente pelo oráculo FHS.
    """
    rng = np.random.default_rng(seed)
    valid_rows = []

    print(f"Gerando dataset com {n_samples} amostras topologicamente válidas...")
    pbar = tqdm(total=n_samples, desc="FHS Integrator Rigoroso")

    while len(valid_rows) < n_samples:
        batch_size = min(500, n_samples - len(valid_rows) + 200)

        M_v = rng.uniform(*_BOUNDS["M"], batch_size)
        p2_v = rng.uniform(*_BOUNDS["p2"], batch_size)
        p3_v = rng.uniform(*_BOUNDS["p3"], batch_size)
        p4_v = rng.uniform(*_BOUNDS["p4"], batch_size)

        for i in range(batch_size):
            if len(valid_rows) >= n_samples:
                break

            c = compute_chern_rigorous(M_v[i], p2_v[i], p3_v[i], p4_v[i], N_init=N_bz, n_occ=n_occ)

            if c is not None:
                valid_rows.append((M_v[i], p2_v[i], p3_v[i], p4_v[i], c))
                pbar.update(1)

    pbar.close()

    df = pd.DataFrame(valid_rows, columns=["M", "p2", "p3", "p4", "chern"])
    df.to_csv(out, index=False)

    print(f"\nDataset gerado -> {out}")
    print("Distribuição Real das Classes Topológicas (Chern):")
    counts = df["chern"].value_counts().sort_index()
    print(counts.to_string())

    non_trivial = (df["chern"] != 0).sum()
    print(f"Prevalência Real de Fases Não-Triviais (C != 0): {non_trivial / len(df):.2%}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3b — GERAÇÃO PARALELA
# ══════════════════════════════════════════════════════════════════════════════

from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def _generate_chunk(n_target: int, N_bz: int, n_occ: int, entropy: int, worker_id: int):
    """
    Executado em processo separado.
    O oráculo físico compute_chern_rigorous não é modificado; apenas distribuído entre processos.
    Cada processo recebe um stream aleatório derivado de SeedSequence.spawn.
    """
    rng = np.random.default_rng(entropy)
    rows = []

    while len(rows) < n_target:
        batch = min(500, n_target - len(rows) + 200)

        M_v = rng.uniform(*_BOUNDS["M"], batch)
        p2_v = rng.uniform(*_BOUNDS["p2"], batch)
        p3_v = rng.uniform(*_BOUNDS["p3"], batch)
        p4_v = rng.uniform(*_BOUNDS["p4"], batch)

        for i in range(batch):
            if len(rows) >= n_target:
                break

            c = compute_chern_rigorous(M_v[i], p2_v[i], p3_v[i], p4_v[i], N_init=N_bz, n_occ=n_occ)

            if c is not None:
                rows.append((M_v[i], p2_v[i], p3_v[i], p4_v[i], c))

    return worker_id, rows


def generate_dataset_parallel(n_samples=50_000, N_bz=60, n_occ=3, seed=42, out=CSV_PATH, n_workers=None):
    """
    Versão paralela do gerador de dataset.
    A física e o oráculo são idênticos aos utilizados por generate_dataset().
    Os streams pseudoaleatórios individuais são derivados de SeedSequence.spawn.
    """
    n_workers = n_workers or os.cpu_count() or 1
    per_worker = [n_samples // n_workers] * n_workers

    for i in range(n_samples % n_workers):
        per_worker[i] += 1

    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(n_workers)
    entropies = [int(child.generate_state(1)[0]) for child in child_sequences]

    print(f"Gerando {n_samples} amostras em {n_workers} processo(s)...")
    t0 = time.perf_counter()
    by_worker = {}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_generate_chunk, per_worker[w], N_bz, n_occ, entropies[w], w)
            for w in range(n_workers)
        ]

        for future in as_completed(futures):
            worker_id, rows = future.result()
            print(f"  worker {worker_id}: {len(rows)} amostras validas")
            by_worker[worker_id] = rows

    all_rows = []
    for w in range(n_workers):
        all_rows.extend(by_worker[w])

    dt = time.perf_counter() - t0

    df = pd.DataFrame(all_rows, columns=["M", "p2", "p3", "p4", "chern"])
    df.to_csv(out, index=False)

    print(f"\n{len(df)} amostras -> {out} ({dt:.1f}s, {dt / len(df) * 1000:.2f} ms/amostra efetivo)")

    counts = df["chern"].value_counts().sort_index()
    print(counts.to_string())

    prevalence = (df["chern"] != 0).sum() / len(df)
    print(f"Prevalencia nao-trivial: {prevalence:.2%}")

    return df


if __name__ == "__main__":
    generate_dataset_parallel(n_samples=50_000, N_bz=60, n_occ=3, seed=42)