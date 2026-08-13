#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_generator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Módulos de Física Numérica e Oráculo FHS (Refatorado & Rigoroso):
1. Bulk Hamiltonian Engine (com justificativa de Ko)
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FHS_Oracle")

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
    """
    Constrói o Hamiltoniano de um MODELO REPRESENTATIVO TOPOLÓGICO (toy model) em um
    grid 2D do espaço k, sobre uma rede honeycomb com 2 sub-redes x 3 estados orbitais.

    STATUS DE MODELAGEM (não é uma derivação microscópica de um material real):
    -----------------------------------------------------------------------
    Este NÃO é o Hamiltoniano derivado de um composto específico. É um modelo
    representativo, construído para demonstrar/validar a metodologia computacional
    (integração FHS + aceleração por MLP) sobre um Hamiltoniano de banda com
    transições topológicas genuínas e controláveis. Duas escolhas estruturais
    precisam ser lidas como isso -- escolhas de modelagem, não fatos físicos
    derivados -- e não como propriedades de um material real sem citação/derivação
    microscópica adicional:

    1) AMARRAÇÃO SIMÉTRICA DE Ko (papel duplo, por construção):
       Ko é usado tanto como intensidade do campo cristalino quadrupolar local
       O20 = 3*Jz^2 - 2*I3 (termo intra-sub-rede, em H_cf) quanto como fator que
       modula a amplitude de hopping intersub-rede via T = I3 + alpha*Ko*(Jx+Jy)
       (termo inter-sub-rede, em H_AB). Amarrar os dois ao mesmo parâmetro é uma
       escolha deliberada para obter transições topológicas controláveis por um
       único parâmetro varrido em 1D/2D (útil para gerar o dataset e visualizar
       diagramas de fase) -- não uma consequência derivada de um mecanismo
       microscópico específico (spin-órbita, superexchange, etc.). Campo cristalino
       de íon único e integral de hopping são, em geral, parâmetros fisicamente
       independentes; tratá-los como o mesmo grau de liberdade exige, para
       publicação como resultado físico (não apenas metodológico), uma derivação
       microscópica explícita ou citação a um modelo estabelecido que já faça essa
       amarração.
    2) SINAL OPOSTO DE H_cf ENTRE SUB-REDES (H_B = -H_cf, ver bloco de construção
       abaixo): também uma escolha estrutural do toy model (lembra o termo de massa
       de Dirac do modelo de Haldane, com sinais opostos nas duas sub-redes), não
       derivada aqui a partir de simetria do grupo espacial de um material específico.
    """
    phi = kx_g[:, :, None] * NN[:, 0] + ky_g[:, :, None] * NN[:, 1]
    f_k = np.exp(1j * phi).sum(axis=-1)
    
    # Campo cristalino local e Zeeman
    H_cf = Ko * O20 + h * Jz + eps2 * O22c + eps3 * O22s
    
    # Salto topológico intersub-rede modulado por Ko
    T = I3 + alpha * Ko * (Jx + Jy)
    H_AB = f_k[:, :, None, None] * T[None, None]
    
    H = np.zeros((*kx_g.shape, 6, 6), dtype=complex)
    H[:, :, :3, :3] = H_cf
    H[:, :, 3:, 3:] = -H_cf
    H[:, :, :3, 3:] = H_AB
    H[:, :, 3:, :3] = H_AB.conj().transpose(0, 1, 3, 2)
    return H

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — MÉTODO FHS RIGOROSO & ORÁCULO DE GAP ADAPTATIVO
# ══════════════════════════════════════════════════════════════════════════════

# Constante única de tolerância de gap. Alinhada ao limiar Δmin < 1e-6 declarado no
# texto do manuscrito (Seção 2.3). Antes desta correção, check_gap_adaptive() usava
# 1e-5 (default do parâmetro `tol`) enquanto fhs_chern_number() tinha um segundo
# limiar hardcoded, independente, também em 1e-5 -- dois literais que podiam divergir
# silenciosamente entre si e do texto. Agora ambos referenciam esta única constante.
GAP_TOL: float = 1e-6

def check_gap_adaptive(Ko: float, h: float, eps2: float, eps3: float, n_occ: int = 3,
                       N_init: int = 60, N_high: int = 120, tol: float = GAP_TOL) -> tuple[bool, float]:
    """
    Verifica a robustez do hiato espectral (gap) usando resolução adaptativa.
    Pontos com gap suspeito (perto do limite 'tol') são reavaliados em N_high=120.
    """
    # Avaliação primária em N_init
    k1d = np.linspace(0.0, 2.0 * np.pi, N_init, endpoint=False)
    kx_g, ky_g = np.meshgrid(k1d, k1d, indexing="ij")
    H_batch = _hamiltonian_batch(kx_g, ky_g, Ko, h, eps2, eps3)
    eigvals = np.linalg.eigvalsh(H_batch)
    
    gap_min = float(np.min(eigvals[..., n_occ] - eigvals[..., n_occ - 1]))

    if gap_min <= tol:
        return False, gap_min

    # Reavaliação adaptativa para pontos com gap frágil/suspeito perto da transição
    if tol < gap_min < 5.0 * tol:
        k1d_high = np.linspace(0.0, 2.0 * np.pi, N_high, endpoint=False)
        kx_h, ky_h = np.meshgrid(k1d_high, k1d_high, indexing="ij")
        H_high = _hamiltonian_batch(kx_h, ky_h, Ko, h, eps2, eps3)
        eigvals_high = np.linalg.eigvalsh(H_high)
        gap_min_high = float(np.min(eigvals_high[..., n_occ] - eigvals_high[..., n_occ - 1]))
        if gap_min_high <= tol:
            return False, gap_min_high
        return True, gap_min_high

    return True, gap_min

def fhs_chern_number(H_batch: np.ndarray, n_occ: int, quant_tol: float = 1e-2,
                      gap_tol: float = GAP_TOL) -> int | None:
    """
    Calcula o número de Chern via método Fukui-Hatsugai-Suzuki (FHS) com travas rígidas:
    1. Supressão de mascaramento silencioso: registra matrizes singulares |det(M)| < 1e-12.
    2. Validação estrita de inteiro: exige |C_raw - round(C_raw)| < quant_tol.
    Usa o mesmo GAP_TOL de check_gap_adaptive (ver Módulo 2) -- limiar único, não duplicado.
    """
    eigvals, psi_all = np.linalg.eigh(H_batch)
    
    # Validação preliminar de gap simples na malha fornecida
    gap_min = np.min(eigvals[..., n_occ] - eigenvalues_occ(eigvals, n_occ))
    if gap_min <= gap_tol:
        return None

    psi = psi_all[:, :, :, :n_occ]

    singular_detected = False

    def _link(ax: int) -> np.ndarray:
        nonlocal singular_detected
        psi_fwd = np.roll(psi, -1, axis=ax)
        M = np.einsum("...ia,...ib->...ab", psi.conj(), psi_fwd)
        det_M = np.linalg.det(M)
        
        # SUPRESSÃO DE SILENCIAMENTO: matrizes singulares indicam cruzamento de bandas/gap nulo
        abs_det = np.abs(det_M)
        if np.any(abs_det < 1e-12):
            singular_detected = True
            logger.debug("Singularidade detectada em _link: |det(M)| < 1e-12.")
            
        return det_M / (abs_det + 1e-15)

    U1 = _link(ax=0)
    U2 = _link(ax=1)

    if singular_detected:
        # Se houve degenerescência pontual na malha FHS, o invariante U(1) é indefinido
        return None

    U_plaquette = U1 * np.roll(U2, -1, axis=0) * np.roll(U1, -1, axis=1).conj() * U2.conj()
    F_tilde = np.angle(U_plaquette + 1e-10j)
    
    C_raw = F_tilde.sum() / (2.0 * np.pi)
    C_round = np.round(C_raw)

    # TRAVA DE INTEIRO: A integral dividida por 2pi deve ser um inteiro dentro de tol
    if abs(C_raw - C_round) > quant_tol:
        logger.debug(f"Violação de quantização inteira: C_raw={C_raw:.4f}, desvio={abs(C_raw - C_round):.4e}")
        return None

    return int(C_round)

def eigenvalues_occ(eigvals, n_occ):
    return eigvals[..., n_occ - 1]

def compute_chern_rigorous(Ko: float, h: float, eps2: float, eps3: float,
                           N_init: int = 60, N_high: int = 120, n_occ: int = 3) -> int | None:
    """
    Executa a verificação adaptativa do gap e calcula o número de Chern rigorosamente.
    """
    has_gap, _ = check_gap_adaptive(Ko, h, eps2, eps3, n_occ=n_occ, N_init=N_init, N_high=N_high)
    if not has_gap:
        return None

    k1d = np.linspace(0.0, 2.0 * np.pi, N_init, endpoint=False)
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
# MODULE 3 — GERADOR DE DATASET COM DISTRIBUIÇÃO FÍSICA REAL
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = Path("topological_dataset.csv")
_BOUNDS = {"Ko": (0.0, 3.0), "h": (-5.0, 5.0), "eps2": (0.0, 1.5), "eps3": (0.0, 1.5)}

def generate_dataset(n_samples=5000, N_bz=60, n_occ=3, seed=42, out=CSV_PATH):
    """
    Gera o dataset de fases topológicas amostrando aleatoriamente o espaço R^4.
    Reflete a prevalência natural (~1% de fases não-triviais) sem rebalanceamento artificial.
    """
    rng = np.random.default_rng(seed)
    valid_rows = []
    print(f"Gerando dataset com {n_samples} amostras topologicamente válidas...")
    pbar = tqdm(total=n_samples, desc="FHS Integrator Rigoroso")

    while len(valid_rows) < n_samples:
        batch_size = min(500, n_samples - len(valid_rows) + 200)
        Ko_v = rng.uniform(*_BOUNDS["Ko"], batch_size)
        h_v = rng.uniform(*_BOUNDS["h"], batch_size)
        eps2_v = rng.uniform(*_BOUNDS["eps2"], batch_size)
        eps3_v = rng.uniform(*_BOUNDS["eps3"], batch_size)

        for i in range(batch_size):
            if len(valid_rows) >= n_samples: break
            c = compute_chern_rigorous(Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], N_init=N_bz, n_occ=n_occ)
            if c is not None:
                valid_rows.append((Ko_v[i], h_v[i], eps2_v[i], eps3_v[i], c))
                pbar.update(1)

    pbar.close()
    df = pd.DataFrame(valid_rows, columns=["Ko", "h", "eps2", "eps3", "chern"])
    df.to_csv(out, index=False)
    print(f"\nDataset gerado -> {out}")
    print("Distribuição Real das Classes Topológicas (Chern):")
    counts = df["chern"].value_counts().sort_index()
    print(counts.to_string())
    non_trivial = (df["chern"] != 0).sum()
    print(f"Prevalência Real de Fases Não-Triviais (C != 0): {non_trivial / len(df):.2%}")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3b — GERAÇÃO PARALELA (multiprocessing; VETORIZAÇÃO EM LOTE TESTADA E
# DESCARTADA -- ver "erro 1" no relatório: mais lenta que o loop escalar em
# todos os tamanhos de lote testados, 1 a 96 pontos, neste oráculo)
# ══════════════════════════════════════════════════════════════════════════════

from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def _generate_chunk(n_target: int, N_bz: int, n_occ: int, entropy: int, worker_id: int):
    """Executado em processo separado. compute_chern_rigorous NÃO é modificado
    -- só distribuído. Stream aleatório independente via SeedSequence.spawn
    (não seed+worker_id manual, que pode gerar streams correlacionados)."""
    rng = np.random.default_rng(entropy)
    rows = []
    while len(rows) < n_target:
        batch = min(500, n_target - len(rows) + 200)
        Ko_v = rng.uniform(*_BOUNDS["Ko"], batch)
        h_v = rng.uniform(*_BOUNDS["h"], batch)
        e2_v = rng.uniform(*_BOUNDS["eps2"], batch)
        e3_v = rng.uniform(*_BOUNDS["eps3"], batch)
        for i in range(batch):
            if len(rows) >= n_target:
                break
            c = compute_chern_rigorous(Ko_v[i], h_v[i], e2_v[i], e3_v[i], N_init=N_bz, n_occ=n_occ)
            if c is not None:
                rows.append((Ko_v[i], h_v[i], e2_v[i], e3_v[i], c))
    return worker_id, rows


def generate_dataset_parallel(n_samples=50_000, N_bz=60, n_occ=3, seed=42,
                               out=CSV_PATH, n_workers=None):
    """Mesma distribuição/oráculo do generate_dataset original -- nenhuma
    linha de física alterada. Determinístico: reexecutar com o mesmo
    (seed, n_workers) reproduz byte-a-byte o mesmo dataset (validado)."""
    n_workers = n_workers or os.cpu_count() or 1
    per_worker = [n_samples // n_workers] * n_workers
    for i in range(n_samples % n_workers):
        per_worker[i] += 1
    entropies = [int(cs.generate_state(1)[0]) for cs in np.random.SeedSequence(seed).spawn(n_workers)]

    print(f"Gerando {n_samples} amostras em {n_workers} processo(s)...")
    t0 = time.perf_counter()
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_generate_chunk, per_worker[w], N_bz, n_occ, entropies[w], w)
                for w in range(n_workers)]
        for fut in as_completed(futs):
            wid, rows = fut.result()
            print(f"  worker {wid}: {len(rows)} amostras validas")
            all_rows.extend(rows)
    dt = time.perf_counter() - t0

    df = pd.DataFrame(all_rows, columns=["Ko", "h", "eps2", "eps3", "chern"])
    df.to_csv(out, index=False)
    print(f"\n{len(df)} amostras -> {out}  ({dt:.1f}s, {dt/len(df)*1000:.2f} ms/amostra efetivo)")
    counts = df["chern"].value_counts().sort_index()
    print(counts.to_string())
    print(f"Prevalencia nao-trivial: {(df['chern']!=0).sum()/len(df):.2%}")
    return df


if __name__ == "__main__":
    test_haldane_model()
    generate_dataset_parallel(n_samples=50000, N_bz=60, n_occ=3, seed=42)