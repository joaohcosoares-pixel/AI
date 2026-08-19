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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("FHS_Oracle")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — BULK HAMILTONIAN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_r3i = 1.0 / np.sqrt(3.0)

NN: np.ndarray = np.array(
    [
        [0.0, _r3i],
        [0.5, -0.5 * _r3i],
        [-0.5, -0.5 * _r3i],
    ],
    dtype=np.float64,
)

_r2 = np.sqrt(2.0)

Jx: np.ndarray = (
    np.array(
        [
            [0, _r2, 0],
            [_r2, 0, _r2],
            [0, _r2, 0],
        ],
        dtype=complex,
    )
    * 0.5
)

Jy: np.ndarray = (
    np.array(
        [
            [0, -1j * _r2, 0],
            [1j * _r2, 0, -1j * _r2],
            [0, 1j * _r2, 0],
        ],
        dtype=complex,
    )
    * 0.5
)

Jz: np.ndarray = np.diag([1.0, 0.0, -1.0]).astype(complex)
I3: np.ndarray = np.eye(3, dtype=complex)

O20: np.ndarray = 3.0 * (Jz @ Jz) - 2.0 * I3
O22c: np.ndarray = Jx @ Jx - Jy @ Jy
O22s: np.ndarray = Jx @ Jy + Jy @ Jx


def _hamiltonian_batch(
    kx_g,
    ky_g,
    Ko,
    h,
    eps2,
    eps3,
    alpha=0.5,
):
    """
    Constrói o Hamiltoniano de um MODELO REPRESENTATIVO TOPOLÓGICO (toy model)
    em um grid 2D do espaço k, sobre uma rede honeycomb com 2 sub-redes x
    3 estados orbitais.

    STATUS DE MODELAGEM (não é uma derivação microscópica de um material real):
    ---------------------------------------------------------------------------
    Este NÃO é o Hamiltoniano derivado de um composto específico. É um modelo
    representativo, construído para demonstrar/validar a metodologia
    computacional (integração FHS + aceleração por MLP) sobre um Hamiltoniano
    de banda com transições topológicas genuínas e controláveis.

    Duas escolhas estruturais precisam ser interpretadas como escolhas de
    modelagem, não como fatos físicos derivados:

    1) AMARRAÇÃO SIMÉTRICA DE Ko:
       Ko é usado tanto como intensidade do campo cristalino quadrupolar local
       O20 = 3*Jz^2 - 2*I3 quanto como fator que modula a amplitude do hopping
       intersub-rede:

           T = I3 + alpha*Ko*(Jx + Jy)

       Campo cristalino de íon único e integral de hopping são, em geral,
       parâmetros fisicamente independentes. A identificação dos dois através
       de Ko é uma escolha estrutural deste toy model.

    2) SINAL OPOSTO DE H_cf ENTRE SUB-REDES:
       H_A = H_cf e H_B = -H_cf também constitui uma escolha estrutural do
       modelo, não uma derivação microscópica a partir de simetria de um
       material específico.
    """

    phi = (
        kx_g[:, :, None] * NN[:, 0]
        + ky_g[:, :, None] * NN[:, 1]
    )

    f_k = np.exp(1j * phi).sum(axis=-1)

    # Campo cristalino local e Zeeman
    H_cf = (
        Ko * O20
        + h * Jz
        + eps2 * O22c
        + eps3 * O22s
    )

    # Salto intersub-rede modulado por Ko
    T = I3 + alpha * Ko * (Jx + Jy)

    H_AB = (
        f_k[:, :, None, None]
        * T[None, None]
    )

    H = np.zeros(
        (*kx_g.shape, 6, 6),
        dtype=complex,
    )

    H[:, :, :3, :3] = H_cf
    H[:, :, 3:, 3:] = -H_cf

    H[:, :, :3, 3:] = H_AB

    H[:, :, 3:, :3] = (
        H_AB
        .conj()
        .transpose(0, 1, 3, 2)
    )

    return H


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — GEOMETRIA DA ZONA DE BRILLOUIN + MÉTODO FHS RIGOROSO
# ══════════════════════════════════════════════════════════════════════════════

GAP_TOL: float = 1e-6

# Limiar absoluto de gap.
#
# Substitui a janela multiplicativa estreita
#
#     GAP_TOL < gap_min < 5*GAP_TOL
#
# porque o mínimo verdadeiro pode ocorrer entre pontos da malha grosseira.
#
# Calibração registrada na auditoria:
# apenas 2,75% dos pontos da varredura apresentaram gap_min(N=60) < 0.08.
REFINE_GAP_ABS: float = 0.08


# Critério secundário:
# concentração do fluxo de Berry nas plaquetas de maior |F_tilde|.
FLUX_TOPK: int = 4
FLUX_FRAC_THRESH: float = 0.10


# Vetores primitivos da rede recíproca.
#
# A malha da BZ deve ser parametrizada em coordenadas associadas a B1 e B2,
# e não em um toro cartesiano artificial [0,2π) × [0,2π).
B1: np.ndarray = np.array(
    [
        2.0 * np.pi,
        -2.0 * np.pi / np.sqrt(3.0),
    ],
    dtype=np.float64,
)

B2: np.ndarray = np.array(
    [
        0.0,
        -4.0 * np.pi / np.sqrt(3.0),
    ],
    dtype=np.float64,
)


def bz_grid(
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói a malha discreta da zona de Brillouin:

        k(n1,n2)
            = (n1/N) * B1
            + (n2/N) * B2,

    para

        n1,n2 = 0,...,N-1.

    Com essa parametrização:

        np.roll(..., axis=0)

    corresponde ao deslocamento B1/N e

        np.roll(..., axis=1)

    corresponde ao deslocamento B2/N.

    Portanto, o fechamento periódico da malha é realizado pelas translações
    recíprocas físicas da rede.
    """

    n = np.arange(
        N,
        dtype=np.float64,
    )

    N1, N2 = np.meshgrid(
        n,
        n,
        indexing="ij",
    )

    f1 = N1 / N
    f2 = N2 / N

    kx = (
        f1 * B1[0]
        + f2 * B2[0]
    )

    ky = (
        f1 * B1[1]
        + f2 * B2[1]
    )

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

    1. Verificação explícita do gap entre a banda ocupada mais alta e a
       primeira banda desocupada.

    2. Rejeição do ponto quando:

           gap_min <= gap_tol

    3. Detecção de matrizes de overlap praticamente singulares:

           |det(M)| < 1e-12

    4. Validação da quantização inteira:

           |C_raw - round(C_raw)| <= quant_tol

    Se return_diag=True, retorna:

        (chern, gap_min, F_tilde)

    Caso contrário, retorna apenas:

        chern

    ou None caso o ponto não possa receber um rótulo topológico confiável.
    """

    eigvals, psi_all = np.linalg.eigh(
        H_batch
    )

    # Gap direto entre:
    #   primeira banda não ocupada -> índice n_occ
    #   última banda ocupada       -> índice n_occ - 1
    #
    # Esta forma explícita elimina a dependência inexistente
    # eigenvalues_occ(...) presente na integração intermediária.
    gap = (
        eigvals[..., n_occ]
        - eigvals[..., n_occ - 1]
    )

    gap_min = float(
        np.min(gap)
    )

    if gap_min <= gap_tol:
        if return_diag:
            return None, gap_min, None

        return None

    # Subespaço ocupado.
    psi = psi_all[
        :,
        :,
        :,
        :n_occ,
    ]

    singular_detected = False

    def _link(
        ax: int,
    ) -> np.ndarray:
        """
        Determinante do overlap não-Abeliano entre os subespaços ocupados em
        pontos vizinhos da malha.
        """

        nonlocal singular_detected

        psi_fwd = np.roll(
            psi,
            -1,
            axis=ax,
        )

        M = np.einsum(
            "...ia,...ib->...ab",
            psi.conj(),
            psi_fwd,
        )

        det_M = np.linalg.det(
            M
        )

        abs_det = np.abs(
            det_M
        )

        if np.any(
            abs_det < 1e-12
        ):
            singular_detected = True

            logger.debug(
                "Singularidade detectada em _link: |det(M)| < 1e-12."
            )

        return (
            det_M
            / (abs_det + 1e-15)
        )

    U1 = _link(
        ax=0
    )

    U2 = _link(
        ax=1
    )

    if singular_detected:
        if return_diag:
            return None, gap_min, None

        return None

    U_plaquette = (
        U1
        * np.roll(
            U2,
            -1,
            axis=0,
        )
        * np.roll(
            U1,
            -1,
            axis=1,
        ).conj()
        * U2.conj()
    )

    F_tilde = np.angle(
        U_plaquette + 1e-10j
    )

    C_raw = (
        F_tilde.sum()
        / (2.0 * np.pi)
    )

    C_round = np.round(
        C_raw
    )

    quant_error = abs(
        C_raw - C_round
    )

    if quant_error > quant_tol:
        logger.debug(
            "Violação de quantização inteira: "
            f"C_raw={C_raw:.4f}, "
            f"desvio={quant_error:.4e}"
        )

        if return_diag:
            return None, gap_min, F_tilde

        return None

    C = int(
        C_round
    )

    if return_diag:
        return C, gap_min, F_tilde

    return C


def _needs_mesh_refinement(
    gap_min: float,
    F_tilde: np.ndarray | None,
) -> tuple[bool, str | None]:
    """
    Critério duplo para decidir se a malha grosseira deve ser substituída pela
    malha N_high.

    Critério 1:
        gap_min < REFINE_GAP_ABS

    Critério 2:
        fração do fluxo absoluto concentrada nas FLUX_TOPK maiores plaquetas
        superior a FLUX_FRAC_THRESH.
    """

    if gap_min < REFINE_GAP_ABS:
        return (
            True,
            "gap_absoluto",
        )

    if F_tilde is not None:
        flat = np.abs(
            F_tilde
        ).ravel()

        topk_flux = np.sort(
            flat
        )[-FLUX_TOPK:].sum()

        total_flux = flat.sum()

        topk_frac = (
            topk_flux
            / total_flux
        )

        if topk_frac > FLUX_FRAC_THRESH:
            return (
                True,
                "concentracao_de_fluxo_berry",
            )

    return (
        False,
        None,
    )


def compute_chern_rigorous(
    Ko: float,
    h: float,
    eps2: float,
    eps3: float,
    N_init: int = 60,
    N_high: int = 120,
    n_occ: int = 3,
) -> int | None:
    """
    Calcula o número de Chern sobre a geometria correta da zona de Brillouin.

    Procedimento:

    1. Constrói a malha N_init na base recíproca B1/B2.
    2. Diagonaliza o Hamiltoniano uma única vez nessa malha.
    3. Calcula:
         - Chern;
         - gap mínimo;
         - fluxo FHS por plaqueta.
    4. Avalia _needs_mesh_refinement.
    5. Se necessário, repete o cálculo em N_high.

    A assinatura externa é mantida compatível com generate_dataset() e
    generate_dataset_parallel().
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Malha inicial
    # ─────────────────────────────────────────────────────────────────────────

    kx0, ky0 = bz_grid(
        N_init
    )

    H0 = _hamiltonian_batch(
        kx0,
        ky0,
        Ko,
        h,
        eps2,
        eps3,
    )

    C0, gap0, F0 = fhs_chern_number(
        H0,
        n_occ,
        return_diag=True,
    )

    refine, _reason = _needs_mesh_refinement(
        gap0,
        F0,
    )

    if not refine:
        return C0

    # ─────────────────────────────────────────────────────────────────────────
    # Malha refinada
    # ─────────────────────────────────────────────────────────────────────────

    kx1, ky1 = bz_grid(
        N_high
    )

    H1 = _hamiltonian_batch(
        kx1,
        ky1,
        Ko,
        h,
        eps2,
        eps3,
    )

    C1, _gap1, _F1 = fhs_chern_number(
        H1,
        n_occ,
        return_diag=True,
    )

    return C1


def test_haldane_model():
    """
    Teste independente de regressão do integrador FHS usando o modelo de
    Haldane em regime topológico.
    """

    print(
        "Executando teste de validação do método FHS (Modelo de Haldane)..."
    )

    N = 30
    n_occ = 1

    k1d = np.linspace(
        0,
        2 * np.pi,
        N,
        endpoint=False,
    )

    kx, ky = np.meshgrid(
        k1d,
        k1d,
        indexing="ij",
    )

    delta = np.array(
        [
            [0.0, 1 / np.sqrt(3)],
            [0.5, -0.5 / np.sqrt(3)],
            [-0.5, -0.5 / np.sqrt(3)],
        ]
    )

    v1 = delta[1] - delta[2]
    v2 = delta[2] - delta[0]
    v3 = delta[0] - delta[1]

    M_mass = 0.3
    t1 = 1.0
    t2 = 0.1
    phi = np.pi / 2

    f_k = sum(
        np.exp(
            1j
            * (
                kx * d[0]
                + ky * d[1]
            )
        )
        for d in delta
    )

    sum_sin = sum(
        np.sin(
            kx * v[0]
            + ky * v[1]
        )
        for v in [
            v1,
            v2,
            v3,
        ]
    )

    d_z = (
        M_mass
        - 2
        * t2
        * np.sin(phi)
        * sum_sin
    )

    H = np.zeros(
        (
            N,
            N,
            2,
            2,
        ),
        dtype=complex,
    )

    H[:, :, 0, 0] = d_z
    H[:, :, 1, 1] = -d_z

    H[:, :, 0, 1] = (
        t1 * f_k
    )

    H[:, :, 1, 0] = (
        t1 * f_k.conj()
    )

    chern_val = fhs_chern_number(
        H,
        n_occ,
    )

    assert chern_val in [
        1,
        -1,
    ], (
        f"FALHA! Obtido C = {chern_val}."
    )

    print(
        f"Sucesso! C = {chern_val}.\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — GERADOR DE DATASET COM DISTRIBUIÇÃO FÍSICA REAL
# ══════════════════════════════════════════════════════════════════════════════

CSV_PATH = Path(
    "topological_dataset.csv"
)

_BOUNDS = {
    "Ko": (
        0.0,
        3.0,
    ),
    "h": (
        -5.0,
        5.0,
    ),
    "eps2": (
        0.0,
        1.5,
    ),
    "eps3": (
        0.0,
        1.5,
    ),
}


def generate_dataset(
    n_samples=5000,
    N_bz=60,
    n_occ=3,
    seed=42,
    out=CSV_PATH,
):
    """
    Gera dataset de fases topológicas amostrando aleatoriamente o espaço R^4.

    A prevalência das classes é determinada diretamente pelo oráculo FHS.
    """

    rng = np.random.default_rng(
        seed
    )

    valid_rows = []

    print(
        f"Gerando dataset com {n_samples} "
        "amostras topologicamente válidas..."
    )

    pbar = tqdm(
        total=n_samples,
        desc="FHS Integrator Rigoroso",
    )

    while len(valid_rows) < n_samples:
        batch_size = min(
            500,
            n_samples
            - len(valid_rows)
            + 200,
        )

        Ko_v = rng.uniform(
            *_BOUNDS["Ko"],
            batch_size,
        )

        h_v = rng.uniform(
            *_BOUNDS["h"],
            batch_size,
        )

        eps2_v = rng.uniform(
            *_BOUNDS["eps2"],
            batch_size,
        )

        eps3_v = rng.uniform(
            *_BOUNDS["eps3"],
            batch_size,
        )

        for i in range(
            batch_size
        ):
            if len(valid_rows) >= n_samples:
                break

            c = compute_chern_rigorous(
                Ko_v[i],
                h_v[i],
                eps2_v[i],
                eps3_v[i],
                N_init=N_bz,
                n_occ=n_occ,
            )

            if c is not None:
                valid_rows.append(
                    (
                        Ko_v[i],
                        h_v[i],
                        eps2_v[i],
                        eps3_v[i],
                        c,
                    )
                )

                pbar.update(
                    1
                )

    pbar.close()

    df = pd.DataFrame(
        valid_rows,
        columns=[
            "Ko",
            "h",
            "eps2",
            "eps3",
            "chern",
        ],
    )

    df.to_csv(
        out,
        index=False,
    )

    print(
        f"\nDataset gerado -> {out}"
    )

    print(
        "Distribuição Real das Classes Topológicas (Chern):"
    )

    counts = (
        df["chern"]
        .value_counts()
        .sort_index()
    )

    print(
        counts.to_string()
    )

    non_trivial = (
        df["chern"] != 0
    ).sum()

    print(
        "Prevalência Real de Fases Não-Triviais "
        f"(C != 0): {non_trivial / len(df):.2%}"
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3b — GERAÇÃO PARALELA
# ══════════════════════════════════════════════════════════════════════════════

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
import os


def _generate_chunk(
    n_target: int,
    N_bz: int,
    n_occ: int,
    entropy: int,
    worker_id: int,
):
    """
    Executado em processo separado.

    O oráculo físico compute_chern_rigorous não é modificado; apenas distribuído
    entre processos.

    Cada processo recebe um stream aleatório derivado de SeedSequence.spawn.
    """

    rng = np.random.default_rng(
        entropy
    )

    rows = []

    while len(rows) < n_target:
        batch = min(
            500,
            n_target
            - len(rows)
            + 200,
        )

        Ko_v = rng.uniform(
            *_BOUNDS["Ko"],
            batch,
        )

        h_v = rng.uniform(
            *_BOUNDS["h"],
            batch,
        )

        e2_v = rng.uniform(
            *_BOUNDS["eps2"],
            batch,
        )

        e3_v = rng.uniform(
            *_BOUNDS["eps3"],
            batch,
        )

        for i in range(
            batch
        ):
            if len(rows) >= n_target:
                break

            c = compute_chern_rigorous(
                Ko_v[i],
                h_v[i],
                e2_v[i],
                e3_v[i],
                N_init=N_bz,
                n_occ=n_occ,
            )

            if c is not None:
                rows.append(
                    (
                        Ko_v[i],
                        h_v[i],
                        e2_v[i],
                        e3_v[i],
                        c,
                    )
                )

    return (
        worker_id,
        rows,
    )


def generate_dataset_parallel(
    n_samples=50_000,
    N_bz=60,
    n_occ=3,
    seed=42,
    out=CSV_PATH,
    n_workers=None,
):
    """
    Versão paralela do gerador de dataset.

    A física e o oráculo são idênticos aos utilizados por generate_dataset().
    Os streams pseudoaleatórios individuais são derivados de SeedSequence.spawn.
    """

    n_workers = (
        n_workers
        or os.cpu_count()
        or 1
    )

    per_worker = (
        [n_samples // n_workers]
        * n_workers
    )

    for i in range(
        n_samples % n_workers
    ):
        per_worker[i] += 1

    seed_sequence = np.random.SeedSequence(
        seed
    )

    child_sequences = seed_sequence.spawn(
        n_workers
    )

    entropies = [
        int(
            child.generate_state(
                1
            )[0]
        )
        for child in child_sequences
    ]

    print(
        f"Gerando {n_samples} amostras "
        f"em {n_workers} processo(s)..."
    )

    t0 = time.perf_counter()

    all_rows = []

    with ProcessPoolExecutor(
        max_workers=n_workers
    ) as executor:
        futures = [
            executor.submit(
                _generate_chunk,
                per_worker[w],
                N_bz,
                n_occ,
                entropies[w],
                w,
            )
            for w in range(
                n_workers
            )
        ]

        for future in as_completed(
            futures
        ):
            worker_id, rows = (
                future.result()
            )

            print(
                f"  worker {worker_id}: "
                f"{len(rows)} amostras validas"
            )

            all_rows.extend(
                rows
            )

    dt = (
        time.perf_counter()
        - t0
    )

    df = pd.DataFrame(
        all_rows,
        columns=[
            "Ko",
            "h",
            "eps2",
            "eps3",
            "chern",
        ],
    )

    df.to_csv(
        out,
        index=False,
    )

    print(
        f"\n{len(df)} amostras -> {out} "
        f"({dt:.1f}s, "
        f"{dt / len(df) * 1000:.2f} ms/amostra efetivo)"
    )

    counts = (
        df["chern"]
        .value_counts()
        .sort_index()
    )

    print(
        counts.to_string()
    )

    prevalence = (
        (df["chern"] != 0).sum()
        / len(df)
    )

    print(
        "Prevalencia nao-trivial: "
        f"{prevalence:.2%}"
    )

    return df


if __name__ == "__main__":
    test_haldane_model()

    generate_dataset_parallel(
        n_samples=50_000,
        N_bz=60,
        n_occ=3,
        seed=42,
    )
