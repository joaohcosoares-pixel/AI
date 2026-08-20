#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_domain_guard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bateria de Testes Unitários e Validação de Confiabilidade de Software Científico
para a classe TopoDomainGuard.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler

from domain_guard import TopoDomainGuard


def test_bounding_box_strict_containment():
    """Testa a barreira geométrica estrita com limites exatos."""
    bounds = {
        "M": (1.95, 6.95),
        "p2": (-1.0, 1.0),
        "p3": (-1.0, 1.0),
        "p4": (-1.0, 1.0),
    }

    # Dados de treino fictícios para inicializar
    rng = np.random.default_rng(42)
    X_train = np.column_stack([
        rng.uniform(1.95, 6.95, 1000),
        rng.uniform(-1.0, 1.0, 1000),
        rng.uniform(-1.0, 1.0, 1000),
        rng.uniform(-1.0, 1.0, 1000),
    ])
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    guard = TopoDomainGuard(bounds, scaler, X_train_scaled)

    # Casos de Teste:
    # 1. Ponto perfeitamente central -> in_bbox = True
    # 2. Ponto exatamente nas bordas mínimas -> in_bbox = True
    # 3. Ponto exatamente nas bordas máximas -> in_bbox = True
    # 4. Violação de M abaixo do mínimo -> in_bbox = False
    # 5. Violação de M acima do máximo -> in_bbox = False
    # 6. Violação de p2 abaixo do mínimo -> in_bbox = False
    # 7. Violação de p4 acima do máximo -> in_bbox = False
    # 8. Múltiplas violações -> in_bbox = False
    test_points = np.array([
        [4.45, 0.0, 0.0, 0.0],         # 1. Dentro
        [1.95, -1.0, -1.0, -1.0],      # 2. Borda mínima exata
        [6.95, 1.0, 1.0, 1.0],         # 3. Borda máxima exata
        [1.9499, 0.0, 0.0, 0.0],       # 4. M < min
        [6.9501, 0.0, 0.0, 0.0],       # 5. M > max
        [4.0, -1.0001, 0.0, 0.0],      # 6. p2 < min
        [4.0, 0.0, 0.0, 1.0001],       # 7. p4 > max
        [0.0, 5.0, -10.0, 20.0],       # 8. Múltiplas violações extremas
    ], dtype=np.float64)

    expected_in_bbox = np.array([True, True, True, False, False, False, False, False], dtype=bool)

    bbox_result = guard.check_bbox(test_points)
    np.testing.assert_array_equal(bbox_result, expected_in_bbox)
    print("✓ Teste 1: Barreira Geométrica (Bounding Box) aprovado com rigor.")


def test_mahalanobis_vectorization_and_regularization():
    """Testa a precisão da distância de Mahalanobis regularizada vs Scipy."""
    bounds = {
        "M": (1.95, 6.95),
        "p2": (-1.0, 1.0),
        "p3": (-1.0, 1.0),
        "p4": (-1.0, 1.0),
    }

    rng = np.random.default_rng(123)
    n_train = 5000
    X_train = np.column_stack([
        rng.uniform(1.95, 6.95, n_train),
        rng.uniform(-1.0, 1.0, n_train),
        rng.uniform(-1.0, 1.0, n_train),
        rng.uniform(-1.0, 1.0, n_train),
    ])

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    guard = TopoDomainGuard(bounds, scaler, X_train_scaled, maha_percentile=99.5, epsilon=1e-6)

    # Validação do limiar de percentil 99.5
    train_dists = guard.train_maha_distances
    assert np.isclose(guard.maha_threshold, np.percentile(train_dists, 99.5)), "Limiar de percentil 99.5 incorreto!"

    # Gera 100 pontos de teste aleatórios no espaço escalado
    X_test_scaled = rng.normal(0.0, 1.5, size=(100, 4))
    fast_dists = guard._mahalanobis(X_test_scaled)

    # Compara contra o cálculo ponto a ponto do scipy usando a exata matriz inversa regularizada
    scipy_dists = np.array([
        mahalanobis(X_test_scaled[i], guard.mu, guard.cov_inv)
        for i in range(100)
    ])

    np.testing.assert_allclose(fast_dists, scipy_dists, rtol=1e-10, atol=1e-10)
    print("✓ Teste 2: Vetorização de Mahalanobis (np.einsum) e Regularização (eps=1e-6) aprovados.")


def test_oracle_routing_and_confidence_overconfidence_mitigation():
    """Testa a política mandatória de fallback ao oráculo e contenção de confiança."""
    bounds = {
        "M": (1.95, 6.95),
        "p2": (-1.0, 1.0),
        "p3": (-1.0, 1.0),
        "p4": (-1.0, 1.0),
    }

    rng = np.random.default_rng(999)
    X_train = rng.uniform([1.95, -1, -1, -1], [6.95, 1, 1, 1], size=(10000, 4))
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    guard = TopoDomainGuard(bounds, scaler, X_train_scaled, maha_percentile=99.5)

    # Cenários sintéticos de teste:
    # 1. Ponto Dentro do BBox, Densidade Normal, MLP prediz C=0 com alta confiança (P_nt = 0.01 < tau=0.1)
    #    -> in_bbox=True, in_density=True, ood=False, thr=False -> flagged_for_oracle = False (MLP decide)
    # 2. Ponto Dentro do BBox, Densidade Normal, MLP prediz C!=0 (P_nt = 0.85 >= tau=0.1)
    #    -> in_bbox=True, in_density=True, ood=False, thr=True -> flagged_for_oracle = True (Triagem positiva)
    # 3. Ponto FORA do BBox (M=10.0), mas MLP com saturação cega (P_nt = 0.001)
    #    -> in_bbox=False, in_density=?, ood=True, thr=False -> flagged_for_oracle = True (Interceptado por BBox!)
    # 4. Ponto Dentro do BBox, mas Anomalia de Densidade extrema (outlier espectral), MLP cega (P_nt = 0.001)
    #    -> in_bbox=True, in_density=False, ood=True, thr=False -> flagged_for_oracle = True (Interceptado por Mahalanobis!)

    # Criando um ponto que está dentro do bbox mas é outlier espectral extremo no scaler:
    # Como a distribuição de treino é uniforme no hiper-cubo, o limite de Mahalanobis p99.5 corta os vértices extremos
    outlier_scaled = guard.mu + 10.0 * np.std(X_train_scaled, axis=0)
    outlier_raw = scaler.inverse_transform(outlier_scaled.reshape(1, -1))[0]

    p1 = np.array([4.45, 0.0, 0.0, 0.0])              # Dentro, normal
    p2 = np.array([2.05, 0.0, 0.0, 0.0])              # Dentro, candidato a não-trivial
    p3 = np.array([12.50, 0.0, 0.0, 0.0])             # FORA do BBox (M=12.5)
    p4 = np.array([6.949, 0.999, 0.999, 0.999])       # Vértice extremo dentro do BBox (alta distância de Mahalanobis)

    X_eval = np.vstack([p1, p2, p3, p4])
    probs_nt = np.array([0.01, 0.85, 0.001, 0.002])   # MLP tem confiança cega nos pontos 1, 3 e 4
    threshold = 0.10

    routing = guard.route_inference(X_eval, probs_nt, threshold=threshold)

    # Ponto 3 DEVE ser OOD e DEVE ser flagged para oráculo mesmo com P_nt=0.001
    assert routing["in_bbox"][2] == False, "Ponto 3 deveria falhar no BBox!"
    assert routing["ood_mask"][2] == True, "Ponto 3 deveria ser OOD!"
    assert routing["flagged_for_oracle"][2] == True, "Ponto 3 deveria ser encaminhado ao oráculo!"

    # Ponto 1 deve ser confiável e não flagrado
    assert routing["in_bbox"][0] == True
    assert routing["in_density"][0] == True
    assert routing["ood_mask"][0] == False
    assert routing["flagged_for_oracle"][0] == False

    # Ponto 2 é confiável no domínio mas ultrapassa o threshold de triagem -> flagged
    assert routing["ood_mask"][1] == False
    assert routing["threshold_mask"][1] == True
    assert routing["flagged_for_oracle"][1] == True

    # Ponto 4: dentro do BBox, mas interceptado incondicionalmente pelo portão de Mahalanobis
    assert routing["in_bbox"][3] == True
    assert routing["in_density"][3] == False
    assert routing["ood_mask"][3] == True
    assert routing["flagged_for_oracle"][3] == True

    print("✓ Teste 3: Política de Fallback Mandatório e Mitigação de Saturação Cega aprovados.")


def test_inference_performance_benchmark():
    """Benchmark de vazão para inferência 100% vetorizada em 100k pontos."""
    bounds = {
        "M": (1.95, 6.95),
        "p2": (-1.0, 1.0),
        "p3": (-1.0, 1.0),
        "p4": (-1.0, 1.0),
    }
    rng = np.random.default_rng(42)
    X_train = rng.uniform([1.95, -1, -1, -1], [6.95, 1, 1, 1], size=(10000, 4))
    scaler = StandardScaler().fit(X_train)
    guard = TopoDomainGuard(bounds, scaler, scaler.transform(X_train))

    n_eval = 100_000
    X_large = rng.uniform([1.0, -2, -2, -2], [8.0, 2, 2, 2], size=(n_eval, 4)).astype(np.float32)
    probs_nt = rng.uniform(0.0, 1.0, size=n_eval).astype(np.float32)

    t0 = time.perf_counter()
    routing = guard.route_inference(X_large, probs_nt, threshold=0.1)
    dt = time.perf_counter() - t0

    throughput = n_eval / dt
    print(f"✓ Teste 4: Benchmark de Performance: {n_eval:,} pontos avaliados em {dt * 1000:.2f} ms ({throughput:,.0f} pts/segundo).")
    assert dt < 0.5, f"Desempenho abaixo do esperado ({dt:.3f}s para 100k pontos)!"


if __name__ == "__main__":
    print("=" * 70)
    print(" INICIANDO BATERIA DE TESTES DO GUARDA DE DOMÍNIO OOD (TopoDomainGuard)")
    print("=" * 70)
    test_bounding_box_strict_containment()
    test_mahalanobis_vectorization_and_regularization()
    test_oracle_routing_and_confidence_overconfidence_mitigation()
    test_inference_performance_benchmark()
    print("=" * 70)
    print(" TODOS OS TESTES PASSARAM COM 100% DE SUCESSO!")
    print("=" * 70)
