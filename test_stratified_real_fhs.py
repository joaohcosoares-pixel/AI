#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stratified_real_fhs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validação de Integração Rigorosa: Pipeline de Roteamento OOD + Oráculo FHS
Mede o Speedup Estratificado Estimado via TopoDomainGuard real e FHS numérico.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from data_generator import _BOUNDS
from domain_guard import TopoDomainGuard
from train_mlp import measure_stratified_fhs_speedup


def test_real_fhs_stratification():
    rng = np.random.default_rng(42)
    N = 1000

    # 1. Gera 1000 pontos contínuos no espaço de parâmetros físico
    X_test = np.column_stack([
        rng.uniform(*_BOUNDS["M"], N),
        rng.uniform(*_BOUNDS["p2"], N),
        rng.uniform(*_BOUNDS["p3"], N),
        rng.uniform(*_BOUNDS["p4"], N),
    ]).astype(np.float32)

    # 2. Instancia TopoDomainGuard com dados de calibração sintéticos
    scaler = StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    guard = TopoDomainGuard(
        bounds=_BOUNDS,
        scaler=scaler,
        X_train_scaled=X_test_scaled,
        maha_percentile=99.5,
    )

    # 3. Cria vetor de probabilidades P(Cn != 0) simulado da MLP
    # 950 pontos de fase trivial clara (baixa probabilidade) + 50 candidatos positivos
    probs_trivial = rng.uniform(0.0001, 0.05, size=950)
    probs_candidate = rng.uniform(0.40, 0.95, size=50)
    probs_nt = np.concatenate([probs_trivial, probs_candidate]).astype(np.float32)
    rng.shuffle(probs_nt)

    # 4. Roteamento real através do TopoDomainGuard
    threshold_screening = 0.10
    routing = guard.route_inference(X_test, probs_nt, threshold=threshold_screening)

    # 5. Captura das máscaras e particionamento dos índices reais
    flagged_mask = routing["flagged_for_oracle"]
    flagged_idx = np.where(flagged_mask)[0]
    unflagged_idx = np.where(~flagged_mask)[0]

    assert len(flagged_idx) > 0, "Deveria haver pontos sinalizados para o oráculo!"
    assert len(unflagged_idx) > 0, "Deveria haver pontos filtrados pela MLP!"

    t_screening_total = 0.005  # 5 ms de screening

    # 6. Executa benchmark estratificado com amostragem dupla FHS real
    report = measure_stratified_fhs_speedup(
        X_clean=X_test,
        flagged_idx=flagged_idx,
        unflagged_idx=unflagged_idx,
        t_screening_total_sec=t_screening_total,
        k_sample=5,
        rng=rng,
        n_bootstrap=200,
    )

    # Asserções de conformidade estatística e nomenclatura
    assert ("speedup_stratified_estimated" in report or "speedup_measured" in report)
    speedup_val = report.get("speedup_stratified_estimated", report.get("speedup_measured"))
    assert speedup_val is not None and speedup_val > 0
    assert "t_fhs_flagged_ms" in report
    assert "t_fhs_unflagged_ms" in report
    assert "speedup_ci95" in report
    assert report["oracle_reduction_pct"] > 0
    print("✓ Teste de integração rigoroso com TopoDomainGuard e Oráculo FHS passou com 100% de sucesso!")


if __name__ == "__main__":
    test_real_fhs_stratification()
