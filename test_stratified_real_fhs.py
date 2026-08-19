#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stratified_real_fhs.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validação de execução real do measure_stratified_fhs_speedup com oráculo FHS físico.
"""

import numpy as np
from train_mlp import measure_stratified_fhs_speedup
from data_generator import _BOUNDS

def test_real_fhs_stratification():
    rng = np.random.default_rng(42)
    N = 1000
    
    # Gera 1000 pontos aleatórios
    X_test = np.column_stack([
        rng.uniform(*_BOUNDS["M"], N),
        rng.uniform(*_BOUNDS["p2"], N),
        rng.uniform(*_BOUNDS["p3"], N),
        rng.uniform(*_BOUNDS["p4"], N),
    ]).astype(np.float32)

    # Simula partição de índices (ex: 50 flagged, 950 unflagged)
    flagged_idx = np.arange(0, 50)
    unflagged_idx = np.arange(50, 1000)

    t_mlp_total = 0.005 # 5 ms

    # Executa o benchmark com k=5 para teste rápido
    report = measure_stratified_fhs_speedup(
        X_clean=X_test,
        flagged_idx=flagged_idx,
        unflagged_idx=unflagged_idx,
        t_mlp_total_sec=t_mlp_total,
        k_sample=5,
        rng=rng,
        n_bootstrap=200,
    )

    assert "speedup_measured" in report
    assert "t_fhs_flagged_ms" in report
    assert "t_fhs_unflagged_ms" in report
    assert "speedup_ci95" in report
    assert report["speedup_measured"] > 0
    print("✓ Teste de integração real com oráculo FHS passou com sucesso!")

if __name__ == "__main__":
    test_real_fhs_stratification()
