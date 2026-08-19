#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stratified_speedup.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Validação Metodológica do Benchmark Estratificado de Speedup (Amostragem Dupla Independente).
"""

from __future__ import annotations

import time
import numpy as np


def compute_stratified_speedup(
    n_flagged: int,
    n_unflagged: int,
    latencies_flagged: np.ndarray,
    latencies_unflagged: np.ndarray,
    t_mlp_total: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Calcula o Speedup Despoluído via Amostragem Dupla Independente com IC 95% via Bootstrap.
    """
    t_flagged_mean = float(np.mean(latencies_flagged))
    t_flagged_std = float(np.std(latencies_flagged, ddof=1)) if len(latencies_flagged) > 1 else 0.0

    t_unflagged_mean = float(np.mean(latencies_unflagged))
    t_unflagged_std = float(np.std(latencies_unflagged, ddof=1)) if len(latencies_unflagged) > 1 else 0.0

    latency_ratio = t_flagged_mean / t_unflagged_mean if t_unflagged_mean > 0 else 1.0

    # Projeção de custos totais (segundos)
    t_pure_fhs = (n_flagged * t_flagged_mean) + (n_unflagged * t_unflagged_mean)
    t_fhs_audit_flagged = n_flagged * t_flagged_mean
    t_hybrid = t_mlp_total + t_fhs_audit_flagged

    speedup = t_pure_fhs / t_hybrid if t_hybrid > 0 else float("inf")

    # Incerteza estatística via Bootstrap não-paramétrico
    rng = np.random.default_rng(seed)
    boot_speedups = []
    k_flagged = len(latencies_flagged)
    k_unflagged = len(latencies_unflagged)

    for _ in range(n_bootstrap):
        b_f = rng.choice(latencies_flagged, size=k_flagged, replace=True)
        b_u = rng.choice(latencies_unflagged, size=k_unflagged, replace=True)
        mean_f = np.mean(b_f)
        mean_u = np.mean(b_u)
        b_pure = (n_flagged * mean_f) + (n_unflagged * mean_u)
        b_hyb = t_mlp_total + (n_flagged * mean_f)
        boot_speedups.append(b_pure / b_hyb if b_hyb > 0 else 0.0)

    ci_low = float(np.percentile(boot_speedups, 2.5))
    ci_high = float(np.percentile(boot_speedups, 97.5))

    return {
        "n_total": n_flagged + n_unflagged,
        "n_flagged": n_flagged,
        "n_unflagged": n_unflagged,
        "t_fhs_flagged_mean_s": t_flagged_mean,
        "t_fhs_flagged_std_s": t_flagged_std,
        "t_fhs_unflagged_mean_s": t_unflagged_mean,
        "t_fhs_unflagged_std_s": t_unflagged_std,
        "latency_ratio": latency_ratio,
        "t_pure_fhs_s": t_pure_fhs,
        "t_hybrid_s": t_hybrid,
        "t_mlp_total_s": t_mlp_total,
        "speedup": speedup,
        "speedup_ci95": (ci_low, ci_high),
    }


def test_stratified_speedup_math():
    """Testa se a fórmula estratificada previne o viés de latência uniforme."""
    N_eval = 100_000
    N_flagged = 2_500     # 2.5% dos pontos são encaminhados ao oráculo
    N_unflagged = 97_500  # 97.5% dos pontos são descartados pela MLP

    # Suponha que pontos flagged demoram 15 ms (com refinamento N=120) e unflagged 3 ms (N=60)
    latencies_flagged = np.array([0.015] * 40, dtype=np.float64)
    latencies_unflagged = np.array([0.003] * 40, dtype=np.float64)
    t_mlp_total = 0.080  # 80 ms para forward pass de 100k pontos

    res = compute_stratified_speedup(
        n_flagged=N_flagged,
        n_unflagged=N_unflagged,
        latencies_flagged=latencies_flagged,
        latencies_unflagged=latencies_unflagged,
        t_mlp_total=t_mlp_total,
    )

    # Cálculo manual esperado:
    # T_pure = 2,500 * 0.015 + 97,500 * 0.003 = 37.5 + 292.5 = 330.0 s
    # T_hybrid = 0.080 + 2,500 * 0.015 = 0.080 + 37.5 = 37.580 s
    # Speedup = 330.0 / 37.580 = 8.7812x
    assert np.isclose(res["t_pure_fhs_s"], 330.0), f"T_pure incorreto: {res['t_pure_fhs_s']}"
    assert np.isclose(res["t_hybrid_s"], 37.58), f"T_hybrid incorreto: {res['t_hybrid_s']}"
    assert np.isclose(res["speedup"], 330.0 / 37.58), f"Speedup incorreto: {res['speedup']}"

    # Comparação com o método ingênuo (assumindo que todos os pontos demoram a média dos flagged = 15 ms):
    naive_t_pure = 100_000 * 0.015 # 1500 s
    naive_speedup = naive_t_pure / 37.58 # 39.9x (inflado em 4.5x!)
    print(f"✓ Validação Matemática:")
    print(f"   Speedup Ingênuo Inflado:    {naive_speedup:.2f}x")
    print(f"   Speedup Estratificado Real: {res['speedup']:.2f}x  (IC 95%: [{res['speedup_ci95'][0]:.2f}x, {res['speedup_ci95'][1]:.2f}x])")
    print(f"   Razão de Latência FHS:      {res['latency_ratio']:.2f}x mais lento em pontos críticos.")
    print("✓ Teste de Estratificação Aprovado!")


if __name__ == "__main__":
    test_stratified_speedup_math()
