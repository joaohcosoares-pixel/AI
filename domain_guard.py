#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
domain_guard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Módulo de Confiabilidade de Software Científico e Guarda de Domínio OOD
(Out-Of-Distribution Domain Guard) para Classificadores Neurais Topológicos.

Mitiga a falha estrutural de "Saturação Cega de Confiança" (Blind Softmax
Overconfidence) através de dupla contenção geométrica e estatística:
  1. Barreira Geométrica (Bounding Box Físico)
  2. Filtro de Densidade Espectral (Distância de Mahalanobis com Regularização)
  3. Política de Fallback Mandatório ao Oráculo FHS (threshold_mask | ood_mask)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler


class TopoDomainGuard:
    """
    Guarda de Domínio Fora de Distribuição (Out-Of-Distribution - OOD).

    Implementa uma barreira dupla para contenção de inferência neural:
      1. Checagem Geométrica Estrita (Bounding Box Físico):
         Verifica se todas as features estão contidas nos limites físicos
         exatos [_BOUNDS] do modelo. Amostras com pelo menos uma feature
         violando [min, max] são flagradas como in_bbox = False.

      2. Filtro Espectral de Mahalanobis (Elipsoide de Centralidade):
         Mede a distância estatística de Mahalanobis no espaço padronizado
         exclusivamente com base no conjunto de treino:
           D_M(x) = sqrt( (x - mu)^T * (Sigma + eps*I)^(-1) * (x - mu) )
         com eps = 1e-6 na diagonal principal da matriz de covariância.
         O limiar estatístico (maha_threshold) é fixado no percentil 99.5
         (q_99.5) do conjunto de treino. Amostras com D_M > maha_threshold
         são flagradas como in_density = False.

      3. Política de Fallback Mandatório ao Oráculo:
         A confiança da rede neural é estritamente submissa ao Guarda de Domínio:
           ood_mask = ~(in_bbox & in_density)
           flagged_for_oracle = threshold_mask | ood_mask
         Pontos OOD têm sua predição neural sumariamente descartada e são
         roteados para auditoria numérica pesada (Oráculo FHS).
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        scaler: StandardScaler,
        X_train_scaled: np.ndarray,
        maha_percentile: float = 99.5,
        epsilon: float = 1e-6,
        feature_order: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Inicializa e calibra o Guarda de Domínio OOD.

        Parâmetros:
            bounds: Dicionário mapeando nome da feature -> (min_val, max_val).
            scaler: StandardScaler ajustado estritamente no conjunto de treino.
            X_train_scaled: Matriz de features de treino padronizadas (N_train, D).
            maha_percentile: Percentil para corte da distância de Mahalanobis (padrão: 99.5).
            epsilon: Fator de regularização da diagonal da covariância (padrão: 1e-6).
            feature_order: Ordem das features nas colunas de X (se None, usa chaves de bounds).
        """
        self.bounds: Dict[str, Tuple[float, float]] = bounds
        self.scaler: StandardScaler = scaler
        self.epsilon: float = float(epsilon)
        self.maha_percentile: float = float(maha_percentile)

        if feature_order is None:
            self.feature_order: Tuple[str, ...] = tuple(bounds.keys())
        else:
            self.feature_order = tuple(feature_order)

        # Pré-computação vetorizada das barreiras do Bounding Box físico (D,)
        self.mins: np.ndarray = np.array(
            [self.bounds[f][0] for f in self.feature_order], dtype=np.float64
        )
        self.maxs: np.ndarray = np.array(
            [self.bounds[f][1] for f in self.feature_order], dtype=np.float64
        )

        if X_train_scaled.ndim != 2:
            raise ValueError(
                f"X_train_scaled deve ser bidimensional (N, D), recebido: {X_train_scaled.shape}"
            )
        if X_train_scaled.shape[1] != len(self.feature_order):
            raise ValueError(
                f"Dimensão de X_train_scaled ({X_train_scaled.shape[1]}) não confere com "
                f"número de features ({len(self.feature_order)})."
            )

        # 1. Vetor de médias empíricas dos dados de treino padronizados (D,)
        self.mu: np.ndarray = np.mean(X_train_scaled, axis=0, dtype=np.float64)

        # 2. Matriz de covariância empírica com regularização numérica (D, D)
        cov: np.ndarray = np.cov(X_train_scaled, rowvar=False).astype(np.float64)
        cov_reg: np.ndarray = cov + self.epsilon * np.eye(cov.shape[0], dtype=np.float64)

        # 3. Matriz de precisão (inversa regularizada)
        self.cov_inv: np.ndarray = np.linalg.inv(cov_reg)

        # 4. Calibração empírica do limiar no percentil 99.5 dos dados de treino
        train_maha: np.ndarray = self._mahalanobis(X_train_scaled)
        self.maha_threshold: float = float(np.percentile(train_maha, self.maha_percentile))
        self.train_maha_distances: np.ndarray = train_maha

    def _mahalanobis(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Cálculo 100% vetorizado da distância de Mahalanobis via np.einsum.

        D_M(x_i) = sqrt( (x_i - mu) * Sigma^(-1) * (x_i - mu)^T )
        """
        X_arr = np.asarray(X_scaled, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        d = X_arr - self.mu
        # d @ cov_inv @ d.T vetorizado para N amostras simultâneas
        maha_sq = np.einsum("ij,jk,ik->i", d, self.cov_inv, d)
        # Proteção contra artefatos numéricos negativos de ponto flutuante
        return np.sqrt(np.maximum(maha_sq, 0.0))

    def check_bbox(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Verificação 100% vetorizada dos limites físicos (Bounding Box).

        Retorna:
            in_bbox (np.ndarray[bool]): True para amostras onde TODAS as features
                                       estão estritamente no intervalo [min, max].
        """
        X_arr = np.asarray(X_raw, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Vetorizado ao longo do eixo das features (axis=-1)
        in_bbox = np.all((X_arr >= self.mins) & (X_arr <= self.maxs), axis=-1)
        return in_bbox

    def check_density(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Verificação estatística baseada no elipsoide de Mahalanobis.

        Retorna:
            (in_density, maha_dist): in_density é True para D_M <= maha_threshold.
        """
        maha_dist = self._mahalanobis(X_scaled)
        in_density = maha_dist <= self.maha_threshold
        return in_density, maha_dist

    def check(self, X_raw: np.ndarray) -> Dict[str, Any]:
        """
        Executa a avaliação completa OOD (Geométrica + Densidade Espectral).

        Parâmetros:
            X_raw: Array NumPy de formato (N, D) ou (D,) com as features brutas.

        Retorna:
            Dicionário com:
              - 'trusted': in_bbox & in_density (bool array)
              - 'in_bbox': conformidade física estrita (bool array)
              - 'in_density': conformidade estatística com o domínio de treino (bool array)
              - 'ood_mask': ~(in_bbox & in_density) (bool array)
              - 'maha_dist': distâncias de Mahalanobis computadas (float64 array)
        """
        X_arr = np.asarray(X_raw, dtype=np.float64)
        is_single = X_arr.ndim == 1
        if is_single:
            X_arr = X_arr.reshape(1, -1)

        in_bbox = self.check_bbox(X_arr)
        X_scaled = self.scaler.transform(X_arr)
        in_density, maha_dist = self.check_density(X_scaled)

        trusted = in_bbox & in_density
        ood_mask = ~trusted

        return {
            "trusted": trusted if not is_single else trusted[0],
            "in_bbox": in_bbox if not is_single else in_bbox[0],
            "in_density": in_density if not is_single else in_density[0],
            "ood_mask": ood_mask if not is_single else ood_mask[0],
            "maha_dist": maha_dist if not is_single else maha_dist[0],
        }

    def route_inference(
        self,
        X_raw: np.ndarray,
        probs_nt: np.ndarray,
        threshold: float,
    ) -> Dict[str, Any]:
        """
        Aplica a política estrita de roteamento ao Oráculo FHS:

          ood_mask = ~(in_bbox & in_density)
          flagged_for_oracle = threshold_mask | ood_mask

        Se um ponto for classificado como OOD (ood_mask=True), a predição da rede
        neural é sumariamente descartada e o ponto é encaminhado para cálculo numérico.

        Parâmetros:
            X_raw: Amostras com features brutas (N, D).
            probs_nt: Vetor de probabilidades preditas P(C != 0) da MLP (N,).
            threshold: Limiar operacional de alta sensibilidade (ex: tau k=0).

        Retorna:
            Dicionário estruturado com máscaras e contabilidades de roteamento:
              - 'in_bbox': bool array (N,)
              - 'in_density': bool array (N,)
              - 'ood_mask': bool array (N,)
              - 'threshold_mask': bool array (N,)
              - 'flagged_for_oracle': bool array (N,)
              - 'n_total': int
              - 'n_flagged': int
              - 'n_unflagged': int
              - 'n_ood_intercepted': int
              - 'oracle_reduction_pct': float
        """
        guard_res = self.check(X_raw)
        in_bbox = np.asarray(guard_res["in_bbox"], dtype=bool)
        in_density = np.asarray(guard_res["in_density"], dtype=bool)
        maha_dist = np.asarray(guard_res["maha_dist"], dtype=np.float64)

        ood_mask = ~(in_bbox & in_density)
        threshold_mask = np.asarray(probs_nt >= threshold, dtype=bool)
        flagged_for_oracle = threshold_mask | ood_mask

        n_total = len(X_raw)
        n_flagged = int(np.sum(flagged_for_oracle))
        n_unflagged = n_total - n_flagged
        n_ood = int(np.sum(ood_mask))
        reduction_pct = (1.0 - n_flagged / n_total) * 100.0 if n_total > 0 else 0.0

        return {
            "in_bbox": in_bbox,
            "in_density": in_density,
            "maha_dist": maha_dist,
            "ood_mask": ood_mask,
            "threshold_mask": threshold_mask,
            "flagged_for_oracle": flagged_for_oracle,
            "n_total": n_total,
            "n_flagged": n_flagged,
            "n_unflagged": n_unflagged,
            "n_ood_intercepted": n_ood,
            "oracle_reduction_pct": reduction_pct,
        }
