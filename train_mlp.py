#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Módulo 4: Topological Phase Classifier (PyTorch MLP)

CORREÇÕES APLICADAS (anti-alucinação / anti-viés para Chern=0):
1. Remoção do RandomOverSampler (dados sintéticos são fisicamente inválidos).
2. Class weights no CrossEntropyLoss (ponderados pela frequência real das classes).
3. Split estratificado (mantém a proporção de classes na validação).
4. Seleção do melhor modelo por F1 macro (não apenas val_loss).
5. Métricas de validação por classe + função de inferência com confiança.
"""

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

warnings.filterwarnings("ignore")

CSV_PATH = Path("topological_balanced_dataset.csv")
FEATURES = ["Ko", "h", "eps2", "eps3"]


class ChernDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class TopoPhaseMLP(nn.Module):
    def __init__(self, n_classes: int, p: float = 0.25):
        super().__init__()

        def _block(d_in: int, d_out: int):
            return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(p))

        self.net = nn.Sequential(
            _block(4, 128),
            _block(128, 256),
            _block(256, 128),
            _block(128, 64),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def train_classifier(csv_path=CSV_PATH, epochs=200, batch_size=256, lr=1e-3,
                     val_frac=0.2, patience=20, use_class_weights=True):
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

def predict_chern(model, scaler, classes, Ko, h, eps2, eps3, threshold=0.5):
    """
    Prediz o número de Chern com probabilidade de confiança.

    Retorna (chern, confiança) onde 'confiança' é a probabilidade softmax da
    classe prevista. Se confiança < threshold, o ponto está numa região de
    fronteira/incerta — o modelo avisa em vez de "chutar".
    """
    device = next(model.parameters()).device
    x = np.array([[Ko, h, eps2, eps3]], dtype=np.float32)
    x_scaled = scaler.transform(x)
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


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"ERRO: Dataset {CSV_PATH} não encontrado. Execute data_generator.py primeiro.")
    else:
        model, scaler, chern_classes = train_classifier(epochs=200, batch_size=256, lr=1e-3)
        torch.save({
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "chern_classes": chern_classes.tolist(),
        }, "topological_mlp.pt")
        print("Salvo com sucesso -> topological_mlp.pt")

