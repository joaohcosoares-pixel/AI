#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mlp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Módulo 4: Topological Phase Classifier (PyTorch MLP)
Requer: topological_dataset.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

CSV_PATH = Path("topological_dataset.csv")
FEATURES = ["Ko", "h", "eps2", "eps3"]

class ChernDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

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

def train_classifier(csv_path=CSV_PATH, epochs=150, batch_size=256, lr=1e-3, val_frac=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTreinando modelo no dispositivo: {device}")

    df = pd.read_csv(csv_path)
    X_raw = df[FEATURES].values.astype(np.float32)
    y_raw = df["chern"].values

    classes = np.sort(np.unique(y_raw))
    c2i = {int(c): i for i, c in enumerate(classes)}
    y = np.array([c2i[int(c)] for c in y_raw], dtype=np.int64)
    n_classes = len(classes)
    
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_frac)
    tr_idx, va_idx = idx[n_val:], idx[:n_val]

    X_tr_raw, y_tr_raw = X_raw[tr_idx], y[tr_idx]

    # Implementação da repetição estrita dos estados quânticos válidos
    print("\nAplicando RandomOverSampler (Cópia Exata) no conjunto de treinamento...")
    ros = RandomOverSampler(random_state=42)
    X_tr_res, y_tr_res = ros.fit_resample(X_tr_raw, y_tr_raw)
    
    # Padronização e DataLoaders
    scaler = StandardScaler().fit(X_tr_res)
    X_tr = scaler.transform(X_tr_res)
    X_va = scaler.transform(X_raw[va_idx])

    tr_loader = DataLoader(ChernDataset(X_tr, y_tr_res), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(ChernDataset(X_va, y[va_idx]), batch_size=512, shuffle=False)

    model = TopoPhaseMLP(n_classes).to(device)
    
    # Restauração da métrica escalar simétrica padrão
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, best_state = float('inf'), None
    patience, epochs_no_improve = 15, 0
    
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
        tr_loss /= len(y_tr_res)

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
                
        va_loss /= len(va_idx)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
        scheduler.step()

        if va_loss < best_val_loss:
            best_val_loss = va_loss
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
    print(f"\nMétricas Finais de Validação -> F1-Score: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")
    return model, scaler, classes

if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"ERRO: Dataset {CSV_PATH} não encontrado. Execute data_generator.py primeiro.")
    else:
        model, scaler, chern_classes = train_classifier(epochs=150, batch_size=256, lr=1e-3)
        torch.save({
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "chern_classes": chern_classes.tolist(),
        }, "topological_mlp.pt")
        print("Salvo com sucesso -> topological_mlp.pt")