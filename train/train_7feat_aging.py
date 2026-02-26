#!/usr/bin/env python3
"""
7-Feature LSTM Aging Model Training
입력/출력: Ar.MFC.i, Baratron.Gauge.i, PLA5.Match.DCBias, EN4.Power,
           SBRF5.SetPower, PWPDS.Data, Ion.Gauge.i (7개)
입력 윈도우: 10초, 예측: 5초 후
데이터 증강: 50배 (복사 후 이어붙이기)
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== Config ====================
FEATURES = [
    'Ar.MFC.i', 'Baratron.Gauge.i', 'PLA5.Match.DCBias',
    'EN4.Power', 'SBRF5.SetPower', 'PWPDS.Data', 'Ion.Gauge.i'
]
INPUT_WINDOW = 10
PREDICTION_HORIZON = 5
AUGMENTATION = 50

MINIO_GLOB = '/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4/**/*.csv'
PDS_GLOB   = '/home/goo4168/baco/PDS_Data_Log/*.csv'

SAVE_DIR = '/home/goo4168/baco/train/models/PVD4'
MODEL_NAME = 'lstm_7feat'

HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 0.0005
PATIENCE     = 15


# ==================== Dataset ====================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ==================== Model ====================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size,
                 hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn = self.attention(lstm_out)
        ctx = torch.sum(attn * lstm_out, dim=1)
        return self.fc(ctx)


# ==================== Data Loading ====================
def load_csv(path):
    """단일 CSV 로드 후 필요 컬럼만 반환. 없으면 None."""
    try:
        df = pd.read_csv(path)
        if not all(c in df.columns for c in FEATURES):
            return None
        df = df[FEATURES].copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.ffill().bfill().fillna(0)
        # 모든 값이 0인 파일 제외
        if (df.abs().max() == 0).all():
            return None
        # 최소 길이 체크
        if len(df) < INPUT_WINDOW + PREDICTION_HORIZON + 1:
            return None
        return df
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(path)}: {e}")
        return None


def load_all_data():
    minio_files = glob.glob(MINIO_GLOB, recursive=True)
    minio_files = [f for f in minio_files if os.path.basename(f).startswith('PVD4')]
    pds_files   = glob.glob(PDS_GLOB)

    print(f"Found {len(minio_files)} minio PVD4 files")
    print(f"Found {len(pds_files)} PDS_Data_Log files")

    dfs = []
    for f in sorted(minio_files) + sorted(pds_files):
        df = load_csv(f)
        if df is not None:
            dfs.append(df)

    print(f"Valid files: {len(dfs)}")
    if not dfs:
        raise ValueError("No valid CSV files with required columns found")

    total_rows = sum(len(d) for d in dfs)
    print(f"Total rows before augmentation: {total_rows:,}")
    return dfs


def augment_and_combine(dfs, factor=50):
    """50배 증강: 동일 데이터를 factor번 이어 붙이기"""
    base = pd.concat(dfs, ignore_index=True)
    augmented = pd.concat([base] * factor, ignore_index=True)
    print(f"Total rows after {factor}x augmentation: {len(augmented):,}")
    return augmented


def make_sequences(df, scaler_input, scaler_target, fit=True):
    data = df.values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        inp_scaled = scaler_input.fit_transform(data)
        tgt_scaled = scaler_target.fit_transform(data)  # same columns
    else:
        inp_scaled = scaler_input.transform(data)
        tgt_scaled = scaler_target.transform(data)

    inp_scaled = np.nan_to_num(inp_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    tgt_scaled = np.nan_to_num(tgt_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    seqs, targets = [], []
    for i in range(len(inp_scaled) - INPUT_WINDOW - PREDICTION_HORIZON + 1):
        seq = inp_scaled[i: i + INPUT_WINDOW]
        tgt = tgt_scaled[i + INPUT_WINDOW + PREDICTION_HORIZON - 1]
        if not (np.isnan(seq).any() or np.isnan(tgt).any()):
            seqs.append(seq)
            targets.append(tgt)

    return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)


# ==================== Training ====================
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터 로드
    dfs = load_all_data()
    combined = augment_and_combine(dfs, AUGMENTATION)

    # 스케일러 & 시퀀스 생성
    scaler_input  = MinMaxScaler(feature_range=(-1, 1))
    scaler_target = MinMaxScaler(feature_range=(-1, 1))
    print("Creating sequences...")
    X, y = make_sequences(combined, scaler_input, scaler_target, fit=True)
    print(f"Sequences: {len(X):,}  (input={X.shape}, target={y.shape})")

    # 스케일러 저장
    joblib.dump(scaler_input,  os.path.join(SAVE_DIR, f'scaler_input_{MODEL_NAME}.pkl'))
    joblib.dump(scaler_target, os.path.join(SAVE_DIR, f'scaler_target_{MODEL_NAME}.pkl'))
    print("Scalers saved.")

    # Train/Val 분할
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"Train: {len(X_tr):,}  Val: {len(X_val):,}")

    train_loader = DataLoader(SeqDataset(X_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(SeqDataset(X_val, y_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # 모델
    n_feat = len(FEATURES)
    model = LSTMPredictor(n_feat, n_feat, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_cnt  = 0
    best_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_best.pth')

    print(f"\n{'='*55}")
    print(f"Training {MODEL_NAME} — {EPOCHS} epochs max")
    print(f"{'='*55}")

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        tr_loss, n_batches = 0.0, 0
        for Xb, yb in train_loader:
            if torch.isnan(Xb).any() or torch.isnan(yb).any():
                continue
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            if torch.isnan(out).any():
                continue
            loss = criterion(out, yb)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            n_batches += 1
        tr_loss /= max(n_batches, 1)

        # ---- Val ----
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                vl_loss += criterion(out, yb).item()
        vl_loss /= len(val_loader)

        scheduler.step(vl_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
                  f"train={tr_loss:.6f}  val={vl_loss:.6f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_cnt  = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': vl_loss,
                'epoch': epoch,
                'features': FEATURES,
                'input_window': INPUT_WINDOW,
                'prediction_horizon': PREDICTION_HORIZON,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
            }, best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {best_path}")

    # ---- Evaluation ----
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            all_preds.append(out.cpu().numpy())
            all_targets.append(yb.numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    preds_inv   = scaler_target.inverse_transform(preds)
    targets_inv = scaler_target.inverse_transform(targets)

    print(f"\n{'='*55}")
    print("Evaluation Results")
    print(f"{'='*55}")
    for i, col in enumerate(FEATURES):
        rmse = np.sqrt(np.mean((preds_inv[:, i] - targets_inv[:, i]) ** 2))
        mae  = np.mean(np.abs(preds_inv[:, i] - targets_inv[:, i]))
        mask = targets_inv[:, i] != 0
        mape = np.mean(np.abs((preds_inv[mask, i] - targets_inv[mask, i])
                               / targets_inv[mask, i])) * 100 if mask.sum() > 0 else 0
        print(f"  {col:25s}  RMSE={rmse:10.3f}  MAE={mae:10.3f}  MAPE={mape:.4f}%")

    print("\nDone!")


if __name__ == '__main__':
    train()
