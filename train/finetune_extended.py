#!/usr/bin/env python3
"""
PVD4 Fine-tuning: lstm_aug100_extended → PDS_Data_Log 45개 CSV
- 기존 모델 + scalers 로드
- PDS_Data_Log 24개 칼럼만 선택 (기존 모델 입력 칼럼과 일치)
- 50x 노이즈 증강
- 낮은 LR(0.0001)로 추가학습
- 저장: lstm_aug100_extended_finetuned_best.pth (기존 scalers 그대로 사용)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import joblib
import warnings
import time
warnings.filterwarnings('ignore')


# ==================== Configuration ====================
DATA_PATH = "/home/goo4168/baco/PDS_Data_Log"
MODEL_DIR = "/home/goo4168/baco/train/models/PVD4"

# 기존 모델/스케일러 경로
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_aug100_extended_best.pth")
SCALER_INPUT_PATH = os.path.join(MODEL_DIR, "scaler_input_aug100_extended.pkl")
SCALER_TARGET_PATH = os.path.join(MODEL_DIR, "scaler_target_aug100_extended.pkl")

# 저장 경로
FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_aug100_extended_finetuned_best.pth")
FINETUNED_SCALER_INPUT_PATH = os.path.join(MODEL_DIR, "scaler_input_finetuned.pkl")
FINETUNED_SCALER_TARGET_PATH = os.path.join(MODEL_DIR, "scaler_target_finetuned.pkl")
RESULTS_PATH = os.path.join(MODEL_DIR, "lstm_aug100_extended_finetuned_results.txt")

# 24개 입력 칼럼 (기존 모델과 동일 순서)
INPUT_COLUMNS = [
    'ULVAC.Stage1.Temp1', 'ULVAC.Stage2.Temp1',
    'EN4.Power', 'EN4.Current', 'EN4.Volt',
    'PLA5.Match.Load.Posi', 'PLA5.Match.Tune.Posi',
    'PLA5.Match.Load.Pre', 'PLA5.Match.Tune.Pre',
    'PLA5.Match.DCBias',
    'SBRF5.Forward', 'SBRF5.Reflect', 'SBRF5.SetPower',
    'PWESC.Volt1', 'PWESC.Volt2',
    'OES.Data6',
    'Line.Gauge.i', 'Ion.Gauge.i', 'Baratron.Gauge.i',
    'Ar.MFC.i', 'Ar2.MFC.i', 'Ar.MFC.o', 'Ar2.MFC.o',
    'PWPDS.Data',
]

# 7개 타겟 칼럼
TARGET_COLUMNS = [
    'Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i', 'OES.Data6',
    'PLA5.Match.DCBias', 'SBRF5.Forward', 'SBRF5.Reflect',
]

# Log transform 칼럼 (기존 학습과 동일)
LOG_TRANSFORM_COLS = ['Ion.Gauge.i', 'Line.Gauge.i']

# 학습 설정
INPUT_WINDOW = 10
PREDICTION_HORIZON = 5
AUGMENTATION_FACTOR = 50
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001
EARLY_STOPPING_PATIENCE = 15
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2


# ==================== Dataset ====================
class PVDDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ==================== Model ====================
class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


# ==================== Data Loading ====================
def load_pds_data():
    """PDS_Data_Log 45개 CSV 로드, 24개 칼럼만 선택"""
    csv_files = sorted(glob.glob(os.path.join(DATA_PATH, "*.csv")))
    print(f"[FT] Found {len(csv_files)} CSV files in {DATA_PATH}")

    dataframes = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            # 필요한 칼럼 존재 확인
            missing = [c for c in INPUT_COLUMNS if c not in df.columns]
            if missing:
                print(f"  [SKIP] {os.path.basename(fp)}: missing {missing}")
                continue

            # 24개 입력 칼럼만 선택
            df_sel = df[INPUT_COLUMNS].copy()
            df_sel = df_sel.replace([np.inf, -np.inf], np.nan)
            df_sel = df_sel.ffill().bfill().fillna(0)

            # Log transform (기존 학습과 동일)
            for col in LOG_TRANSFORM_COLS:
                if col in df_sel.columns:
                    df_sel[col] = np.log1p(df_sel[col].abs())

            if len(df_sel) > INPUT_WINDOW + PREDICTION_HORIZON:
                dataframes.append(df_sel)
        except Exception as e:
            print(f"  [ERROR] {os.path.basename(fp)}: {e}")

    print(f"[FT] Loaded {len(dataframes)} valid CSV files")
    total_rows = sum(len(df) for df in dataframes)
    print(f"[FT] Total rows: {total_rows}")
    return dataframes


def augment_data(dataframes):
    """50x 노이즈 증강 (기존 방식과 동일)"""
    print(f"[FT] Augmentation factor: {AUGMENTATION_FACTOR}")
    augmented_dfs = []
    for aug_idx in range(AUGMENTATION_FACTOR):
        for df in dataframes:
            if aug_idx == 0:
                augmented_dfs.append(df.copy())
            else:
                df_noisy = df.copy()
                mask = df_noisy != 0
                noise = np.random.normal(0, 0.005, df.shape)
                df_noisy = df_noisy.where(~mask, df_noisy * (1 + noise))
                augmented_dfs.append(df_noisy)

    combined = pd.concat(augmented_dfs, ignore_index=True)
    print(f"[FT] Total rows after augmentation: {len(combined)}")
    return combined


def create_sequences(combined_df):
    """새 scalers로 fit_transform → 시퀀스 생성, scalers 반환"""
    from sklearn.preprocessing import MinMaxScaler

    input_data = combined_df.values.astype(np.float32)
    target_data = combined_df[TARGET_COLUMNS].values.astype(np.float32)

    input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
    target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)

    # PDS_Data_Log 데이터에 맞는 새 scalers 생성
    scaler_input = MinMaxScaler(feature_range=(-1, 1))
    scaler_target = MinMaxScaler(feature_range=(-1, 1))

    input_scaled = scaler_input.fit_transform(input_data)
    target_scaled = scaler_target.fit_transform(target_data)

    input_scaled = np.nan_to_num(input_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    sequences = []
    targets = []
    for i in range(len(input_scaled) - INPUT_WINDOW - PREDICTION_HORIZON + 1):
        seq = input_scaled[i:i + INPUT_WINDOW]
        target = target_scaled[i + INPUT_WINDOW + PREDICTION_HORIZON - 1]
        if not (np.isnan(seq).any() or np.isnan(target).any()):
            sequences.append(seq)
            targets.append(target)

    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32), scaler_input, scaler_target


# ==================== Training ====================
def train_finetune(model, train_loader, val_loader, device):
    """Fine-tuning: 낮은 LR, AdamW, early stopping"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Fine-tuning: LR={LEARNING_RATE}, Epochs={EPOCHS}, Patience={EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        valid_batches = 0

        for sequences, targets in train_loader:
            if torch.isnan(sequences).any() or torch.isnan(targets).any():
                continue
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            if torch.isnan(outputs).any():
                continue
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1

        train_loss = total_loss / max(valid_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        preds = np.concatenate(all_preds, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        val_mae = np.mean(np.abs(preds - targets_np))

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | MAE: {val_mae:.6f} | LR: {current_lr:.2e}")
            sys.stdout.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'epoch': epoch,
            }, FINETUNED_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest: Epoch {best_epoch+1}, Val Loss: {best_val_loss:.6f}, Val MAE: {best_val_mae:.6f}")
    return best_val_loss, best_val_mae, best_epoch


def evaluate_model(model, val_loader, scaler_target, device):
    """평가: 원래 스케일 복원 후 칼럼별 RMSE/MAE/MAPE"""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds_orig = scaler_target.inverse_transform(preds)
    targets_orig = scaler_target.inverse_transform(targets)

    results = {}
    for i, col in enumerate(TARGET_COLUMNS):
        mse = np.mean((preds_orig[:, i] - targets_orig[:, i]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_orig[:, i] - targets_orig[:, i]))
        mask = targets_orig[:, i] != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((preds_orig[mask, i] - targets_orig[mask, i]) / targets_orig[mask, i])) * 100
        else:
            mape = 0
        results[col] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    return results


# ==================== Main ====================
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n{'='*60}")
    print("Fine-tuning lstm_aug100_extended with PDS_Data_Log (45 CSV)")
    print(f"  Input columns: {len(INPUT_COLUMNS)}")
    print(f"  Target columns: {len(TARGET_COLUMNS)}")
    print(f"  Augmentation: {AUGMENTATION_FACTOR}x")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"{'='*60}")

    # 1. PDS_Data_Log 데이터 로드
    print("\n[1] Loading PDS_Data_Log...")
    dataframes = load_pds_data()
    if len(dataframes) == 0:
        raise ValueError("No valid CSV files found!")

    # 2. 증강
    print("\n[2] Augmenting data...")
    combined_df = augment_data(dataframes)

    # 3. 시퀀스 생성 (새 scalers로 fit_transform)
    print("\n[3] Creating sequences with new scalers...")
    sequences, targets, scaler_input, scaler_target = create_sequences(combined_df)
    print(f"  Sequences: {sequences.shape}, Targets: {targets.shape}")

    # 새 scalers 저장
    joblib.dump(scaler_input, FINETUNED_SCALER_INPUT_PATH)
    joblib.dump(scaler_target, FINETUNED_SCALER_TARGET_PATH)
    print(f"  Saved new scalers: {FINETUNED_SCALER_INPUT_PATH}")
    print(f"                     {FINETUNED_SCALER_TARGET_PATH}")

    # 4. Train/Val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    train_loader = DataLoader(PVDDataset(X_train, y_train), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(PVDDataset(X_val, y_val), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # 5. 기존 모델 로드
    print("\n[5] Loading base model...")
    model = LSTMPredictor(
        input_size=len(INPUT_COLUMNS),
        output_size=len(TARGET_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    checkpoint = torch.load(BASE_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Base model loaded (epoch={checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', '?'):.6f})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Fine-tuning
    print("\n[6] Fine-tuning...")
    start_time = time.time()
    best_val_loss, best_val_mae, best_epoch = train_finetune(model, train_loader, val_loader, device)
    elapsed = time.time() - start_time

    # 7. Best 모델 로드 및 평가
    print("\n[7] Evaluating finetuned model...")
    ft_checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ft_checkpoint['model_state_dict'])
    results = evaluate_model(model, val_loader, scaler_target, device)

    # 8. 결과 출력
    print(f"\n{'='*70}")
    print(" FINE-TUNING RESULTS")
    print(f"{'='*70}")
    print(f"  Val Loss: {best_val_loss:.6f}")
    print(f"  Val MAE:  {best_val_mae:.6f}")
    print(f"  Best Epoch: {best_epoch + 1}")
    print(f"  Train Time: {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print(" DETAILED METRICS BY COLUMN (Original Scale)")
    print(f"{'='*70}")
    for col in TARGET_COLUMNS:
        m = results[col]
        print(f"\n  {col}:")
        print(f"    RMSE: {m['RMSE']:.6f}")
        print(f"    MAE:  {m['MAE']:.6f}")
        print(f"    MAPE: {m['MAPE']:.2f}%")

    # 결과 파일 저장
    with open(RESULTS_PATH, 'w') as f:
        f.write("PVD4 Fine-tuned LSTM Results (PDS_Data_Log 45 CSV)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Base model: {BASE_MODEL_PATH}\n")
        f.write(f"Augmentation: {AUGMENTATION_FACTOR}x\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Input columns ({len(INPUT_COLUMNS)}): {INPUT_COLUMNS}\n")
        f.write(f"Target columns ({len(TARGET_COLUMNS)}): {TARGET_COLUMNS}\n\n")
        f.write(f"Overall:\n")
        f.write(f"  Val Loss: {best_val_loss:.6f}\n")
        f.write(f"  Val MAE:  {best_val_mae:.6f}\n")
        f.write(f"  Best Epoch: {best_epoch + 1}\n")
        f.write(f"  Train Time: {elapsed:.1f}s\n\n")
        f.write("Detailed Metrics by Column:\n")
        for col in TARGET_COLUMNS:
            m = results[col]
            f.write(f"\n{col}:\n")
            f.write(f"  RMSE: {m['RMSE']:.6f}\n")
            f.write(f"  MAE:  {m['MAE']:.6f}\n")
            f.write(f"  MAPE: {m['MAPE']:.2f}%\n")

    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Model saved:   {FINETUNED_MODEL_PATH}")
    print(f"Scalers:       (using existing, no change)")


if __name__ == '__main__':
    main()
