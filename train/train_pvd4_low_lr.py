#!/usr/bin/env python3
"""
PVD4 LSTM 학습 - 낮은 Learning Rate (0.0001)
Baratron.Gauge.i, SBRF5.Reflect, OES.Data6, PLA5.Match.DCBias 정확도 개선 시도
"""

import os
import sys
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
import time
warnings.filterwarnings('ignore')

class Config:
    BASE_DATA_PATH = "/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog"
    SAVE_PATH = "/home/goo4168/baco/train/models"

    INPUT_WINDOW = 10
    PREDICTION_HORIZON = 5
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    BATCH_SIZE = 64
    EPOCHS = 150  # 더 많은 에폭
    LEARNING_RATE = 0.0001  # 낮은 학습률
    EARLY_STOPPING_PATIENCE = 20  # 더 긴 patience

    AUGMENTATION_FACTOR = 100

    PVD_TARGETS = {
        'PVD4': [
            'Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i', 'OES.Data6',
            'PLA5.Match.DCBias', 'SBRF5.Forward', 'SBRF5.Reflect'
        ],
    }


class PVDDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1), nn.Softmax(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.target_columns = config.PVD_TARGETS['PVD4']
        self.scaler_input = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_target = MinMaxScaler(feature_range=(-1, 1))
        self.log_transform_cols = ['Ion.Gauge.i', 'Line.Gauge.i']

    def load_csv_files(self):
        data_path = os.path.join(self.config.BASE_DATA_PATH, 'PVD4')
        csv_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)

        dataframes = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                if all(col in df.columns for col in self.target_columns):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0 and len(df) > self.config.INPUT_WINDOW + self.config.PREDICTION_HORIZON:
                        df_numeric = df[numeric_cols].copy()
                        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
                        for col in df_numeric.columns:
                            if col in self.log_transform_cols:
                                df_numeric[col] = np.log1p(df_numeric[col].abs())
                        dataframes.append(df_numeric)
            except:
                continue

        print(f"[PVD4] Loaded {len(dataframes)} CSV files")
        return dataframes

    def augment_data(self, dataframes):
        augmented_dfs = []
        for aug_idx in range(self.config.AUGMENTATION_FACTOR):
            for df in dataframes:
                if aug_idx == 0:
                    augmented_dfs.append(df.copy())
                else:
                    df_noisy = df.copy()
                    mask = df_noisy != 0
                    noise = np.random.normal(0, 0.005, df.shape)
                    df_noisy = df_noisy.where(~mask, df_noisy * (1 + noise))
                    augmented_dfs.append(df_noisy)

        combined_df = pd.concat(augmented_dfs, ignore_index=True)
        print(f"[PVD4] Augmented to {len(combined_df)} rows")
        return combined_df

    def create_sequences(self, df):
        input_data = df.values.astype(np.float32)
        target_data = df[self.target_columns].values.astype(np.float32)

        input_scaled = self.scaler_input.fit_transform(input_data)
        target_scaled = self.scaler_target.fit_transform(target_data)

        sequences, targets = [], []
        window = self.config.INPUT_WINDOW
        horizon = self.config.PREDICTION_HORIZON

        for i in range(len(input_scaled) - window - horizon + 1):
            seq = input_scaled[i:i + window]
            target = target_scaled[i + window + horizon - 1]
            if not (np.isnan(seq).any() or np.isnan(target).any()):
                sequences.append(seq)
                targets.append(target)

        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def prepare_data(self):
        dataframes = self.load_csv_files()
        combined_df = self.augment_data(dataframes)
        sequences, targets = self.create_sequences(combined_df)
        print(f"[PVD4] Created {len(sequences)} sequences")

        X_train, X_val, y_train, y_val = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        train_loader = DataLoader(PVDDataset(X_train, y_train), batch_size=self.config.BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(PVDDataset(X_val, y_val), batch_size=self.config.BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader, sequences.shape[2], targets.shape[1]

    def save_scalers(self, save_dir, suffix=''):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler_input, os.path.join(save_dir, f'scaler_input{suffix}.pkl'))
        joblib.dump(self.scaler_target, os.path.join(save_dir, f'scaler_target{suffix}.pkl'))


def train_model(model, train_loader, val_loader, config, device, save_dir):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    print(f"\nTraining with LR={config.LEARNING_RATE}")
    print("="*60)

    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        total_loss, valid_batches = 0, 0

        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            valid_batches += 1

        train_loss = total_loss / max(valid_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, targets).item()

        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}")
            sys.stdout.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch,
            }, os.path.join(save_dir, 'lstm_low_lr_best.pth'))
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest: Epoch {best_epoch+1}, Val Loss: {best_val_loss:.6f}")
    return best_val_loss, best_epoch


def evaluate_model(model, val_loader, scaler_target, device, target_columns):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for sequences, targets in val_loader:
            outputs = model(sequences.to(device))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    preds_orig = scaler_target.inverse_transform(preds)
    targets_orig = scaler_target.inverse_transform(targets)

    results = {}
    for i, col in enumerate(target_columns):
        mse = np.mean((preds_orig[:, i] - targets_orig[:, i]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_orig[:, i] - targets_orig[:, i]))
        mask = np.abs(targets_orig[:, i]) > 1e-10
        mape = np.mean(np.abs((preds_orig[mask, i] - targets_orig[mask, i]) / targets_orig[mask, i])) * 100 if mask.sum() > 0 else 0
        results[col] = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    return results


def main():
    config = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n" + "="*60)
    print("PVD4 LSTM Training - Low Learning Rate (0.0001)")
    print("="*60)

    processor = DataProcessor(config)
    train_loader, val_loader, input_size, output_size = processor.prepare_data()

    save_dir = os.path.join(config.SAVE_PATH, 'PVD4')
    os.makedirs(save_dir, exist_ok=True)
    processor.save_scalers(save_dir, '_low_lr')

    model = LSTMPredictor(input_size, output_size, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()
    best_val_loss, best_epoch = train_model(model, train_loader, val_loader, config, device, save_dir)
    train_time = time.time() - start_time

    # 평가
    checkpoint = torch.load(os.path.join(save_dir, 'lstm_low_lr_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    results = evaluate_model(model, val_loader, processor.scaler_target, device, config.PVD_TARGETS['PVD4'])

    print("\n" + "="*60)
    print("RESULTS (Low LR = 0.0001)")
    print("="*60)
    print(f"Val Loss: {best_val_loss:.6f}")
    print(f"Train Time: {train_time/60:.1f} min")
    print()

    for col in config.PVD_TARGETS['PVD4']:
        m = results[col]
        print(f"{col:25s} RMSE: {m['RMSE']:10.4f} | MAE: {m['MAE']:10.4f} | MAPE: {m['MAPE']:7.2f}%")

    # 결과 저장
    with open(os.path.join(save_dir, 'lstm_low_lr_results.txt'), 'w') as f:
        f.write("PVD4 LSTM - Low Learning Rate (0.0001)\n")
        f.write("="*60 + "\n")
        f.write(f"Val Loss: {best_val_loss:.6f}\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n\n")
        for col in config.PVD_TARGETS['PVD4']:
            m = results[col]
            f.write(f"{col}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, MAPE={m['MAPE']:.2f}%\n")

    print(f"\nModel saved: {os.path.join(save_dir, 'lstm_low_lr_best.pth')}")


if __name__ == '__main__':
    main()
