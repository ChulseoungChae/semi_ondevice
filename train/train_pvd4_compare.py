#!/usr/bin/env python3
"""
PVD4 모델 비교 학습 스크립트
- 증강 50으로 LSTM vs PatchTST 비교
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

# ==================== Configuration ====================
class Config:
    BASE_DATA_PATH = "/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog"
    SAVE_PATH = "/home/goo4168/baco/train/models"

    INPUT_WINDOW = 10
    PREDICTION_HORIZON = 5
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    EARLY_STOPPING_PATIENCE = 15

    # 증강 50으로 설정
    AUGMENTATION_FACTOR = 50

    PVD_TARGETS = {
        'PVD4': ['Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i', 'OES.Data6'],
    }


# ==================== Dataset ====================
class PVDDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ==================== Models ====================
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
        output = self.fc(context)
        return output


class PatchTSTEncoder(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 3,
                 patch_size: int = 2, dropout: float = 0.2):
        super(PatchTSTEncoder, self).__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        self.patch_embedding = nn.Linear(input_size * patch_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        num_patches = seq_len // self.patch_size
        if num_patches == 0:
            num_patches = 1
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, self.patch_size - seq_len))
            x = x_padded

        x = x[:, :num_patches * self.patch_size, :]
        x = x.reshape(batch_size, num_patches, self.patch_size * input_size)

        x = self.patch_embedding(x)
        x = x + self.pos_encoding[:, :num_patches, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        output = self.fc(x)
        return output


# ==================== Data Processing ====================
class DataProcessor:
    def __init__(self, config: Config):
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
                        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
                        df_numeric = df_numeric.ffill().bfill().fillna(0)
                        for col in df_numeric.columns:
                            if col in self.log_transform_cols:
                                df_numeric[col] = np.log1p(df_numeric[col].abs())
                        dataframes.append(df_numeric)
            except Exception as e:
                continue

        print(f"[PVD4] Loaded {len(dataframes)} valid CSV files")
        return dataframes

    def augment_data(self, dataframes):
        total_rows = sum(len(df) for df in dataframes)
        augmentation_factor = self.config.AUGMENTATION_FACTOR

        print(f"[PVD4] Total rows before augmentation: {total_rows}")
        print(f"[PVD4] Augmentation factor: {augmentation_factor}")

        augmented_dfs = []
        for aug_idx in range(augmentation_factor):
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
        print(f"[PVD4] Total rows after augmentation: {len(combined_df)}")

        return combined_df

    def create_sequences(self, df):
        input_data = df.values.astype(np.float32)
        target_data = df[self.target_columns].values.astype(np.float32)

        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)

        input_scaled = self.scaler_input.fit_transform(input_data)
        target_scaled = self.scaler_target.fit_transform(target_data)

        input_scaled = np.nan_to_num(input_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        sequences = []
        targets = []

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
        if len(dataframes) == 0:
            raise ValueError("No valid CSV files found for PVD4")

        combined_df = self.augment_data(dataframes)
        sequences, targets = self.create_sequences(combined_df)
        print(f"[PVD4] Created {len(sequences)} sequences")

        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )

        train_dataset = PVDDataset(X_train, y_train)
        val_dataset = PVDDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)

        input_size = sequences.shape[2]
        output_size = targets.shape[1]

        return train_loader, val_loader, input_size, output_size

    def save_scalers(self, save_dir, suffix=''):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler_input, os.path.join(save_dir, f'scaler_input{suffix}.pkl'))
        joblib.dump(self.scaler_target, os.path.join(save_dir, f'scaler_target{suffix}.pkl'))


# ==================== Training ====================
def train_model(model, train_loader, val_loader, config, device, model_name):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience_counter = 0
    best_epoch = 0

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")

    for epoch in range(config.EPOCHS):
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] Train: {train_loss:.6f} | Val: {val_loss:.6f} | MAE: {val_mae:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'epoch': epoch,
            }, os.path.join(config.SAVE_PATH, 'PVD4', f'{model_name}_aug50_best.pth'))
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best: Epoch {best_epoch+1}, Val Loss: {best_val_loss:.6f}, Val MAE: {best_val_mae:.6f}")

    return best_val_loss, best_val_mae, best_epoch


def evaluate_model(model, val_loader, scaler_target, device, target_columns):
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
    for i, col in enumerate(target_columns):
        mse = np.mean((preds_orig[:, i] - targets_orig[:, i]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_orig[:, i] - targets_orig[:, i]))

        mask = targets_orig[:, i] != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((preds_orig[mask, i] - targets_orig[mask, i]) / targets_orig[mask, i])) * 100
        else:
            mape = 0

        results[col] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    return results


def main():
    config = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 데이터 준비
    print("\n" + "="*60)
    print("Preparing data with augmentation factor = 50")
    print("="*60)

    processor = DataProcessor(config)
    train_loader, val_loader, input_size, output_size = processor.prepare_data()

    # 스케일러 저장
    save_dir = os.path.join(config.SAVE_PATH, 'PVD4')
    os.makedirs(save_dir, exist_ok=True)
    processor.save_scalers(save_dir, '_aug50')

    results = {}

    # ==================== LSTM 학습 ====================
    print("\n" + "#"*60)
    print("# LSTM Model")
    print("#"*60)

    lstm_model = LSTMPredictor(
        input_size=input_size,
        output_size=output_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    start_time = time.time()
    lstm_val_loss, lstm_val_mae, lstm_best_epoch = train_model(
        lstm_model, train_loader, val_loader, config, device, 'lstm'
    )
    lstm_time = time.time() - start_time

    # Best 모델 로드 및 평가
    checkpoint = torch.load(os.path.join(save_dir, 'lstm_aug50_best.pth'))
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_results = evaluate_model(lstm_model, val_loader, processor.scaler_target, device, config.PVD_TARGETS['PVD4'])

    results['LSTM'] = {
        'val_loss': lstm_val_loss,
        'val_mae': lstm_val_mae,
        'best_epoch': lstm_best_epoch,
        'train_time': lstm_time,
        'metrics': lstm_results
    }

    # ==================== PatchTST 학습 ====================
    print("\n" + "#"*60)
    print("# PatchTST Model")
    print("#"*60)

    patchtst_model = PatchTSTEncoder(
        input_size=input_size,
        output_size=output_size,
        d_model=config.HIDDEN_SIZE,
        nhead=8,
        num_layers=3,
        dropout=config.DROPOUT
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in patchtst_model.parameters()):,}")

    start_time = time.time()
    patchtst_val_loss, patchtst_val_mae, patchtst_best_epoch = train_model(
        patchtst_model, train_loader, val_loader, config, device, 'patchtst'
    )
    patchtst_time = time.time() - start_time

    # Best 모델 로드 및 평가
    checkpoint = torch.load(os.path.join(save_dir, 'patchtst_aug50_best.pth'))
    patchtst_model.load_state_dict(checkpoint['model_state_dict'])
    patchtst_results = evaluate_model(patchtst_model, val_loader, processor.scaler_target, device, config.PVD_TARGETS['PVD4'])

    results['PatchTST'] = {
        'val_loss': patchtst_val_loss,
        'val_mae': patchtst_val_mae,
        'best_epoch': patchtst_best_epoch,
        'train_time': patchtst_time,
        'metrics': patchtst_results
    }

    # ==================== 결과 비교 ====================
    print("\n" + "="*70)
    print(" COMPARISON RESULTS (Augmentation = 50)")
    print("="*70)

    print(f"\n{'Model':<12} {'Val Loss':<12} {'Val MAE':<12} {'Best Epoch':<12} {'Train Time':<12}")
    print("-"*60)
    for model_name, data in results.items():
        print(f"{model_name:<12} {data['val_loss']:<12.6f} {data['val_mae']:<12.6f} {data['best_epoch']+1:<12} {data['train_time']:.1f}s")

    print(f"\n{'='*70}")
    print(" DETAILED METRICS BY COLUMN")
    print("="*70)

    for col in config.PVD_TARGETS['PVD4']:
        print(f"\n{col}:")
        print(f"  {'Model':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
        print(f"  {'-'*48}")
        for model_name, data in results.items():
            m = data['metrics'][col]
            print(f"  {model_name:<12} {m['RMSE']:<12.4f} {m['MAE']:<12.4f} {m['MAPE']:<.2f}%")

    # 결과 파일 저장
    with open(os.path.join(save_dir, 'comparison_aug50_results.txt'), 'w') as f:
        f.write("PVD4 Model Comparison Results (Augmentation = 50)\n")
        f.write("="*70 + "\n\n")

        f.write("Overall Performance:\n")
        f.write(f"{'Model':<12} {'Val Loss':<12} {'Val MAE':<12} {'Best Epoch':<12} {'Train Time':<12}\n")
        f.write("-"*60 + "\n")
        for model_name, data in results.items():
            f.write(f"{model_name:<12} {data['val_loss']:<12.6f} {data['val_mae']:<12.6f} {data['best_epoch']+1:<12} {data['train_time']:.1f}s\n")

        f.write("\n\nDetailed Metrics:\n")
        for col in config.PVD_TARGETS['PVD4']:
            f.write(f"\n{col}:\n")
            for model_name, data in results.items():
                m = data['metrics'][col]
                f.write(f"  {model_name}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, MAPE={m['MAPE']:.2f}%\n")

    print(f"\nResults saved to: {os.path.join(save_dir, 'comparison_aug50_results.txt')}")

    # 승자 결정
    print("\n" + "="*70)
    print(" WINNER")
    print("="*70)

    lstm_avg_mape = np.mean([results['LSTM']['metrics'][col]['MAPE'] for col in config.PVD_TARGETS['PVD4']])
    patchtst_avg_mape = np.mean([results['PatchTST']['metrics'][col]['MAPE'] for col in config.PVD_TARGETS['PVD4']])

    print(f"Average MAPE - LSTM: {lstm_avg_mape:.2f}%, PatchTST: {patchtst_avg_mape:.2f}%")

    if lstm_avg_mape < patchtst_avg_mape:
        print(">>> LSTM wins! <<<")
    else:
        print(">>> PatchTST wins! <<<")


if __name__ == '__main__':
    main()
