#!/usr/bin/env python3
"""
PVD4 LSTM vs PatchTST 모델 평가 스크립트
저장된 모델을 로드하여 성능 비교
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
        self.scaler_input = None
        self.scaler_target = None
        self.log_transform_cols = ['Ion.Gauge.i', 'Line.Gauge.i']

    def load_scalers(self, save_dir, suffix=''):
        self.scaler_input = joblib.load(os.path.join(save_dir, f'scaler_input{suffix}.pkl'))
        self.scaler_target = joblib.load(os.path.join(save_dir, f'scaler_target{suffix}.pkl'))

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
            except Exception:
                continue

        print(f"[PVD4] Loaded {len(dataframes)} valid CSV files")
        return dataframes

    def augment_data(self, dataframes):
        augmentation_factor = self.config.AUGMENTATION_FACTOR

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

        input_scaled = self.scaler_input.transform(input_data)
        target_scaled = self.scaler_target.transform(target_data)

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
        combined_df = self.augment_data(dataframes)
        sequences, targets = self.create_sequences(combined_df)
        print(f"[PVD4] Created {len(sequences)} sequences")

        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )

        val_dataset = PVDDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)

        input_size = sequences.shape[2]
        output_size = targets.shape[1]

        return val_loader, input_size, output_size


def evaluate_model(model, val_loader, scaler_target, device, target_columns):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets_gpu = targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets_gpu)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    val_loss = total_loss / len(val_loader)
    preds = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    val_mae = np.mean(np.abs(preds - targets_np))

    preds_orig = scaler_target.inverse_transform(preds)
    targets_orig = scaler_target.inverse_transform(targets_np)

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

    return val_loss, val_mae, results


def main():
    config = Config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_dir = os.path.join(config.SAVE_PATH, 'PVD4')

    # 스케일러 로드
    processor = DataProcessor(config)
    processor.load_scalers(save_dir, '_aug50')

    print("\n" + "="*60)
    print("Loading data and preparing validation set...")
    print("="*60)

    val_loader, input_size, output_size = processor.prepare_data()

    results = {}

    # ==================== LSTM 평가 ====================
    print("\n" + "#"*60)
    print("# LSTM Model Evaluation")
    print("#"*60)

    lstm_model = LSTMPredictor(
        input_size=input_size,
        output_size=output_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)

    lstm_checkpoint = torch.load(os.path.join(save_dir, 'lstm_aug50_best.pth'), map_location=device)
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    print(f"LSTM - Loaded from epoch {lstm_checkpoint.get('epoch', 'N/A')}")
    print(f"LSTM - Saved Val Loss: {lstm_checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"LSTM - Saved Val MAE: {lstm_checkpoint.get('val_mae', 'N/A'):.6f}")

    lstm_val_loss, lstm_val_mae, lstm_metrics = evaluate_model(
        lstm_model, val_loader, processor.scaler_target, device, config.PVD_TARGETS['PVD4']
    )

    results['LSTM'] = {
        'val_loss': lstm_val_loss,
        'val_mae': lstm_val_mae,
        'saved_epoch': lstm_checkpoint.get('epoch', 'N/A'),
        'metrics': lstm_metrics
    }

    print(f"LSTM - Evaluated Val Loss: {lstm_val_loss:.6f}")
    print(f"LSTM - Evaluated Val MAE: {lstm_val_mae:.6f}")

    # ==================== PatchTST 평가 ====================
    print("\n" + "#"*60)
    print("# PatchTST Model Evaluation")
    print("#"*60)

    patchtst_model = PatchTSTEncoder(
        input_size=input_size,
        output_size=output_size,
        d_model=config.HIDDEN_SIZE,
        nhead=8,
        num_layers=3,
        dropout=config.DROPOUT
    ).to(device)

    patchtst_checkpoint = torch.load(os.path.join(save_dir, 'patchtst_aug50_best.pth'), map_location=device)
    patchtst_model.load_state_dict(patchtst_checkpoint['model_state_dict'])
    print(f"PatchTST - Loaded from epoch {patchtst_checkpoint.get('epoch', 'N/A')}")
    print(f"PatchTST - Saved Val Loss: {patchtst_checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"PatchTST - Saved Val MAE: {patchtst_checkpoint.get('val_mae', 'N/A'):.6f}")

    patchtst_val_loss, patchtst_val_mae, patchtst_metrics = evaluate_model(
        patchtst_model, val_loader, processor.scaler_target, device, config.PVD_TARGETS['PVD4']
    )

    results['PatchTST'] = {
        'val_loss': patchtst_val_loss,
        'val_mae': patchtst_val_mae,
        'saved_epoch': patchtst_checkpoint.get('epoch', 'N/A'),
        'metrics': patchtst_metrics
    }

    print(f"PatchTST - Evaluated Val Loss: {patchtst_val_loss:.6f}")
    print(f"PatchTST - Evaluated Val MAE: {patchtst_val_mae:.6f}")

    # ==================== 결과 비교 ====================
    print("\n" + "="*70)
    print(" COMPARISON RESULTS (Augmentation = 50)")
    print("="*70)

    print(f"\n{'Model':<12} {'Val Loss':<14} {'Val MAE':<14} {'Best Epoch':<12}")
    print("-"*52)
    for model_name, data in results.items():
        epoch = data['saved_epoch'] + 1 if isinstance(data['saved_epoch'], int) else data['saved_epoch']
        print(f"{model_name:<12} {data['val_loss']:<14.6f} {data['val_mae']:<14.6f} {epoch}")

    print(f"\n{'='*70}")
    print(" DETAILED METRICS BY COLUMN (Original Scale)")
    print("="*70)

    for col in config.PVD_TARGETS['PVD4']:
        print(f"\n{col}:")
        print(f"  {'Model':<12} {'RMSE':<14} {'MAE':<14} {'MAPE':<12}")
        print(f"  {'-'*50}")
        for model_name, data in results.items():
            m = data['metrics'][col]
            print(f"  {model_name:<12} {m['RMSE']:<14.6f} {m['MAE']:<14.6f} {m['MAPE']:<.2f}%")

    # ==================== 승자 결정 ====================
    print("\n" + "="*70)
    print(" WINNER DETERMINATION")
    print("="*70)

    lstm_avg_mape = np.mean([results['LSTM']['metrics'][col]['MAPE'] for col in config.PVD_TARGETS['PVD4']])
    patchtst_avg_mape = np.mean([results['PatchTST']['metrics'][col]['MAPE'] for col in config.PVD_TARGETS['PVD4']])

    lstm_avg_mae = np.mean([results['LSTM']['metrics'][col]['MAE'] for col in config.PVD_TARGETS['PVD4']])
    patchtst_avg_mae = np.mean([results['PatchTST']['metrics'][col]['MAE'] for col in config.PVD_TARGETS['PVD4']])

    print(f"\nAverage MAPE:")
    print(f"  LSTM:     {lstm_avg_mape:.2f}%")
    print(f"  PatchTST: {patchtst_avg_mape:.2f}%")

    print(f"\nAverage MAE (Original Scale):")
    print(f"  LSTM:     {lstm_avg_mae:.6f}")
    print(f"  PatchTST: {patchtst_avg_mae:.6f}")

    print(f"\nValidation Loss:")
    print(f"  LSTM:     {results['LSTM']['val_loss']:.6f}")
    print(f"  PatchTST: {results['PatchTST']['val_loss']:.6f}")

    print("\n" + "="*70)
    if results['LSTM']['val_loss'] < results['PatchTST']['val_loss']:
        improvement = (results['PatchTST']['val_loss'] - results['LSTM']['val_loss']) / results['PatchTST']['val_loss'] * 100
        print(f" >>> LSTM WINS! (Val Loss {improvement:.1f}% lower) <<<")
    else:
        improvement = (results['LSTM']['val_loss'] - results['PatchTST']['val_loss']) / results['LSTM']['val_loss'] * 100
        print(f" >>> PatchTST WINS! (Val Loss {improvement:.1f}% lower) <<<")
    print("="*70)

    # 결과 저장
    with open(os.path.join(save_dir, 'comparison_aug50_results.txt'), 'w') as f:
        f.write("PVD4 LSTM vs PatchTST Comparison Results (Augmentation = 50)\n")
        f.write("="*70 + "\n\n")

        f.write("Overall Performance:\n")
        f.write(f"{'Model':<12} {'Val Loss':<14} {'Val MAE':<14} {'Best Epoch':<12}\n")
        f.write("-"*52 + "\n")
        for model_name, data in results.items():
            epoch = data['saved_epoch'] + 1 if isinstance(data['saved_epoch'], int) else data['saved_epoch']
            f.write(f"{model_name:<12} {data['val_loss']:<14.6f} {data['val_mae']:<14.6f} {epoch}\n")

        f.write("\n\nDetailed Metrics by Column:\n")
        for col in config.PVD_TARGETS['PVD4']:
            f.write(f"\n{col}:\n")
            for model_name, data in results.items():
                m = data['metrics'][col]
                f.write(f"  {model_name}: RMSE={m['RMSE']:.6f}, MAE={m['MAE']:.6f}, MAPE={m['MAPE']:.2f}%\n")

        f.write("\n\nWinner Summary:\n")
        f.write(f"LSTM Val Loss: {results['LSTM']['val_loss']:.6f}\n")
        f.write(f"PatchTST Val Loss: {results['PatchTST']['val_loss']:.6f}\n")
        if results['LSTM']['val_loss'] < results['PatchTST']['val_loss']:
            f.write("Winner: LSTM\n")
        else:
            f.write("Winner: PatchTST\n")

    print(f"\nResults saved to: {os.path.join(save_dir, 'comparison_aug50_results.txt')}")


if __name__ == '__main__':
    main()
