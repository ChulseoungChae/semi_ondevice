#!/usr/bin/env python3
"""
PVD Chamber Sensor Prediction Model
- LSTM-based time series prediction for anomaly detection
- Predicts Ar.MFC.i, Ion.Gauge.i, Baratron.Gauge.i (+ OES.Data6 for PVD4)
- Input window: 10 timesteps, Prediction horizon: 5 timesteps ahead
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class Config:
    # Data paths
    BASE_DATA_PATH = "/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog"
    SAVE_PATH = "/home/goo4168/baco/train/models"

    # Model parameters
    INPUT_WINDOW = 10  # 입력 시퀀스 길이
    PREDICTION_HORIZON = 5  # 5초 후 예측
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # Training parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.0005
    EARLY_STOPPING_PATIENCE = 15

    # Data augmentation
    MIN_AUGMENTATION = 10  # 최소 증강 횟수
    MAX_AUGMENTATION = 30  # 최대 증강 횟수

    # PVD-specific target columns
    PVD_TARGETS = {
        'PVD1': ['Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i'],
        'PVD2': ['Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i'],
        'PVD3': ['Ar.200.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i'],  # PVD3는 Ar.200.MFC.i
        'PVD4': ['Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i', 'OES.Data6'],
    }


# ==================== Dataset ====================
class PVDDataset(Dataset):
    """PVD 시계열 데이터셋"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ==================== Models ====================
class LSTMPredictor(nn.Module):
    """LSTM 기반 예측 모델"""

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
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # Prediction
        output = self.fc(context)
        return output


class PatchTSTEncoder(nn.Module):
    """PatchTST 스타일의 Transformer 인코더"""

    def __init__(self, input_size: int, output_size: int,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 3,
                 patch_size: int = 2, dropout: float = 0.2):
        super(PatchTSTEncoder, self).__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        # Patch embedding
        self.patch_embedding = nn.Linear(input_size * patch_size, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape

        # Patching
        num_patches = seq_len // self.patch_size
        if num_patches == 0:
            num_patches = 1
            x_padded = torch.nn.functional.pad(x, (0, 0, 0, self.patch_size - seq_len))
            x = x_padded

        # Reshape for patching
        x = x[:, :num_patches * self.patch_size, :]
        x = x.reshape(batch_size, num_patches, self.patch_size * input_size)

        # Patch embedding
        x = self.patch_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :num_patches, :]

        # Transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output
        output = self.fc(x)
        return output


# ==================== Data Processing ====================
class DataProcessor:
    """데이터 로딩 및 전처리"""

    def __init__(self, pvd_name: str, config: Config):
        self.pvd_name = pvd_name
        self.config = config
        self.target_columns = config.PVD_TARGETS[pvd_name]
        self.scaler_input = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_target = MinMaxScaler(feature_range=(-1, 1))
        self.log_transform_cols = ['Ion.Gauge.i', 'Line.Gauge.i']  # Log 변환할 컬럼

    def load_csv_files(self) -> List[pd.DataFrame]:
        """CSV 파일들 로드"""
        data_path = os.path.join(self.config.BASE_DATA_PATH, self.pvd_name)
        csv_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)

        dataframes = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                # 필요한 컬럼이 모두 있는지 확인
                if all(col in df.columns for col in self.target_columns):
                    # Timer 컬럼 제외하고 숫자 컬럼만 사용
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0 and len(df) > self.config.INPUT_WINDOW + self.config.PREDICTION_HORIZON:
                        df_numeric = df[numeric_cols].copy()
                        # NaN 및 Inf 처리
                        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
                        df_numeric = df_numeric.ffill().bfill().fillna(0)
                        # Log 변환 (작은 값 컬럼에 적용)
                        for col in df_numeric.columns:
                            if col in self.log_transform_cols:
                                # Log1p 변환 (log(1+x))
                                df_numeric[col] = np.log1p(df_numeric[col].abs())
                        dataframes.append(df_numeric)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        print(f"[{self.pvd_name}] Loaded {len(dataframes)} valid CSV files")
        return dataframes

    def augment_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """데이터 증강 (연결 및 반복)"""
        # 각 DataFrame의 길이 확인
        total_rows = sum(len(df) for df in dataframes)

        # 증강 횟수 결정 (데이터가 적을수록 더 많이 증강)
        if total_rows < 1000:
            augmentation_factor = self.config.MAX_AUGMENTATION
        elif total_rows < 5000:
            augmentation_factor = 20
        else:
            augmentation_factor = self.config.MIN_AUGMENTATION

        print(f"[{self.pvd_name}] Total rows before augmentation: {total_rows}")
        print(f"[{self.pvd_name}] Augmentation factor: {augmentation_factor}")

        # 데이터 증강
        augmented_dfs = []
        for aug_idx in range(augmentation_factor):
            for df in dataframes:
                if aug_idx == 0:
                    # 첫번째는 원본 데이터
                    augmented_dfs.append(df.copy())
                else:
                    # 노이즈 추가 (약간의 변동성, 0이 아닌 값에만)
                    df_noisy = df.copy()
                    mask = df_noisy != 0
                    noise = np.random.normal(0, 0.005, df.shape)
                    df_noisy = df_noisy.where(~mask, df_noisy * (1 + noise))
                    augmented_dfs.append(df_noisy)

        # 모든 데이터 연결
        combined_df = pd.concat(augmented_dfs, ignore_index=True)
        print(f"[{self.pvd_name}] Total rows after augmentation: {len(combined_df)}")

        return combined_df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 생성"""
        # 입력 특성 (모든 숫자 컬럼)
        input_data = df.values.astype(np.float32)

        # 타겟 특성 (예측할 컬럼들)
        target_data = df[self.target_columns].values.astype(np.float32)

        # NaN/Inf 최종 체크
        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
        target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)

        # 스케일링
        input_scaled = self.scaler_input.fit_transform(input_data)
        target_scaled = self.scaler_target.fit_transform(target_data)

        # 스케일링 후 NaN 체크
        input_scaled = np.nan_to_num(input_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        sequences = []
        targets = []

        window = self.config.INPUT_WINDOW
        horizon = self.config.PREDICTION_HORIZON

        for i in range(len(input_scaled) - window - horizon + 1):
            seq = input_scaled[i:i + window]
            target = target_scaled[i + window + horizon - 1]
            # NaN이 없는 시퀀스만 추가
            if not (np.isnan(seq).any() or np.isnan(target).any()):
                sequences.append(seq)
                targets.append(target)

        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, int, int]:
        """전체 데이터 준비 파이프라인"""
        # 데이터 로드
        dataframes = self.load_csv_files()
        if len(dataframes) == 0:
            raise ValueError(f"No valid CSV files found for {self.pvd_name}")

        # 데이터 증강
        combined_df = self.augment_data(dataframes)

        # 시퀀스 생성
        sequences, targets = self.create_sequences(combined_df)
        print(f"[{self.pvd_name}] Created {len(sequences)} sequences")

        # Train/Validation 분할
        X_train, X_val, y_train, y_val = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )

        # Dataset 및 DataLoader 생성
        train_dataset = PVDDataset(X_train, y_train)
        val_dataset = PVDDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        input_size = sequences.shape[2]
        output_size = targets.shape[1]

        return train_loader, val_loader, input_size, output_size

    def save_scalers(self, save_dir: str):
        """스케일러 저장"""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler_input, os.path.join(save_dir, 'scaler_input.pkl'))
        joblib.dump(self.scaler_target, os.path.join(save_dir, 'scaler_target.pkl'))


# ==================== Training ====================
class Trainer:
    """모델 학습 클래스"""

    def __init__(self, model: nn.Module, device: torch.device, config: Config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        valid_batches = 0

        for sequences, targets in train_loader:
            # NaN 체크
            if torch.isnan(sequences).any() or torch.isnan(targets).any():
                continue

            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(sequences)

            # 출력 NaN 체크
            if torch.isnan(outputs).any():
                continue

            loss = self.criterion(outputs, targets)

            # Loss NaN 체크
            if torch.isnan(loss):
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            valid_batches += 1

        return total_loss / max(valid_batches, 1)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """검증"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss = total_loss / len(val_loader)

        # MAE 계산
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        mae = np.mean(np.abs(preds - targets))

        return val_loss, mae

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              pvd_name: str, model_type: str) -> Dict:
        """전체 학습 루프"""
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

        save_dir = os.path.join(self.config.SAVE_PATH, pvd_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Training {model_type} for {pvd_name}")
        print(f"{'='*50}")

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)

            self.scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.config.EPOCHS}] "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val MAE: {val_mae:.6f}")

            # Best model 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'epoch': epoch,
                }, os.path.join(save_dir, f'{model_type}_best.pth'))
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nBest validation loss: {best_val_loss:.6f}")
        return history


# ==================== Evaluation ====================
def evaluate_model(model: nn.Module, val_loader: DataLoader,
                   scaler_target: StandardScaler, device: torch.device,
                   target_columns: List[str]) -> Dict:
    """모델 평가"""
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

    # Inverse transform
    preds_orig = scaler_target.inverse_transform(preds)
    targets_orig = scaler_target.inverse_transform(targets)

    # 메트릭 계산
    results = {}
    for i, col in enumerate(target_columns):
        mse = np.mean((preds_orig[:, i] - targets_orig[:, i]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds_orig[:, i] - targets_orig[:, i]))

        # MAPE (0으로 나누기 방지)
        mask = targets_orig[:, i] != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((preds_orig[mask, i] - targets_orig[mask, i]) / targets_orig[mask, i])) * 100
        else:
            mape = 0

        results[col] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }

    return results


def print_evaluation_results(results: Dict, pvd_name: str):
    """평가 결과 출력"""
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {pvd_name}")
    print(f"{'='*60}")

    for col, metrics in results.items():
        print(f"\n{col}:")
        print(f"  MSE:  {metrics['MSE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")


# ==================== Main ====================
def train_pvd_model(pvd_name: str, model_type: str = 'lstm', gpu_id: int = 0):
    """단일 PVD 모델 학습"""
    config = Config()

    # Device 설정
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 데이터 준비
    processor = DataProcessor(pvd_name, config)
    train_loader, val_loader, input_size, output_size = processor.prepare_data()

    # 모델 생성
    if model_type.lower() == 'lstm':
        model = LSTMPredictor(
            input_size=input_size,
            output_size=output_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        )
    elif model_type.lower() == 'patchtst':
        model = PatchTSTEncoder(
            input_size=input_size,
            output_size=output_size,
            d_model=config.HIDDEN_SIZE,
            nhead=8,
            num_layers=3,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 학습
    trainer = Trainer(model, device, config)
    history = trainer.train(train_loader, val_loader, pvd_name, model_type)

    # 스케일러 저장
    save_dir = os.path.join(config.SAVE_PATH, pvd_name)
    processor.save_scalers(save_dir)

    # 최고 모델 로드 및 평가
    checkpoint = torch.load(os.path.join(save_dir, f'{model_type}_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    results = evaluate_model(
        model, val_loader, processor.scaler_target,
        device, processor.target_columns
    )
    print_evaluation_results(results, pvd_name)

    # 결과 저장
    results_path = os.path.join(save_dir, f'{model_type}_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Training Results for {pvd_name}\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Best Epoch: {checkpoint['epoch']}\n")
        f.write(f"Best Val Loss: {checkpoint['val_loss']:.6f}\n")
        f.write(f"Best Val MAE: {checkpoint['val_mae']:.6f}\n\n")

        for col, metrics in results.items():
            f.write(f"\n{col}:\n")
            f.write(f"  MSE:  {metrics['MSE']:.6f}\n")
            f.write(f"  RMSE: {metrics['RMSE']:.6f}\n")
            f.write(f"  MAE:  {metrics['MAE']:.6f}\n")
            f.write(f"  MAPE: {metrics['MAPE']:.2f}%\n")

    return model, processor, results


def main():
    parser = argparse.ArgumentParser(description='Train PVD Prediction Models')
    parser.add_argument('--pvd', type=str, default='all',
                        choices=['PVD1', 'PVD2', 'PVD3', 'PVD4', 'all'],
                        help='PVD chamber to train (default: all)')
    parser.add_argument('--model', type=str, default='lstm',
                        choices=['lstm', 'patchtst'],
                        help='Model architecture (default: lstm)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    pvd_list = ['PVD1', 'PVD2', 'PVD3', 'PVD4'] if args.pvd == 'all' else [args.pvd]

    for pvd_name in pvd_list:
        print(f"\n{'#'*60}")
        print(f"# Training model for {pvd_name}")
        print(f"{'#'*60}")

        try:
            train_pvd_model(pvd_name, args.model, args.gpu)
        except Exception as e:
            print(f"Error training {pvd_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
