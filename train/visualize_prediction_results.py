#!/usr/bin/env python3
"""
PVD4 LSTM 모델 예측 결과 시각화
학습에 사용되지 않은 데이터로 예측 수행 후 PNG 저장
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 설정
MODEL_DIR = "/home/goo4168/baco/train/models/PVD4"
DATA_PATH = "/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4"
OUTPUT_PATH = "/home/goo4168/baco/train/models/PVD4"

INPUT_WINDOW = 10
PREDICTION_HORIZON = 5

TARGET_COLUMNS = [
    'Ar.MFC.i',
    'Ion.Gauge.i',
    'Baratron.Gauge.i',
    'OES.Data6',
    'PLA5.Match.DCBias',
    'SBRF5.Forward',
    'SBRF5.Reflect'
]

LOG_TRANSFORM_COLS = ['Ion.Gauge.i', 'Line.Gauge.i']


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
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


def load_and_prepare_data():
    """데이터 로드 및 전처리 (학습에 사용되지 않은 테스트 데이터 분리)"""
    csv_files = glob.glob(os.path.join(DATA_PATH, "**/*.csv"), recursive=True)

    dataframes = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if all(col in df.columns for col in TARGET_COLUMNS):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 0 and len(df) > INPUT_WINDOW + PREDICTION_HORIZON:
                    df_numeric = df[numeric_cols].copy()
                    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
                    df_numeric = df_numeric.ffill().bfill().fillna(0)
                    for col in df_numeric.columns:
                        if col in LOG_TRANSFORM_COLS:
                            df_numeric[col] = np.log1p(df_numeric[col].abs())
                    dataframes.append(df_numeric)
        except:
            continue

    print(f"Loaded {len(dataframes)} valid CSV files")

    # 스케일러 로드
    scaler_input = joblib.load(os.path.join(MODEL_DIR, 'scaler_input_aug100_extended.pkl'))
    scaler_target = joblib.load(os.path.join(MODEL_DIR, 'scaler_target_aug100_extended.pkl'))

    # 데이터 결합 (증강 없이)
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows: {len(combined_df)}")

    # 시퀀스 생성
    input_data = combined_df.values.astype(np.float32)
    target_data = combined_df[TARGET_COLUMNS].values.astype(np.float32)

    input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
    target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)

    input_scaled = scaler_input.transform(input_data)
    target_scaled = scaler_target.transform(target_data)

    sequences = []
    targets = []

    for i in range(len(input_scaled) - INPUT_WINDOW - PREDICTION_HORIZON + 1):
        seq = input_scaled[i:i + INPUT_WINDOW]
        target = target_scaled[i + INPUT_WINDOW + PREDICTION_HORIZON - 1]
        if not (np.isnan(seq).any() or np.isnan(target).any()):
            sequences.append(seq)
            targets.append(target)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    print(f"Created {len(sequences)} sequences")

    # Train/Test 분리 (학습 시와 동일한 seed 사용하여 테스트 데이터 추출)
    _, X_test, _, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )

    print(f"Test set size: {len(X_test)}")

    return X_test, y_test, scaler_target


def predict(model, X_test, device):
    """예측 수행"""
    model.eval()
    predictions = []

    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            output = model(batch)
            predictions.append(output.cpu().numpy())

    return np.concatenate(predictions, axis=0)


def create_visualization(y_true, y_pred, scaler_target, output_path):
    """예측 결과 시각화 및 PNG 저장"""

    # 역변환 (원본 스케일)
    y_true_orig = scaler_target.inverse_transform(y_true)
    y_pred_orig = scaler_target.inverse_transform(y_pred)

    # 7개 컬럼 시각화 (4행 2열)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    colors = ['#00d9ff', '#ff9800', '#4caf50', '#e91e63', '#9c27b0', '#00bcd4', '#ff5722']

    for idx, col in enumerate(TARGET_COLUMNS):
        ax = axes[idx]

        true_vals = y_true_orig[:, idx]
        pred_vals = y_pred_orig[:, idx]

        # 샘플링 (너무 많은 데이터 시각화 방지)
        n_samples = min(500, len(true_vals))
        indices = np.linspace(0, len(true_vals)-1, n_samples, dtype=int)

        true_sampled = true_vals[indices]
        pred_sampled = pred_vals[indices]
        x_axis = np.arange(n_samples)

        # 실제값과 예측값 플롯
        ax.plot(x_axis, true_sampled, color=colors[idx], alpha=0.7, linewidth=1, label='Actual')
        ax.plot(x_axis, pred_sampled, color='white', alpha=0.8, linewidth=1, linestyle='--', label='Predicted')

        # 메트릭 계산
        mse = np.mean((true_vals - pred_vals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_vals - pred_vals))

        mask = np.abs(true_vals) > 1e-10
        if mask.sum() > 0:
            mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100
        else:
            mape = 0

        # 스타일 설정
        ax.set_facecolor('#1a1a2e')
        ax.set_title(f'{col}\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%',
                     fontsize=11, color='white', fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=9, color='#888')
        ax.set_ylabel('Value', fontsize=9, color='#888')
        ax.legend(loc='upper right', fontsize=8)
        ax.tick_params(colors='#888')
        ax.grid(True, alpha=0.2, color='white')

        for spine in ax.spines.values():
            spine.set_color('#333')

    # 마지막 빈 subplot 숨기기
    axes[-1].set_visible(False)

    # 전체 figure 스타일
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle('PVD4 LSTM Prediction Results (Test Data)\nModel: lstm_aug100_extended',
                 fontsize=14, color='#00d9ff', fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 저장
    save_path = os.path.join(output_path, 'PVD4_prediction_results.png')
    plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {save_path}")

    return save_path


def main():
    print("="*60)
    print("PVD4 LSTM Prediction Visualization")
    print("="*60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 데이터 로드
    print("\n[1/4] Loading test data...")
    X_test, y_test, scaler_target = load_and_prepare_data()

    # 모델 로드
    print("\n[2/4] Loading model...")
    model_path = os.path.join(MODEL_DIR, 'lstm_aug100_extended_best.pth')
    checkpoint = torch.load(model_path, map_location=device)

    input_size = X_test.shape[2]
    output_size = len(TARGET_COLUMNS)

    model = LSTMPredictor(
        input_size=input_size,
        output_size=output_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Saved Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")

    # 예측
    print("\n[3/4] Running predictions...")
    y_pred = predict(model, X_test, device)

    # 시각화
    print("\n[4/4] Creating visualization...")
    save_path = create_visualization(y_test, y_pred, scaler_target, OUTPUT_PATH)

    # 최종 메트릭 출력
    print("\n" + "="*60)
    print("Final Metrics (Original Scale)")
    print("="*60)

    y_true_orig = scaler_target.inverse_transform(y_test)
    y_pred_orig = scaler_target.inverse_transform(y_pred)

    for idx, col in enumerate(TARGET_COLUMNS):
        true_vals = y_true_orig[:, idx]
        pred_vals = y_pred_orig[:, idx]

        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        mae = np.mean(np.abs(true_vals - pred_vals))

        mask = np.abs(true_vals) > 1e-10
        mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100 if mask.sum() > 0 else 0

        print(f"{col:25s} RMSE: {rmse:12.4f} | MAE: {mae:12.4f} | MAPE: {mape:8.2f}%")

    print("\n" + "="*60)
    print(f"Result saved to: {save_path}")
    print("="*60)


if __name__ == '__main__':
    main()
