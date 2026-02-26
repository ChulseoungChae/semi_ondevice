#!/usr/bin/env python3
"""
PVD4 단일 공정 파일 전체 추론 시각화
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

MODEL_DIR = "/home/goo4168/baco/train/models/PVD4"
DATA_PATH = "/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4"
OUTPUT_PATH = "/home/goo4168/baco/train/models/PVD4"

INPUT_WINDOW = 10
PREDICTION_HORIZON = 5

TARGET_COLUMNS = [
    'Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i', 'OES.Data6',
    'PLA5.Match.DCBias', 'SBRF5.Forward', 'SBRF5.Reflect'
]

LOG_TRANSFORM_COLS = ['Ion.Gauge.i', 'Line.Gauge.i']


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


def find_best_csv():
    """모든 타겟 컬럼에 의미있는 데이터가 있는 CSV 찾기"""
    csv_files = glob.glob(os.path.join(DATA_PATH, "**/*.csv"), recursive=True)

    best_file = None
    best_score = 0

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            if not all(col in df.columns for col in TARGET_COLUMNS):
                continue
            if len(df) < 50:  # 최소 50행
                continue

            # 각 타겟 컬럼의 비제로 비율 계산
            score = 0
            valid = True
            for col in TARGET_COLUMNS:
                non_zero_ratio = (df[col].abs() > 1e-10).mean()

                # 최소 10% 이상 비제로 데이터
                if non_zero_ratio < 0.10:
                    valid = False
                    break
                score += non_zero_ratio

            # 추가: 파일 크기(행 수)도 점수에 반영
            if valid:
                score = score * len(df)
                if score > best_score:
                    best_score = score
                    best_file = fpath

        except Exception as e:
            continue

    return best_file


def main():
    print("="*60)
    print("PVD4 Single Process Prediction Visualization")
    print("="*60)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 적합한 CSV 파일 찾기
    print("\n[1/5] Finding suitable CSV file...")
    csv_file = find_best_csv()
    if not csv_file:
        print("ERROR: No suitable CSV file found!")
        return

    print(f"Selected: {csv_file}")

    # 데이터 로드
    print("\n[2/5] Loading data...")
    df = pd.read_csv(csv_file)
    print(f"Rows: {len(df)}")

    # 전처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    for col in df_numeric.columns:
        if col in LOG_TRANSFORM_COLS:
            df_numeric[col] = np.log1p(df_numeric[col].abs())

    # 스케일러 로드
    scaler_input = joblib.load(os.path.join(MODEL_DIR, 'scaler_input_aug100_extended.pkl'))
    scaler_target = joblib.load(os.path.join(MODEL_DIR, 'scaler_target_aug100_extended.pkl'))

    # 시퀀스 생성
    input_data = df_numeric.values.astype(np.float32)
    target_data = df_numeric[TARGET_COLUMNS].values.astype(np.float32)

    input_scaled = scaler_input.transform(input_data)
    target_scaled = scaler_target.transform(target_data)

    sequences = []
    targets = []
    target_indices = []

    for i in range(len(input_scaled) - INPUT_WINDOW - PREDICTION_HORIZON + 1):
        seq = input_scaled[i:i + INPUT_WINDOW]
        target = target_scaled[i + INPUT_WINDOW + PREDICTION_HORIZON - 1]
        sequences.append(seq)
        targets.append(target)
        target_indices.append(i + INPUT_WINDOW + PREDICTION_HORIZON - 1)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    print(f"Sequences: {len(X)}")

    # 모델 로드
    print("\n[3/5] Loading model...")
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'lstm_aug100_extended_best.pth'), map_location=device)

    model = LSTMPredictor(X.shape[2], len(TARGET_COLUMNS)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'N/A')})")

    # 예측
    print("\n[4/5] Predicting...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_pred = model(X_tensor).cpu().numpy()

    # 역변환
    y_true_orig = scaler_target.inverse_transform(y)
    y_pred_orig = scaler_target.inverse_transform(y_pred)

    # 시각화
    print("\n[5/5] Creating visualization...")
    fig, axes = plt.subplots(7, 1, figsize=(16, 24))

    colors = {
        'actual': '#00d9ff',
        'predicted': '#ff9800'
    }

    filename = os.path.basename(csv_file)

    for idx, col in enumerate(TARGET_COLUMNS):
        ax = axes[idx]

        true_vals = y_true_orig[:, idx]
        pred_vals = y_pred_orig[:, idx]
        x_axis = np.array(target_indices)

        # 실제값, 예측값 플롯
        ax.plot(x_axis, true_vals, color=colors['actual'], linewidth=1.5, label='Actual', alpha=0.9)
        ax.plot(x_axis, pred_vals, color=colors['predicted'], linewidth=1.5, label='Predicted', linestyle='--', alpha=0.9)

        # 메트릭
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        mae = np.mean(np.abs(true_vals - pred_vals))
        mask = np.abs(true_vals) > 1e-10
        mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100 if mask.sum() > 0 else 0

        ax.set_facecolor('#1a1a2e')
        ax.set_title(f'{col}  |  RMSE: {rmse:.4f}  |  MAE: {mae:.4f}  |  MAPE: {mape:.2f}%',
                    fontsize=11, color='white', fontweight='bold', loc='left')
        ax.set_xlabel('Time Index', fontsize=9, color='#888')
        ax.set_ylabel('Value', fontsize=9, color='#888')
        ax.legend(loc='upper right', fontsize=9)
        ax.tick_params(colors='#888')
        ax.grid(True, alpha=0.2, color='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle(f'PVD4 Single Process Prediction\nFile: {filename}',
                fontsize=14, color='#00d9ff', fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_path = os.path.join(OUTPUT_PATH, 'PVD4_single_process_prediction.png')
    plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {save_path}")

    # 메트릭 출력
    print("\n" + "="*60)
    print(f"File: {filename}")
    print("="*60)
    for idx, col in enumerate(TARGET_COLUMNS):
        true_vals = y_true_orig[:, idx]
        pred_vals = y_pred_orig[:, idx]
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        mae = np.mean(np.abs(true_vals - pred_vals))
        mask = np.abs(true_vals) > 1e-10
        mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100 if mask.sum() > 0 else 0
        print(f"{col:25s} RMSE: {rmse:10.4f} | MAE: {mae:10.4f} | MAPE: {mape:7.2f}%")


if __name__ == '__main__':
    main()
