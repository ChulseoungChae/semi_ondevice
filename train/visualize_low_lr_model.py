#!/usr/bin/env python3
"""
Low LR 모델 단일 공정 예측 시각화
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

MODEL_DIR = "/home/goo4168/baco/train/models/PVD4"
DATA_PATH = "/home/goo4168/baco/minio_csv/Baco_origin_3/DataLog/PVD4"

TARGET_COLUMNS = ['Ar.MFC.i', 'Ion.Gauge.i', 'Baratron.Gauge.i', 'OES.Data6',
                  'PLA5.Match.DCBias', 'SBRF5.Forward', 'SBRF5.Reflect']
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
    csv_files = glob.glob(os.path.join(DATA_PATH, "**/*.csv"), recursive=True)
    best_file, best_score = None, 0

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            if not all(col in df.columns for col in TARGET_COLUMNS) or len(df) < 50:
                continue
            score = sum((df[col].abs() > 1e-10).mean() for col in TARGET_COLUMNS)
            if all((df[col].abs() > 1e-10).mean() >= 0.10 for col in TARGET_COLUMNS):
                score *= len(df)
                if score > best_score:
                    best_score, best_file = score, fpath
        except:
            continue
    return best_file


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # CSV 찾기
    csv_file = find_best_csv()
    print(f"File: {os.path.basename(csv_file)}")

    # 데이터 로드 및 전처리
    df = pd.read_csv(csv_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    for col in df_numeric.columns:
        if col in LOG_TRANSFORM_COLS:
            df_numeric[col] = np.log1p(df_numeric[col].abs())

    # 스케일러 로드
    scaler_input = joblib.load(os.path.join(MODEL_DIR, 'scaler_input_low_lr.pkl'))
    scaler_target = joblib.load(os.path.join(MODEL_DIR, 'scaler_target_low_lr.pkl'))

    # 시퀀스 생성
    input_scaled = scaler_input.transform(df_numeric.values.astype(np.float32))
    target_scaled = scaler_target.transform(df_numeric[TARGET_COLUMNS].values.astype(np.float32))

    sequences, targets, indices = [], [], []
    for i in range(len(input_scaled) - 14):
        sequences.append(input_scaled[i:i+10])
        targets.append(target_scaled[i+14])
        indices.append(i+14)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    # 모델 로드
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'lstm_low_lr_best.pth'), map_location=device)
    model = LSTMPredictor(X.shape[2], len(TARGET_COLUMNS)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 예측
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X).to(device)).cpu().numpy()

    # 역변환
    y_true_orig = scaler_target.inverse_transform(y)
    y_pred_orig = scaler_target.inverse_transform(y_pred)

    # 시각화
    fig, axes = plt.subplots(7, 1, figsize=(16, 24))
    filename = os.path.basename(csv_file)

    for idx, col in enumerate(TARGET_COLUMNS):
        ax = axes[idx]
        true_vals, pred_vals = y_true_orig[:, idx], y_pred_orig[:, idx]

        ax.plot(indices, true_vals, color='#00d9ff', linewidth=1.5, label='Actual', alpha=0.9)
        ax.plot(indices, pred_vals, color='#ff9800', linewidth=1.5, label='Predicted', linestyle='--', alpha=0.9)

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
    fig.suptitle(f'PVD4 Low LR Model (0.0001) Prediction\nFile: {filename}',
                fontsize=14, color='#00d9ff', fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(MODEL_DIR, 'PVD4_low_lr_prediction.png')
    plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")

    # 메트릭 출력
    print("\n" + "="*60)
    for idx, col in enumerate(TARGET_COLUMNS):
        true_vals, pred_vals = y_true_orig[:, idx], y_pred_orig[:, idx]
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        mae = np.mean(np.abs(true_vals - pred_vals))
        mask = np.abs(true_vals) > 1e-10
        mape = np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100 if mask.sum() > 0 else 0
        print(f"{col:25s} RMSE: {rmse:10.4f} | MAE: {mae:10.4f} | MAPE: {mape:7.2f}%")


if __name__ == '__main__':
    main()
