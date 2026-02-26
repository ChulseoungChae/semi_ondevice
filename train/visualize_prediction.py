#!/usr/bin/env python3
"""
PVD 예측 모델 시각화 스크립트
- 학습된 모델로 실제 공정 데이터 추론
- 실제값 vs 예측값 비교 차트 생성
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 서버 환경용

from pvd_predictor import LSTMPredictor, Config


def load_model_and_scalers(pvd_name: str, model_type: str = 'lstm'):
    """모델과 스케일러 로드"""
    config = Config()
    model_dir = os.path.join(config.SAVE_PATH, pvd_name)

    # 스케일러 로드
    scaler_input = joblib.load(os.path.join(model_dir, 'scaler_input.pkl'))
    scaler_target = joblib.load(os.path.join(model_dir, 'scaler_target.pkl'))

    # 모델 로드
    model_path = os.path.join(model_dir, f'{model_type}_best.pth')
    checkpoint = torch.load(model_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')

    # 입력 크기 추론
    input_size = checkpoint['model_state_dict']['lstm.weight_ih_l0'].shape[1]
    output_size = len(config.PVD_TARGETS[pvd_name])

    # 모델 생성
    model = LSTMPredictor(
        input_size=input_size,
        output_size=output_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, scaler_input, scaler_target, config, device


def get_csv_file(pvd_name: str, scaler_input):
    """스케일러와 호환되는 CSV 파일 선택"""
    config = Config()
    data_path = os.path.join(config.BASE_DATA_PATH, pvd_name)
    csv_files = glob.glob(os.path.join(data_path, "**/*.csv"), recursive=True)

    if not csv_files:
        return None

    expected_features = scaler_input.n_features_in_
    target_columns = config.PVD_TARGETS[pvd_name]

    # 스케일러와 호환되는 파일 찾기
    compatible_files = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=5)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # 타겟 컬럼이 모두 있고, 피처 수가 맞는 파일
            if all(col in numeric_cols for col in target_columns):
                if len(numeric_cols) == expected_features:
                    compatible_files.append((f, os.path.getsize(f)))
        except:
            continue

    if not compatible_files:
        print(f"No compatible CSV found for {pvd_name} (expected {expected_features} features)")
        return None

    # 가장 큰 파일 선택
    compatible_files.sort(key=lambda x: x[1], reverse=True)
    return compatible_files[0][0]


def run_inference(pvd_name: str):
    """추론 및 시각화"""
    print(f"\n{'='*60}")
    print(f"Processing {pvd_name}")
    print(f"{'='*60}")

    # 모델 로드
    try:
        model, scaler_input, scaler_target, config, device = load_model_and_scalers(pvd_name)
    except FileNotFoundError as e:
        print(f"Model not found for {pvd_name}: {e}")
        return
    except Exception as e:
        print(f"Error loading model for {pvd_name}: {e}")
        return

    target_columns = config.PVD_TARGETS[pvd_name]

    # CSV 파일 로드
    csv_file = get_csv_file(pvd_name, scaler_input)
    if csv_file is None:
        print(f"No compatible CSV file found for {pvd_name}")
        return

    print(f"Using file: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"Data shape: {df.shape}")

    # 숫자 컬럼만 사용
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()

    # Log 변환 (학습 시와 동일하게)
    log_transform_cols = ['Ion.Gauge.i', 'Line.Gauge.i']
    for col in df_numeric.columns:
        if col in log_transform_cols:
            df_numeric[col] = np.log1p(df_numeric[col].abs())

    # NaN 처리
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.ffill().bfill().fillna(0)

    input_data = df_numeric.values.astype(np.float32)

    # 타겟 컬럼이 있는지 확인
    missing_cols = [col for col in target_columns if col not in df_numeric.columns]
    if missing_cols:
        print(f"Missing target columns: {missing_cols}")
        return

    target_data = df_numeric[target_columns].values.astype(np.float32)

    # 스케일링
    input_scaled = scaler_input.transform(input_data)

    # 시퀀스 생성 및 예측
    window = config.INPUT_WINDOW
    horizon = config.PREDICTION_HORIZON

    predictions = []
    actuals = []
    indices = []

    print(f"Running inference... (window={window}, horizon={horizon})")

    with torch.no_grad():
        for i in range(len(input_scaled) - window - horizon + 1):
            seq = input_scaled[i:i + window]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)

            pred = model(seq_tensor).cpu().numpy()[0]
            predictions.append(pred)

            # 실제값 (5초 후)
            actual_idx = i + window + horizon - 1
            actuals.append(target_data[actual_idx])
            indices.append(actual_idx)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 역변환
    predictions_orig = scaler_target.inverse_transform(predictions)

    # Log 역변환 (Ion.Gauge.i가 타겟에 있으면)
    actuals_orig = actuals.copy()
    for i, col in enumerate(target_columns):
        if col in log_transform_cols:
            # Log1p 역변환: expm1(x) = exp(x) - 1
            predictions_orig[:, i] = np.expm1(predictions_orig[:, i])
            actuals_orig[:, i] = np.expm1(actuals_orig[:, i])

    print(f"Generated {len(predictions)} predictions")

    # 시각화
    n_targets = len(target_columns)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 4 * n_targets))

    if n_targets == 1:
        axes = [axes]

    for i, (col, ax) in enumerate(zip(target_columns, axes)):
        ax.plot(indices, actuals_orig[:, i], 'b-', label='Actual', alpha=0.7, linewidth=1)
        ax.plot(indices, predictions_orig[:, i], 'r--', label='Predicted', alpha=0.7, linewidth=1)

        ax.set_xlabel('Time Index')
        ax.set_ylabel(col)
        ax.set_title(f'{pvd_name} - {col}: Actual vs Predicted (5-step ahead)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 메트릭 계산
        mse = np.mean((predictions_orig[:, i] - actuals_orig[:, i]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_orig[:, i] - actuals_orig[:, i]))

        # MAPE (0 제외)
        mask = actuals_orig[:, i] != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((predictions_orig[mask, i] - actuals_orig[mask, i]) / actuals_orig[mask, i])) * 100
        else:
            mape = 0

        # 메트릭 텍스트 추가
        metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.2f}%'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 저장
    save_path = os.path.join(config.SAVE_PATH, pvd_name, f'{pvd_name}_prediction_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Chart saved: {save_path}")

    return save_path


def main():
    """메인 함수"""
    # 학습된 모델이 있는 PVD만 처리
    pvd_list = ['PVD1', 'PVD3', 'PVD4']  # PVD2는 스케일러가 없음

    saved_charts = []

    for pvd_name in pvd_list:
        try:
            chart_path = run_inference(pvd_name)
            if chart_path:
                saved_charts.append(chart_path)
        except Exception as e:
            print(f"Error processing {pvd_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Generated {len(saved_charts)} charts:")
    for path in saved_charts:
        print(f"  - {path}")


if __name__ == '__main__':
    main()
