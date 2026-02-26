"""
v3_CNN_aug20x 모델 상세 성능 평가
- 전체 메트릭 + 레시피별 메트릭
- 학습 히스토리 분석
- 오차 분포 분석
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import parse_recipe_name
from models_v2 import PVD1DCNNModel
from train_v3 import (load_all_csvs_full, create_windows,
                       V3_INPUT_COLS, V3_OUTPUT_COLS, V3_INPUT_WINDOW, V3_PREDICT_AHEAD)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

CONFIG_NAME = 'v3_CNN_aug20x'


def load_model(device):
    ckpt = torch.load(os.path.join(MODEL_DIR, f'{CONFIG_NAME}_best.pt'),
                      map_location=device, weights_only=False)
    with open(os.path.join(MODEL_DIR, f'{CONFIG_NAME}_scaler_x.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f'{CONFIG_NAME}_scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    config = ckpt['config']
    model = PVD1DCNNModel(n_features=config['n_features'], n_outputs=config['n_outputs'])
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, scaler_x, scaler_y, config


@torch.no_grad()
def predict_process(model, df, scaler_x, scaler_y, device):
    X, y = create_windows(df, V3_INPUT_COLS, V3_OUTPUT_COLS, V3_INPUT_WINDOW, V3_PREDICT_AHEAD)
    if len(X) == 0:
        return None, None

    n, seq, nf = X.shape
    no = y.shape[1] if y.ndim > 1 else 1
    X_sc = scaler_x.transform(X.reshape(-1, nf)).reshape(n, seq, nf)
    preds_sc = model(torch.FloatTensor(X_sc).to(device)).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_sc)
    actual = y.reshape(-1, no)
    return actual, preds


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, scaler_x, scaler_y, config = load_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {CONFIG_NAME}, Params: {total_params:,}")
    print(f"Input features: {V3_INPUT_COLS}")
    print(f"Input window: {V3_INPUT_WINDOW}, Predict ahead: {V3_PREDICT_AHEAD}")

    # ── 1. Training History 분석 ──
    with open(os.path.join(RESULT_DIR, f'{CONFIG_NAME}_history.json')) as f:
        history = json.load(f)

    print(f"\n{'='*70}")
    print("1. Training History 분석")
    print(f"{'='*70}")
    print(f"총 에폭: {len(history['train_loss'])}")
    best_epoch = np.argmin(history['val_loss']) + 1
    print(f"Best epoch (min val_loss): {best_epoch}")
    print(f"Best val_loss: {min(history['val_loss']):.6f}")
    print(f"Final train_loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val_loss: {history['val_loss'][-1]:.6f}")

    # Train vs Val loss gap
    train_losses = np.array(history['train_loss'])
    val_losses = np.array(history['val_loss'])
    gap = val_losses - train_losses
    print(f"\nTrain-Val gap (mean): {np.mean(gap):.6f}")
    print(f"Train-Val gap (at best): {gap[best_epoch-1]:.6f}")
    print(f"Val loss 변동성 (std): {np.std(val_losses):.6f}")
    print(f"Val loss min: {np.min(val_losses):.6f}, max: {np.max(val_losses):.6f}")

    # ── 2. 전체 데이터 레시피별 평가 ──
    print(f"\n{'='*70}")
    print("2. 레시피별 상세 평가 (전체 45개 파일)")
    print(f"{'='*70}")

    all_data = load_all_csvs_full(DATA_DIR)

    # Group by recipe
    recipes = {}
    for d in all_data:
        key = f"DC{d['dc_setting']}_RF{d['rf_setting']}"
        if key not in recipes:
            recipes[key] = []
        recipes[key].append(d)

    results = []
    all_actual, all_preds = [], []

    for recipe_key in sorted(recipes.keys()):
        for proc in recipes[recipe_key]:
            actual, preds = predict_process(model, proc['data'], scaler_x, scaler_y, device)
            if actual is None:
                continue

            mae = mean_absolute_error(actual, preds)
            rmse = np.sqrt(mean_squared_error(actual, preds))
            r2 = r2_score(actual, preds)
            mask = np.abs(actual) > 1e-6
            mape = np.mean(np.abs((actual[mask] - preds[mask]) / actual[mask])) * 100

            results.append({
                'recipe': recipe_key,
                'file': proc['filename'],
                'process_num': proc['process_num'],
                'n_windows': len(actual),
                'actual_mean': float(np.mean(actual)),
                'actual_std': float(np.std(actual)),
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
                'MAPE': float(mape),
            })

            all_actual.append(actual)
            all_preds.append(preds)

    df_results = pd.DataFrame(results)

    # Print per-recipe
    print(f"\n{'File':<25s} {'Windows':>7s} {'Mean':>10s} {'Std':>10s} "
          f"{'MAE':>10s} {'RMSE':>10s} {'R2':>8s} {'MAPE%':>8s}")
    print('-' * 90)
    for _, r in df_results.iterrows():
        print(f"{r['file']:<25s} {r['n_windows']:>7d} {r['actual_mean']:>10.0f} {r['actual_std']:>10.0f} "
              f"{r['MAE']:>10.0f} {r['RMSE']:>10.0f} {r['R2']:>8.4f} {r['MAPE']:>8.4f}")

    # ── 3. 전체 통합 메트릭 ──
    print(f"\n{'='*70}")
    print("3. 전체 통합 성능 메트릭")
    print(f"{'='*70}")

    all_a = np.concatenate(all_actual)
    all_p = np.concatenate(all_preds)
    overall_mae = mean_absolute_error(all_a, all_p)
    overall_rmse = np.sqrt(mean_squared_error(all_a, all_p))
    overall_r2 = r2_score(all_a, all_p)
    mask = np.abs(all_a) > 1e-6
    overall_mape = np.mean(np.abs((all_a[mask] - all_p[mask]) / all_a[mask])) * 100

    print(f"MAE:  {overall_mae:.2f}")
    print(f"RMSE: {overall_rmse:.2f}")
    print(f"R2:   {overall_r2:.6f}")
    print(f"MAPE: {overall_mape:.4f}%")

    # ── 4. Train/Test 분리 메트릭 ──
    print(f"\n{'='*70}")
    print("4. Train / Test 분리 성능")
    print(f"{'='*70}")

    train_actual, train_preds = [], []
    test_actual, test_preds = [], []

    for recipe_key in sorted(recipes.keys()):
        procs = sorted(recipes[recipe_key], key=lambda x: x['process_num'])
        for i, proc in enumerate(procs):
            actual, preds = predict_process(model, proc['data'], scaler_x, scaler_y, device)
            if actual is None:
                continue
            if i < len(procs) - 1:
                train_actual.append(actual)
                train_preds.append(preds)
            else:
                test_actual.append(actual)
                test_preds.append(preds)

    for split, a_list, p_list in [('Train', train_actual, train_preds),
                                   ('Test', test_actual, test_preds)]:
        a = np.concatenate(a_list)
        p = np.concatenate(p_list)
        mae = mean_absolute_error(a, p)
        rmse = np.sqrt(mean_squared_error(a, p))
        r2 = r2_score(a, p)
        m = np.abs(a) > 1e-6
        mape = np.mean(np.abs((a[m] - p[m]) / a[m])) * 100
        print(f"  {split:5s}: MAE={mae:10.2f}  RMSE={rmse:10.2f}  R2={r2:.6f}  MAPE={mape:.4f}%  N={len(a)}")

    # ── 5. 레시피 유형별 분석 ──
    print(f"\n{'='*70}")
    print("5. 레시피 유형별 성능")
    print(f"{'='*70}")

    df_results['type'] = df_results['recipe'].apply(
        lambda x: 'DC_only' if 'RF0' in x else ('RF_only' if 'DC0' in x else 'DC+RF'))

    for rtype in ['DC_only', 'DC+RF', 'RF_only']:
        sub = df_results[df_results['type'] == rtype]
        if len(sub) == 0:
            continue
        print(f"\n  [{rtype}] ({len(sub)} files)")
        print(f"    MAE  avg={sub['MAE'].mean():10.2f}  min={sub['MAE'].min():10.2f}  max={sub['MAE'].max():10.2f}")
        print(f"    RMSE avg={sub['RMSE'].mean():10.2f}  min={sub['RMSE'].min():10.2f}  max={sub['RMSE'].max():10.2f}")
        print(f"    R2   avg={sub['R2'].mean():10.6f}  min={sub['R2'].min():10.6f}  max={sub['R2'].max():10.6f}")
        print(f"    MAPE avg={sub['MAPE'].mean():10.4f}%  min={sub['MAPE'].min():10.4f}%  max={sub['MAPE'].max():10.4f}%")

    # ── 6. 오차 분포 ──
    print(f"\n{'='*70}")
    print("6. 오차 분포 분석")
    print(f"{'='*70}")

    errors = np.abs(all_a.flatten() - all_p.flatten())
    pct_errors = errors / np.abs(all_a.flatten() + 1e-10) * 100

    print(f"  절대오차: mean={np.mean(errors):.2f}  std={np.std(errors):.2f}  "
          f"P50={np.percentile(errors, 50):.2f}  P95={np.percentile(errors, 95):.2f}  "
          f"P99={np.percentile(errors, 99):.2f}  max={np.max(errors):.2f}")
    print(f"  상대오차: mean={np.mean(pct_errors):.4f}%  std={np.std(pct_errors):.4f}%  "
          f"P50={np.percentile(pct_errors, 50):.4f}%  P95={np.percentile(pct_errors, 95):.4f}%  "
          f"P99={np.percentile(pct_errors, 99):.4f}%  max={np.max(pct_errors):.4f}%")

    # ── 7. 문제 진단 & 개선 방안 ──
    print(f"\n{'='*70}")
    print("7. 문제점 진단 & 개선 방안")
    print(f"{'='*70}")

    # Check overfitting
    train_a = np.concatenate(train_actual)
    train_p = np.concatenate(train_preds)
    test_a = np.concatenate(test_actual)
    test_p = np.concatenate(test_preds)
    train_rmse = np.sqrt(mean_squared_error(train_a, train_p))
    test_rmse = np.sqrt(mean_squared_error(test_a, test_p))

    print(f"\n  [과적합 진단]")
    print(f"    Train RMSE: {train_rmse:.2f}")
    print(f"    Test RMSE:  {test_rmse:.2f}")
    print(f"    Ratio (Test/Train): {test_rmse/train_rmse:.4f}")
    if test_rmse / train_rmse > 1.5:
        print(f"    ⚠ 과적합 의심: Test/Train RMSE 비율이 높음")
    else:
        print(f"    ✓ 과적합 수준 양호")

    print(f"\n  [Early Stopping 진단]")
    print(f"    Best epoch: {best_epoch} / {len(history['train_loss'])}")
    if best_epoch <= 5:
        print(f"    ⚠ Best epoch이 매우 이름 - 학습률이 너무 높거나 모델이 빠르게 수렴")

    print(f"\n  [Val Loss 불안정성]")
    print(f"    Val loss 표준편차: {np.std(val_losses):.6f}")
    print(f"    Val loss 범위: {np.min(val_losses):.6f} ~ {np.max(val_losses):.6f}")
    if np.std(val_losses) > np.mean(val_losses) * 0.3:
        print(f"    ⚠ Validation loss 변동이 큼 - 배치 크기 증가 또는 학습률 감소 필요")

    # Worst performing recipes
    print(f"\n  [성능 하위 레시피]")
    worst = df_results.nlargest(5, 'RMSE')
    for _, r in worst.iterrows():
        print(f"    {r['file']}: RMSE={r['RMSE']:.0f}, R2={r['R2']:.4f}, MAPE={r['MAPE']:.4f}%")

    # Save results
    df_results.to_csv(os.path.join(RESULT_DIR, f'{CONFIG_NAME}_detailed_eval.csv'), index=False)
    print(f"\n결과 저장: {os.path.join(RESULT_DIR, f'{CONFIG_NAME}_detailed_eval.csv')}")


if __name__ == '__main__':
    main()
