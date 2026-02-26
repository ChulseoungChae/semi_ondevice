"""
v3_CNN_alldata_aug50x 모델 레시피별 상세 평가 + 실제값 vs 추론값 차트
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

CONFIG_NAME = 'v3_CNN_alldata_aug50x'


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
    model, scaler_x, scaler_y, config = load_model(device)
    print(f"Model: {CONFIG_NAME}")

    all_data = load_all_csvs_full(DATA_DIR)

    # Group by recipe
    recipes = {}
    for d in all_data:
        key = f"DC{d['dc_setting']}_RF{d['rf_setting']}"
        if key not in recipes:
            recipes[key] = []
        recipes[key].append(d)

    # ── 1. 레시피별 상세 평가 ──
    results = []
    plot_data = {}  # recipe_key -> list of (filename, actual, preds)

    for recipe_key in sorted(recipes.keys()):
        procs = sorted(recipes[recipe_key], key=lambda x: x['process_num'])
        for proc in procs:
            actual, preds = predict_process(model, proc['data'], scaler_x, scaler_y, device)
            if actual is None:
                continue
            mae = mean_absolute_error(actual, preds)
            rmse = np.sqrt(mean_squared_error(actual, preds))
            r2 = r2_score(actual, preds)
            mask = np.abs(actual) > 1e-6
            mape = np.mean(np.abs((actual[mask] - preds[mask]) / actual[mask])) * 100

            results.append({
                'recipe': recipe_key, 'file': proc['filename'],
                'process_num': proc['process_num'],
                'n_windows': len(actual),
                'actual_mean': float(np.mean(actual)),
                'actual_std': float(np.std(actual)),
                'MAE': float(mae), 'RMSE': float(rmse),
                'R2': float(r2), 'MAPE': float(mape),
            })

            if recipe_key not in plot_data:
                plot_data[recipe_key] = []
            plot_data[recipe_key].append((proc['filename'], actual, preds))

    df_results = pd.DataFrame(results)

    print(f"\n{'='*100}")
    print("레시피별 상세 평가")
    print(f"{'='*100}")
    print(f"{'File':<25s} {'Windows':>7s} {'Mean':>12s} {'Std':>10s} "
          f"{'MAE':>10s} {'RMSE':>10s} {'R2':>8s} {'MAPE%':>8s}")
    print('-' * 100)
    for _, r in df_results.iterrows():
        print(f"{r['file']:<25s} {r['n_windows']:>7d} {r['actual_mean']:>12.0f} {r['actual_std']:>10.0f} "
              f"{r['MAE']:>10.0f} {r['RMSE']:>10.0f} {r['R2']:>8.4f} {r['MAPE']:>8.4f}")

    # 레시피 유형별 요약
    df_results['type'] = df_results['recipe'].apply(
        lambda x: 'DC_only' if 'RF0' in x else ('RF_only' if 'DC0' in x else 'DC+RF'))

    print(f"\n{'='*70}")
    print("레시피 유형별 요약")
    print(f"{'='*70}")
    for rtype in ['DC_only', 'DC+RF', 'RF_only']:
        sub = df_results[df_results['type'] == rtype]
        if len(sub) == 0:
            continue
        print(f"  [{rtype}] ({len(sub)} files)")
        print(f"    MAE  avg={sub['MAE'].mean():8.0f}  RMSE avg={sub['RMSE'].mean():8.0f}  "
              f"R2 avg={sub['R2'].mean():.6f}  MAPE avg={sub['MAPE'].mean():.4f}%")

    # 전체
    all_a = np.concatenate([a for pl in plot_data.values() for _, a, _ in pl])
    all_p = np.concatenate([p for pl in plot_data.values() for _, _, p in pl])
    print(f"\n  [전체] MAE={mean_absolute_error(all_a, all_p):.0f}  "
          f"RMSE={np.sqrt(mean_squared_error(all_a, all_p)):.0f}  "
          f"R2={r2_score(all_a, all_p):.6f}")

    # 결과 저장
    df_results.to_csv(os.path.join(RESULT_DIR, f'{CONFIG_NAME}_detailed_eval.csv'), index=False)

    # ── 2. 실제값 vs 추론값 차트 ──
    sorted_recipes = sorted(plot_data.keys())
    n_recipes = len(sorted_recipes)
    n_cols = 3
    n_rows = (n_recipes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.5 * n_rows))
    fig.suptitle(f'{CONFIG_NAME}: Actual vs Predicted (All 15 Recipes)', fontsize=16, y=1.0)
    axes = axes.flatten()

    for idx, recipe_key in enumerate(sorted_recipes):
        ax = axes[idx]
        offset = 0
        for fname, actual, preds in plot_data[recipe_key]:
            t = np.arange(offset, offset + len(actual))
            ax.plot(t, actual[:, 0], 'b-', linewidth=1.2, alpha=0.8)
            ax.plot(t, preds[:, 0], 'r--', linewidth=1.2, alpha=0.8)
            offset += len(actual) + 5  # gap between processes

        # Compute recipe-level metrics
        r_actual = np.concatenate([a for _, a, _ in plot_data[recipe_key]])
        r_preds = np.concatenate([p for _, _, p in plot_data[recipe_key]])
        r2 = r2_score(r_actual, r_preds)
        rmse = np.sqrt(mean_squared_error(r_actual, r_preds))

        ax.set_title(f'{recipe_key}\nR²={r2:.4f}  RMSE={rmse:.0f}', fontsize=10)
        ax.set_xlabel('Time step', fontsize=9)
        ax.set_ylabel('PWPDS.Data', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

        # Legend only on first
        if idx == 0:
            ax.plot([], [], 'b-', label='Actual')
            ax.plot([], [], 'r--', label='Predicted')
            ax.legend(fontsize=9, loc='upper right')

    for idx in range(n_recipes, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    chart_path = os.path.join(RESULT_DIR, f'{CONFIG_NAME}_predictions.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n차트 저장: {chart_path}")

    # ── 3. 성능 하위 5개 레시피 개별 확대 차트 ──
    worst5 = df_results.nlargest(5, 'RMSE')
    fig2, axes2 = plt.subplots(1, 5, figsize=(28, 4.5))
    fig2.suptitle(f'{CONFIG_NAME}: Worst 5 Recipes (Zoomed)', fontsize=14)

    for i, (_, row) in enumerate(worst5.iterrows()):
        ax = axes2[i]
        recipe_key = row['recipe']
        for fname, actual, preds in plot_data[recipe_key]:
            if fname == row['file']:
                t = np.arange(len(actual))
                ax.plot(t, actual[:, 0], 'b-', linewidth=1.5, label='Actual')
                ax.plot(t, preds[:, 0], 'r--', linewidth=1.5, label='Predicted')
                ax.fill_between(t, actual[:, 0], preds[:, 0], alpha=0.15, color='red')
                break
        ax.set_title(f"{row['file']}\nRMSE={row['RMSE']:.0f} R²={row['R2']:.4f}", fontsize=9)
        ax.set_xlabel('Time step', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    chart2_path = os.path.join(RESULT_DIR, f'{CONFIG_NAME}_worst5.png')
    plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"차트 저장: {chart2_path}")


if __name__ == '__main__':
    main()
