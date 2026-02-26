"""V2 CNN_aug1x: Actual vs Predicted — all 45 CSV files, full timeline."""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

from data_loader import parse_recipe_name
from models_v2 import PVD1DCNNModel
from train_v2 import V2_INPUT_COLS, V2_OUTPUT_COLS, V2_INPUT_WINDOW, V2_PREDICT_AHEAD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'

TYPE_COLORS = {'DC_only': '#1976D2', 'DC_RF': '#4CAF50', 'RF_only': '#FF5722'}


def load_model_and_scalers():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_best.pt'),
                      map_location=device, weights_only=False)
    model = PVD1DCNNModel(n_features=ckpt['config']['n_features'],
                          n_outputs=ckpt['config']['n_outputs'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    with open(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_scaler_x.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    return model, scaler_x, scaler_y, device


@torch.no_grad()
def predict_windows(X, model, scaler_x, scaler_y, device):
    n, s, f = X.shape
    X_sc = scaler_x.transform(X.reshape(-1, f)).reshape(n, s, f)
    preds_sc = model(torch.FloatTensor(X_sc).to(device)).cpu().numpy()
    return scaler_y.inverse_transform(preds_sc).flatten()


def create_windows_full(data, input_cols, output_col, input_window, predict_ahead):
    """Create windows from full (unfiltered) DataFrame.
    Returns: time_indices (target position), X, actuals
    """
    input_data = data[input_cols].values.astype(np.float64)
    output_data = data[output_col].values.astype(np.float64)

    n = len(data)
    X_list, actual_list, idx_list = [], [], []
    for i in range(n - input_window - predict_ahead + 1):
        target_idx = i + input_window + predict_ahead - 1
        X_list.append(input_data[i:i + input_window])
        actual_list.append(output_data[target_idx])
        idx_list.append(target_idx)

    return np.array(idx_list), np.array(X_list), np.array(actual_list)


def main():
    model, scaler_x, scaler_y, device = load_model_and_scalers()

    # Load all CSVs (full, no active filter)
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    all_files = []
    for f in files:
        recipe = parse_recipe_name(f)
        df = pd.read_csv(f)
        recipe['data'] = df
        recipe['total_rows'] = len(df)
        all_files.append(recipe)

    print(f"Loaded {len(all_files)} files (full timeline)")

    # Sort: DC_only → DC_RF → RF_only
    type_order = {'DC_only': 0, 'DC_RF': 1, 'RF_only': 2, 'unknown': 3}
    all_files.sort(key=lambda d: (type_order[d['recipe_type']],
                                  d['dc_setting'], d['rf_setting'], d['process_num']))

    # Build concatenated arrays
    all_actual_full = []   # every row's PWPDS.Data
    all_pred_full = []     # NaN where no prediction, value where predicted
    all_active_mask = []   # active flag per row
    segments = []
    offset = 0

    for d in all_files:
        df = d['data']
        n_rows = len(df)
        actual_full = df['PWPDS.Data'].values.astype(np.float64)
        active = ((df['EN4.Power'] > 0) | (df['SBRF5.SetPower'] > 0)).values

        # Predictions: create windows from full data
        pred_full = np.full(n_rows, np.nan)
        idx, X, _ = create_windows_full(
            df, V2_INPUT_COLS, 'PWPDS.Data', V2_INPUT_WINDOW, V2_PREDICT_AHEAD)

        if len(X) > 0:
            preds = predict_windows(X, model, scaler_x, scaler_y, device)
            pred_full[idx] = preds

        segments.append((offset, offset + n_rows, d['filename'], d['recipe_type'],
                         d['dc_setting'], d['rf_setting']))
        all_actual_full.append(actual_full)
        all_pred_full.append(pred_full)
        all_active_mask.append(active)
        offset += n_rows

    all_actual_full = np.concatenate(all_actual_full)
    all_pred_full = np.concatenate(all_pred_full)
    all_active_mask = np.concatenate(all_active_mask)
    total = len(all_actual_full)

    # Metrics on valid predictions only
    valid = ~np.isnan(all_pred_full)
    act = all_actual_full[valid]
    prd = all_pred_full[valid]
    mae = np.mean(np.abs(act - prd))
    rmse = np.sqrt(np.mean((act - prd) ** 2))
    m = np.abs(act) > 1e-6
    mape = np.mean(np.abs((act[m] - prd[m]) / act[m])) * 100

    print(f"Total rows: {total}, Predicted: {valid.sum()}")
    print(f"Global MAE={mae:.0f}, RMSE={rmse:.0f}, MAPE={mape:.4f}%")

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(30, 16),
                             gridspec_kw={'height_ratios': [4, 1.5, 0.6]})
    fig.suptitle(f'V2 CNN_aug1x: Actual vs Predicted — All 45 Files, Full Timeline\n'
                 f'Total {total:,} rows ({valid.sum():,} predicted)  |  '
                 f'MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.4f}%',
                 fontsize=15, fontweight='bold', y=0.99)

    x = np.arange(total)

    # --- Top: Actual vs Predicted ---
    ax = axes[0]

    # Idle shading
    idle_mask = ~all_active_mask
    changes = np.diff(idle_mask.astype(int))
    idle_starts = np.where(changes == 1)[0] + 1
    idle_ends = np.where(changes == -1)[0] + 1
    if idle_mask[0]:
        idle_starts = np.concatenate([[0], idle_starts])
    if idle_mask[-1]:
        idle_ends = np.concatenate([idle_ends, [total]])
    for s, e in zip(idle_starts, idle_ends):
        ax.axvspan(s, e, color='#E0E0E0', alpha=0.5, zorder=0)

    # Recipe type background
    for start, end, fname, rtype, dc, rf in segments:
        ax.axvspan(start, end, color=TYPE_COLORS[rtype], alpha=0.05, zorder=0)

    ax.plot(x, all_actual_full, color='#1976D2', linewidth=0.6, alpha=0.85, label='Actual', zorder=2)
    ax.plot(x, all_pred_full, color='#F44336', linewidth=0.6, alpha=0.75, label='Predicted', zorder=3)

    # File separators
    prev_rk = None
    for start, end, fname, rtype, dc, rf in segments:
        ax.axvline(start, color='gray', linewidth=0.3, alpha=0.4)
        rk = f"DC{dc}_RF{rf}" if dc > 0 else f"DCign_RF{rf}"
        if rk != prev_rk:
            ax.axvline(start, color='black', linewidth=0.7, alpha=0.5)
            prev_rk = rk

    ax.set_ylabel('PWPDS.Data', fontsize=12)
    ax.set_xlim(0, total)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(axis='y', alpha=0.3)

    legend_handles = [
        plt.Line2D([0], [0], color='#1976D2', linewidth=2, label='Actual'),
        plt.Line2D([0], [0], color='#F44336', linewidth=2, label='Predicted'),
        Patch(facecolor='#E0E0E0', alpha=0.7, label='Idle'),
        Patch(facecolor=TYPE_COLORS['DC_only'], alpha=0.3, label='DC only'),
        Patch(facecolor=TYPE_COLORS['DC_RF'], alpha=0.3, label='DC+RF'),
        Patch(facecolor=TYPE_COLORS['RF_only'], alpha=0.3, label='RF only'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower left', ncol=6)

    # --- Middle: Error ---
    ax2 = axes[1]
    error = all_actual_full - all_pred_full  # NaN where no prediction
    ax2.fill_between(x, error, 0, where=(error >= 0), color='#F44336', alpha=0.4, step='mid')
    ax2.fill_between(x, error, 0, where=(error < 0), color='#1976D2', alpha=0.4, step='mid')
    ax2.axhline(0, color='black', linewidth=0.5)

    for start, end, fname, rtype, dc, rf in segments:
        ax2.axvspan(start, end, color=TYPE_COLORS[rtype], alpha=0.05)
        ax2.axvline(start, color='gray', linewidth=0.3, alpha=0.4)

    for s, e in zip(idle_starts, idle_ends):
        ax2.axvspan(s, e, color='#E0E0E0', alpha=0.5, zorder=0)

    ax2.set_ylabel('Error\n(Actual − Pred)', fontsize=10)
    ax2.set_xlim(0, total)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax2.grid(axis='y', alpha=0.3)

    # --- Bottom: Recipe labels ---
    ax3 = axes[2]
    ax3.set_xlim(0, total)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # Group consecutive segments by recipe_key for labels
    recipe_groups = []
    prev_rk = None
    for start, end, fname, rtype, dc, rf in segments:
        rk = f"DC{dc}_RF{rf}" if dc > 0 else f"DCign_RF{rf}"
        if rk != prev_rk:
            recipe_groups.append({'key': rk, 'start': start, 'end': end, 'rtype': rtype})
            prev_rk = rk
        else:
            recipe_groups[-1]['end'] = end

    for grp in recipe_groups:
        ax3.axvspan(grp['start'], grp['end'], color=TYPE_COLORS[grp['rtype']], alpha=0.35)
        mid = (grp['start'] + grp['end']) / 2
        label = grp['key'].replace('_', '\n')
        ax3.text(mid, 0.5, label, ha='center', va='center', fontsize=6.5,
                 fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(RESULT_DIR, 'v2_CNN_aug1x_all_concat.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
