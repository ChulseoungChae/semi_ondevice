"""V2 CNN_aug1x: Per-recipe performance evaluation on all 45 CSV files."""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from data_loader import load_all_csvs, parse_recipe_name, create_windows
from models_v2 import PVD1DCNNModel
from train_v2 import V2_INPUT_COLS, V2_OUTPUT_COLS, V2_INPUT_WINDOW, V2_PREDICT_AHEAD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'


def load_model_and_scalers():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_best.pt'),
                            map_location=device, weights_only=False)
    model = PVD1DCNNModel(n_features=checkpoint['config']['n_features'],
                          n_outputs=checkpoint['config']['n_outputs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    with open(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_scaler_x.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_x, scaler_y, device


@torch.no_grad()
def predict_windows(X, model, scaler_x, scaler_y, device):
    n_samples, seq_len, n_feat = X.shape
    X_scaled = scaler_x.transform(X.reshape(-1, n_feat)).reshape(n_samples, seq_len, n_feat)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    preds_scaled = model(X_tensor).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    return preds


def main():
    model, scaler_x, scaler_y, device = load_model_and_scalers()
    all_data = load_all_csvs(DATA_DIR)

    rows = []
    for d in all_data:
        X, y = create_windows(d['data'], V2_INPUT_COLS, V2_OUTPUT_COLS,
                              input_window=V2_INPUT_WINDOW, predict_ahead=V2_PREDICT_AHEAD)
        if len(X) == 0:
            continue

        preds = predict_windows(X, model, scaler_x, scaler_y, device)
        actuals = y.flatten()

        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        mask = np.abs(actuals) > 1e-6
        mape = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100
        mean_val = np.mean(actuals)
        std_val = np.std(actuals)

        # Train/test label: process (3) = test, (1)(2) = train
        split = 'test' if d['process_num'] == max(
            p['process_num'] for p in all_data
            if p['dc_setting'] == d['dc_setting'] and p['rf_setting'] == d['rf_setting']
        ) else 'train'

        recipe_key = f"DC{d['dc_setting']}" if d['dc_setting'] > 0 else "DCign"
        recipe_key += f"_RF{d['rf_setting']}"

        rows.append({
            'filename': d['filename'],
            'recipe_key': recipe_key,
            'recipe_type': d['recipe_type'],
            'DC': d['dc_setting'],
            'RF': d['rf_setting'],
            'proc': d['process_num'],
            'split': split,
            'samples': len(X),
            'mean': mean_val,
            'std': std_val,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
        })

    df = pd.DataFrame(rows)

    # ── Print per-file table ──
    print(f"\n{'='*110}")
    print("Per-File Performance (v2_CNN_aug1x)")
    print(f"{'='*110}")
    print(f"{'File':<22s} {'Type':<8s} {'Split':<6s} {'N':>4s} "
          f"{'Mean':>12s} {'Std':>10s} {'MAE':>10s} {'RMSE':>10s} {'MAPE%':>10s}")
    print('-' * 110)
    for _, r in df.sort_values(['recipe_type', 'DC', 'RF', 'proc']).iterrows():
        print(f"{r['filename']:<22s} {r['recipe_type']:<8s} {r['split']:<6s} {r['samples']:>4d} "
              f"{r['mean']:>12,.0f} {r['std']:>10,.0f} {r['MAE']:>10,.0f} "
              f"{r['RMSE']:>10,.0f} {r['MAPE']:>9.4f}%")

    # ── Per-recipe group summary ──
    print(f"\n\n{'='*100}")
    print("Per-Recipe Summary (averaged across 3 processes)")
    print(f"{'='*100}")

    group = df.groupby(['recipe_key', 'recipe_type', 'DC', 'RF']).agg(
        files=('filename', 'count'),
        mean_val=('mean', 'mean'),
        std_val=('std', 'mean'),
        MAE=('MAE', 'mean'),
        RMSE=('RMSE', 'mean'),
        MAPE=('MAPE', 'mean'),
    ).reset_index().sort_values(['recipe_type', 'DC', 'RF'])

    print(f"{'Recipe':<16s} {'Type':<8s} {'Files':>5s} "
          f"{'Mean':>12s} {'Std':>10s} {'MAE':>10s} {'RMSE':>10s} {'MAPE%':>10s}")
    print('-' * 100)
    for _, r in group.iterrows():
        print(f"{r['recipe_key']:<16s} {r['recipe_type']:<8s} {r['files']:>5d} "
              f"{r['mean_val']:>12,.0f} {r['std_val']:>10,.0f} {r['MAE']:>10,.0f} "
              f"{r['RMSE']:>10,.0f} {r['MAPE']:>9.4f}%")

    # ── Save CSV ──
    csv_path = os.path.join(RESULT_DIR, 'v2_CNN_aug1x_per_file.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Visualization ──
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('V2 CNN_aug1x: Per-Recipe Performance (All 45 Files)',
                 fontsize=16, fontweight='bold', y=0.99)

    type_colors = {'DC_only': '#1976D2', 'DC_RF': '#4CAF50', 'RF_only': '#FF5722'}
    split_markers = {'train': 'o', 'test': 's'}

    # ── (1) RMSE per file, grouped by recipe ──
    ax = axes[0, 0]
    recipes_sorted = df.sort_values(['recipe_type', 'DC', 'RF', 'proc'])
    x_pos = np.arange(len(recipes_sorted))
    colors = [type_colors[r['recipe_type']] for _, r in recipes_sorted.iterrows()]
    alphas = [1.0 if r['split'] == 'test' else 0.55 for _, r in recipes_sorted.iterrows()]
    edgecolors = ['black' if r['split'] == 'test' else 'none' for _, r in recipes_sorted.iterrows()]

    bars = ax.bar(x_pos, recipes_sorted['RMSE'].values, color=colors, edgecolor=edgecolors, linewidth=1.2)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    # Recipe group separators
    prev_key = None
    tick_positions = []
    tick_labels = []
    group_starts = []
    for i, (_, r) in enumerate(recipes_sorted.iterrows()):
        if r['recipe_key'] != prev_key:
            if prev_key is not None:
                ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
            group_starts.append(i)
            prev_key = r['recipe_key']
    for j, start in enumerate(group_starts):
        end = group_starts[j + 1] if j + 1 < len(group_starts) else len(recipes_sorted)
        mid = (start + end - 1) / 2
        key = recipes_sorted.iloc[start]['recipe_key']
        tick_positions.append(mid)
        tick_labels.append(key.replace('_', '\n'))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7.5, rotation=0)
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE per File (dark border = test set)', fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # Legend for recipe types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.replace('_', ' ')) for t, c in type_colors.items()]
    legend_elements += [Patch(facecolor='gray', edgecolor='black', linewidth=1.5, label='test'),
                        Patch(facecolor='gray', alpha=0.5, label='train')]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    # ── (2) RMSE per recipe group (bar chart) ──
    ax = axes[0, 1]
    grp = group.copy()
    x_pos = np.arange(len(grp))
    bar_colors = [type_colors[r['recipe_type']] for _, r in grp.iterrows()]

    bars = ax.bar(x_pos, grp['RMSE'].values, color=bar_colors, edgecolor='black', linewidth=0.8)
    for i, v in enumerate(grp['RMSE'].values):
        ax.text(i, v + 200, f'{v:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r['recipe_key'].replace('_', '\n') for _, r in grp.iterrows()],
                       fontsize=7.5, rotation=0)
    ax.set_ylabel('RMSE (avg of 3 processes)')
    ax.set_title('RMSE per Recipe (averaged)', fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # ── (3) MAPE per recipe group ──
    ax = axes[1, 0]
    bars = ax.bar(x_pos, grp['MAPE'].values, color=bar_colors, edgecolor='black', linewidth=0.8)
    for i, v in enumerate(grp['MAPE'].values):
        ax.text(i, v + 0.0001, f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r['recipe_key'].replace('_', '\n') for _, r in grp.iterrows()],
                       fontsize=7.5, rotation=0)
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE per Recipe (averaged)', fontweight='bold')

    # ── (4) Scatter: Std vs RMSE ──
    ax = axes[1, 1]
    for _, r in df.iterrows():
        c = type_colors[r['recipe_type']]
        m = split_markers[r['split']]
        ax.scatter(r['std'], r['RMSE'], c=c, marker=m, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('PWPDS.Data Std (within file)')
    ax.set_ylabel('RMSE')
    ax.set_title('Target Variability vs RMSE', fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8,
                      label=t.replace('_', ' ')) for t, c in type_colors.items()]
    handles += [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='train'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='test')]
    ax.legend(handles=handles, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(RESULT_DIR, 'v2_CNN_aug1x_per_recipe.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
