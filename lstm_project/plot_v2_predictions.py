"""V2 CNN_aug1x: Actual vs Predicted PWPDS.Data for 3 selected CSV files."""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from data_loader import parse_recipe_name
from models_v2 import PVD1DCNNModel
from train_v2 import V2_INPUT_COLS, V2_OUTPUT_COLS, V2_INPUT_WINDOW, V2_PREDICT_AHEAD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'

# 3 representative files: DC-only, DC+RF, RF-only
TARGET_FILES = [
    'DC2000(3).csv',        # DC-only
    'DC3000RF500(3).csv',   # DC+RF (high power)
    'DCignRF400(3).csv',    # RF-only
]


def load_model_and_scalers():
    """Load trained CNN_aug1x model and scalers."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_best.pt'),
                            map_location=device, weights_only=False)
    n_features = checkpoint['config']['n_features']
    n_outputs = checkpoint['config']['n_outputs']

    model = PVD1DCNNModel(n_features=n_features, n_outputs=n_outputs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    with open(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_scaler_x.pkl'), 'rb') as f:
        scaler_x = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'v2_CNN_aug1x_scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_x, scaler_y, device


def predict_file(filepath, model, scaler_x, scaler_y, device):
    """Run inference on a single CSV file and return actual/predicted arrays."""
    df = pd.read_csv(filepath)
    active_mask = (df['EN4.Power'] > 0) | (df['SBRF5.SetPower'] > 0)
    df_active = df[active_mask].reset_index(drop=True)

    input_data = df_active[V2_INPUT_COLS].values.astype(np.float64)
    output_data = df_active[V2_OUTPUT_COLS].values.astype(np.float64).flatten()

    n = len(df_active)
    X_list = []
    actual_list = []
    time_indices = []

    for i in range(n - V2_INPUT_WINDOW - V2_PREDICT_AHEAD + 1):
        X_list.append(input_data[i:i + V2_INPUT_WINDOW])
        target_idx = i + V2_INPUT_WINDOW + V2_PREDICT_AHEAD - 1
        actual_list.append(output_data[target_idx])
        time_indices.append(target_idx)

    X = np.array(X_list)
    actuals = np.array(actual_list)

    # Scale input
    n_samples, seq_len, n_feat = X.shape
    X_scaled = scaler_x.transform(X.reshape(-1, n_feat)).reshape(n_samples, seq_len, n_feat)

    # Inference
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        preds_scaled = model(X_tensor).cpu().numpy()

    # Inverse transform predictions
    preds = scaler_y.inverse_transform(preds_scaled).flatten()

    return np.array(time_indices), actuals, preds, df_active


def main():
    model, scaler_x, scaler_y, device = load_model_and_scalers()

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle('V2 CNN_aug1x: Actual vs Predicted PWPDS.Data',
                 fontsize=16, fontweight='bold', y=0.99)

    colors_actual = '#1976D2'
    colors_pred = '#F44336'

    for idx, fname in enumerate(TARGET_FILES):
        filepath = os.path.join(DATA_DIR, fname)
        recipe = parse_recipe_name(filepath)

        time_idx, actuals, preds, df_active = predict_file(
            filepath, model, scaler_x, scaler_y, device)

        # Metrics for this file
        mae = np.mean(np.abs(actuals - preds))
        rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        mask = np.abs(actuals) > 1e-6
        mape = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100

        ax = axes[idx]

        # Plot full actual data (all active timesteps)
        all_actual = df_active['PWPDS.Data'].values
        ax.plot(range(len(all_actual)), all_actual,
                color=colors_actual, linewidth=2, label='Actual', alpha=0.85)

        # Plot predictions aligned to time index
        ax.plot(time_idx, preds,
                color=colors_pred, linewidth=2, linestyle='--', label='Predicted', alpha=0.85)

        # Error band
        ax.fill_between(time_idx, actuals, preds,
                        color=colors_pred, alpha=0.12, label='Error')

        # Recipe info
        recipe_label = fname.replace('.csv', '')
        type_label = recipe['recipe_type'].replace('_', ' ')
        dc_str = f"DC={recipe['dc_setting']}" if recipe['dc_setting'] > 0 else "DC=ign"
        rf_str = f"RF={recipe['rf_setting']}" if recipe['rf_setting'] > 0 else "RF=0"

        ax.set_title(f'{recipe_label}  ({type_label}, {dc_str}, {rf_str})  |  '
                     f'MAE={mae:.0f}  RMSE={rmse:.0f}  MAPE={mape:.4f}%',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('PWPDS.Data')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(RESULT_DIR, 'v2_CNN_aug1x_predictions.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
