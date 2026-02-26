"""
Evaluation & Visualization for PVD LSTM Models
- Compare 12 configurations (4 models x 3 aug)
- Per-recipe evaluation
- Generate comparison plots
- Anomaly detection threshold analysis
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
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import (load_all_csvs, create_windows, MODEL_CONFIGS,
                         INPUT_WINDOW, PREDICT_AHEAD, parse_recipe_name)
from models import get_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')


def load_trained_model(config_name: str, device: torch.device):
    """Load a trained model with its scalers."""
    model_path = os.path.join(MODEL_DIR, f'{config_name}_best.pt')
    scaler_x_path = os.path.join(MODEL_DIR, f'{config_name}_scaler_x.pkl')
    scaler_y_path = os.path.join(MODEL_DIR, f'{config_name}_scaler_y.pkl')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    with open(scaler_x_path, 'rb') as f:
        scaler_x = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    model = get_model(config['model_key'], config['n_features'], config['n_outputs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, scaler_x, scaler_y, config


@torch.no_grad()
def predict_on_data(model, data_df, input_cols, output_cols, scaler_x, scaler_y, device):
    """Run prediction on a single process dataframe."""
    X, y = create_windows(data_df, input_cols, output_cols)
    if len(X) == 0:
        return None, None, None

    n_samples, seq_len, n_feat = X.shape
    n_out = y.shape[1] if y.ndim > 1 else 1

    X_scaled = scaler_x.transform(X.reshape(-1, n_feat)).reshape(n_samples, seq_len, n_feat)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    preds_scaled = model(X_tensor).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled)
    actual = y.reshape(-1, n_out)

    return actual, preds, X


def evaluate_per_recipe(config_name: str, device: torch.device):
    """Evaluate model on each recipe separately."""
    model, scaler_x, scaler_y, config = load_trained_model(config_name, device)
    input_cols = config['input_cols']
    output_cols = config['output_cols']

    all_data = load_all_csvs(DATA_DIR)
    results = []

    for proc in all_data:
        actual, preds, _ = predict_on_data(
            model, proc['data'], input_cols, output_cols, scaler_x, scaler_y, device
        )
        if actual is None:
            continue

        mae = mean_absolute_error(actual, preds)
        rmse = np.sqrt(mean_squared_error(actual, preds))
        r2 = r2_score(actual, preds)
        mask = np.abs(actual) > 1e-6
        mape = np.mean(np.abs((actual[mask] - preds[mask]) / actual[mask])) * 100

        # Error percentage (relative to mean)
        mean_val = np.mean(actual)
        err_pct = (rmse / mean_val) * 100 if mean_val != 0 else float('inf')

        results.append({
            'recipe': proc['filename'],
            'recipe_type': proc['recipe_type'],
            'dc_setting': proc['dc_setting'],
            'rf_setting': proc['rf_setting'],
            'n_samples': len(actual),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'MAPE': float(mape),
            'Error_Pct': float(err_pct),
            'actual_mean': float(mean_val),
            'actual_std': float(np.std(actual)),
        })

    return pd.DataFrame(results)


def plot_comparison_chart(results_path: str):
    """Generate comparison charts from all_results.json."""
    with open(results_path, 'r') as f:
        all_results = json.load(f)

    df = pd.DataFrame(all_results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LSTM Model Comparison: 4 Models x 3 Augmentation Levels', fontsize=14)

    metrics = ['MAE', 'RMSE', 'R2', 'MAPE']
    titles = ['MAE (Lower is Better)', 'RMSE (Lower is Better)',
              'R² Score (Higher is Better)', 'MAPE % (Lower is Better)']

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        for model_key in ['A', 'B', 'C', 'D']:
            subset = df[df['model_key'] == model_key].sort_values('aug_factor')
            ax.plot(subset['aug_factor'].values, subset[metric].values,
                    'o-', label=f'Model {model_key}', markersize=8)

        ax.set_xlabel('Augmentation Factor')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([1, 20, 50])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(RESULT_DIR, 'model_comparison.png')}")


def plot_predictions_by_recipe(config_name: str, device: torch.device, max_recipes: int = 15):
    """Plot predicted vs actual for each recipe type."""
    model, scaler_x, scaler_y, config = load_trained_model(config_name, device)
    input_cols = config['input_cols']
    output_cols = config['output_cols']

    all_data = load_all_csvs(DATA_DIR)

    # Group by recipe type for cleaner visualization
    recipe_groups = {}
    for proc in all_data:
        key = f"DC{proc['dc_setting']}_RF{proc['rf_setting']}"
        if key not in recipe_groups:
            recipe_groups[key] = []
        recipe_groups[key].append(proc)

    # Select representative recipes
    selected = list(recipe_groups.keys())[:max_recipes]
    n_plots = len(selected)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    fig.suptitle(f'Predictions vs Actual - {config_name}', fontsize=14)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, recipe_key in enumerate(selected):
        ax = axes[idx]
        procs = recipe_groups[recipe_key]

        # Use first process for plotting
        proc = procs[0]
        actual, preds, _ = predict_on_data(
            model, proc['data'], input_cols, output_cols, scaler_x, scaler_y, device
        )
        if actual is None:
            continue

        t = np.arange(len(actual))
        ax.plot(t, actual[:, 0], 'b-', label='Actual', linewidth=1.5)
        ax.plot(t, preds[:, 0], 'r--', label='Predicted', linewidth=1.5)
        ax.set_title(f'{recipe_key} ({proc["filename"]})', fontsize=10)
        ax.set_xlabel('Time step')
        ax.set_ylabel('PWPDS.Data')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add error band
        error = np.abs(actual[:, 0] - preds[:, 0])
        rmse = np.sqrt(np.mean(error ** 2))
        ax.fill_between(t, preds[:, 0] - rmse, preds[:, 0] + rmse,
                        alpha=0.15, color='red', label=f'±RMSE={rmse:.0f}')

    # Hide empty subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'{config_name}_predictions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(RESULT_DIR, f'{config_name}_predictions.png')}")


def plot_training_history(config_name: str):
    """Plot training/validation loss curves."""
    history_path = os.path.join(RESULT_DIR, f'{config_name}_history.json')
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History - {config_name}', fontsize=14)

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    ax2.plot(epochs, history['lr'], label='Learning Rate')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'{config_name}_training_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def generate_anomaly_threshold_analysis(config_name: str, device: torch.device):
    """Analyze prediction errors to suggest anomaly detection thresholds."""
    model, scaler_x, scaler_y, config = load_trained_model(config_name, device)
    input_cols = config['input_cols']
    output_cols = config['output_cols']

    all_data = load_all_csvs(DATA_DIR)

    all_errors = []
    recipe_errors = {}

    for proc in all_data:
        actual, preds, _ = predict_on_data(
            model, proc['data'], input_cols, output_cols, scaler_x, scaler_y, device
        )
        if actual is None:
            continue

        error_pct = np.abs((actual - preds) / actual) * 100
        all_errors.extend(error_pct.flatten().tolist())

        key = f"DC{proc['dc_setting']}_RF{proc['rf_setting']}"
        if key not in recipe_errors:
            recipe_errors[key] = []
        recipe_errors[key].extend(error_pct.flatten().tolist())

    all_errors = np.array(all_errors)

    print(f"\n{'='*60}")
    print(f"Anomaly Detection Threshold Analysis ({config_name})")
    print(f"{'='*60}")
    print(f"Overall prediction error (MAPE) on normal data:")
    print(f"  Mean: {np.mean(all_errors):.4f}%")
    print(f"  Std:  {np.std(all_errors):.4f}%")
    print(f"  P95:  {np.percentile(all_errors, 95):.4f}%")
    print(f"  P99:  {np.percentile(all_errors, 99):.4f}%")
    print(f"  Max:  {np.max(all_errors):.4f}%")

    # Suggest thresholds
    p95 = np.percentile(all_errors, 95)
    p99 = np.percentile(all_errors, 99)
    suggested = p99 * 1.5  # safety margin

    print(f"\nSuggested anomaly threshold: {suggested:.4f}%")
    print(f"  (1.5x of 99th percentile)")

    # Per-recipe analysis
    print(f"\nPer-recipe error distribution:")
    print(f"{'Recipe':<20s} {'Mean%':>8s} {'Std%':>8s} {'P95%':>8s} {'P99%':>8s} {'Max%':>8s}")
    print('-' * 60)
    for key in sorted(recipe_errors.keys()):
        errs = np.array(recipe_errors[key])
        print(f"{key:<20s} {np.mean(errs):>8.4f} {np.std(errs):>8.4f} "
              f"{np.percentile(errs, 95):>8.4f} {np.percentile(errs, 99):>8.4f} "
              f"{np.max(errs):>8.4f}")

    # Plot error distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Prediction Error Analysis - {config_name}', fontsize=14)

    ax1.hist(all_errors, bins=50, density=True, alpha=0.7, color='steelblue')
    ax1.axvline(p95, color='orange', linestyle='--', label=f'P95={p95:.4f}%')
    ax1.axvline(p99, color='red', linestyle='--', label=f'P99={p99:.4f}%')
    ax1.axvline(suggested, color='darkred', linestyle=':', label=f'Threshold={suggested:.4f}%')
    ax1.set_xlabel('Error %')
    ax1.set_ylabel('Density')
    ax1.set_title('Overall Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot per recipe
    recipe_keys = sorted(recipe_errors.keys())
    data_boxes = [recipe_errors[k] for k in recipe_keys]
    ax2.boxplot(data_boxes, labels=recipe_keys, vert=True)
    ax2.set_ylabel('Error %')
    ax2.set_title('Error Distribution by Recipe')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'{config_name}_error_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {os.path.join(RESULT_DIR, f'{config_name}_error_analysis.png')}")

    return {
        'mean_error_pct': float(np.mean(all_errors)),
        'std_error_pct': float(np.std(all_errors)),
        'p95': float(p95),
        'p99': float(p99),
        'suggested_threshold': float(suggested),
        'recipe_errors': {k: {'mean': float(np.mean(v)), 'p99': float(np.percentile(v, 99))}
                          for k, v in recipe_errors.items()},
    }


def run_full_evaluation():
    """Run evaluation for all trained models."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    results_path = os.path.join(RESULT_DIR, 'all_results.json')
    if not os.path.exists(results_path):
        print("No training results found. Run train.py first.")
        return

    with open(results_path, 'r') as f:
        all_results = json.load(f)

    # 1. Generate comparison chart
    print("Generating comparison charts...")
    plot_comparison_chart(results_path)

    # 2. Find best model
    best = min(all_results, key=lambda x: x['RMSE'])
    best_config = best['config_name']
    print(f"\nBest model: {best_config} (RMSE={best['RMSE']:.2f}, R2={best['R2']:.6f})")

    # 3. Training history for all configs
    print("\nGenerating training history plots...")
    for r in all_results:
        try:
            plot_training_history(r['config_name'])
        except FileNotFoundError:
            print(f"  Skipping {r['config_name']} (no history file)")

    # 4. Per-recipe evaluation for best model
    print(f"\nPer-recipe evaluation for best model ({best_config})...")
    recipe_df = evaluate_per_recipe(best_config, device)
    recipe_df.to_csv(os.path.join(RESULT_DIR, f'{best_config}_per_recipe.csv'), index=False)
    print(recipe_df.to_string(index=False))

    # 5. Prediction plots for best model
    print(f"\nGenerating prediction plots for {best_config}...")
    plot_predictions_by_recipe(best_config, device)

    # 6. Anomaly threshold analysis for best model
    threshold_info = generate_anomaly_threshold_analysis(best_config, device)
    with open(os.path.join(RESULT_DIR, f'{best_config}_threshold.json'), 'w') as f:
        json.dump(threshold_info, f, indent=2)

    # 7. Also evaluate all models and plot predictions for top 3
    print("\n\nGenerating predictions for top 3 models...")
    sorted_results = sorted(all_results, key=lambda x: x['RMSE'])
    for r in sorted_results[:3]:
        config_name = r['config_name']
        try:
            plot_predictions_by_recipe(config_name, device)
        except Exception as e:
            print(f"  Error for {config_name}: {e}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    run_full_evaluation()
