"""
PWPDS.Data Upper/Lower Bounds Model
- Input: last 10 seconds of EN4.Power & SBRF5.SetPower
- Output: lower_bound, upper_bound for PWPDS.Data
- Uses both statistical analysis and a learned model
- All normal data -> compute optimal bounds per DC/RF recipe
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_all_csvs, INPUT_WINDOW
from models import PVDBoundsModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/home/goo4168/baco/PDS_Data_Log'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def compute_statistical_bounds(sigma_multiplier: float = 3.0):
    """Compute PWPDS.Data bounds per recipe using statistical method.

    Uses mean ± sigma_multiplier * std from normal data.
    """
    all_data = load_all_csvs(DATA_DIR)

    bounds_table = {}
    recipe_stats = []

    for proc in all_data:
        key = f"DC{proc['dc_setting']}_RF{proc['rf_setting']}"
        if key not in bounds_table:
            bounds_table[key] = {
                'dc_setting': proc['dc_setting'],
                'rf_setting': proc['rf_setting'],
                'all_values': [],
            }
        pwpds = proc['data']['PWPDS.Data'].values
        bounds_table[key]['all_values'].extend(pwpds.tolist())

    results = []
    for key in sorted(bounds_table.keys()):
        info = bounds_table[key]
        values = np.array(info['all_values'])
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Statistical bounds
        lower = mean_val - sigma_multiplier * std_val
        upper = mean_val + sigma_multiplier * std_val

        # Also use min/max with margin
        range_margin = (max_val - min_val) * 0.1
        lower_minmax = min_val - range_margin
        upper_minmax = max_val + range_margin

        # Use the wider of the two methods
        final_lower = min(lower, lower_minmax)
        final_upper = max(upper, upper_minmax)

        result = {
            'recipe': key,
            'dc_setting': info['dc_setting'],
            'rf_setting': info['rf_setting'],
            'n_samples': len(values),
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val),
            'lower_bound': float(final_lower),
            'upper_bound': float(final_upper),
            'range': float(final_upper - final_lower),
        }
        results.append(result)

    return pd.DataFrame(results)


class BoundsDataset(Dataset):
    """Dataset for bounds model training."""

    def __init__(self, X, y_lower, y_upper):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(np.column_stack([y_lower, y_upper]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_bounds_training_data():
    """Create training data for bounds model.

    For each 10-second window of EN4.Power & SBRF5.SetPower,
    the target is the recipe's lower and upper bound.
    """
    all_data = load_all_csvs(DATA_DIR)

    # First compute per-recipe bounds
    bounds_df = compute_statistical_bounds(sigma_multiplier=3.0)
    bounds_dict = {}
    for _, row in bounds_df.iterrows():
        bounds_dict[row['recipe']] = (row['lower_bound'], row['upper_bound'])

    X_list, lower_list, upper_list = [], [], []

    for proc in all_data:
        key = f"DC{proc['dc_setting']}_RF{proc['rf_setting']}"
        lower, upper = bounds_dict[key]

        # Extract EN4.Power and SBRF5.SetPower windows
        power_data = proc['data'][['EN4.Power', 'SBRF5.SetPower']].values.astype(np.float64)

        for i in range(len(power_data) - INPUT_WINDOW + 1):
            window = power_data[i:i + INPUT_WINDOW]
            X_list.append(window)
            lower_list.append(lower)
            upper_list.append(upper)

    X = np.array(X_list)
    y_lower = np.array(lower_list)
    y_upper = np.array(upper_list)

    return X, y_lower, y_upper, bounds_df


def train_bounds_model(epochs: int = 200, lr: float = 1e-3):
    """Train the bounds prediction model."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Preparing bounds training data...")
    X, y_lower, y_upper, bounds_df = prepare_bounds_training_data()
    print(f"Training data: {X.shape[0]} samples, input shape: {X.shape}")

    # Scale inputs
    n_samples, seq_len, n_feat = X.shape
    scaler_x = StandardScaler()
    X_flat = X.reshape(-1, n_feat)
    scaler_x.fit(X_flat)
    X_scaled = scaler_x.transform(X_flat).reshape(n_samples, seq_len, n_feat)

    # Scale outputs
    scaler_y = StandardScaler()
    y = np.column_stack([y_lower, y_upper])
    scaler_y.fit(y)
    y_scaled = scaler_y.transform(y)

    # Train/val split (80/20)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n_samples)
    split = int(0.8 * n_samples)
    train_idx, val_idx = idx[:split], idx[split:]

    train_dataset = BoundsDataset(X_scaled[train_idx], y_scaled[train_idx, 0], y_scaled[train_idx, 1])
    val_dataset = BoundsDataset(X_scaled[val_idx], y_scaled[val_idx, 0], y_scaled[val_idx, 1])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = PVDBoundsModel(input_size=2, seq_len=seq_len).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    # Custom loss: MSE + penalty if predicted bounds are narrower than actual data range
    def bounds_loss(pred, target):
        mse = nn.functional.mse_loss(pred, target)
        # Penalty if upper < lower (should be handled by architecture, but extra safety)
        margin_penalty = torch.relu(pred[:, 0] - pred[:, 1]).mean()
        return mse + 0.1 * margin_penalty

    best_val_loss = float('inf')
    patience_counter = 0

    print("Training bounds model...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = bounds_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = bounds_loss(pred, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'bounds_model_best.pt'))
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if patience_counter >= 30:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best and evaluate
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'bounds_model_best.pt'),
                                     map_location=device, weights_only=True))
    model.eval()

    # Save scalers
    with open(os.path.join(MODEL_DIR, 'bounds_scaler_x.pkl'), 'wb') as f:
        pickle.dump(scaler_x, f)
    with open(os.path.join(MODEL_DIR, 'bounds_scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)

    # Evaluate on all data
    print("\nEvaluating bounds model on all data...")
    with torch.no_grad():
        X_all_scaled = torch.FloatTensor(X_scaled).to(device)
        preds_scaled = model(X_all_scaled).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_scaled)

    # Check coverage
    all_data = load_all_csvs(DATA_DIR)
    idx_offset = 0
    coverage_results = []

    for proc in all_data:
        n_windows = len(proc['data']) - INPUT_WINDOW + 1
        pwpds_values = proc['data']['PWPDS.Data'].values[INPUT_WINDOW - 1:]

        pred_lower = preds[idx_offset:idx_offset + n_windows, 0]
        pred_upper = preds[idx_offset:idx_offset + n_windows, 1]

        within = ((pwpds_values >= pred_lower) & (pwpds_values <= pred_upper)).mean() * 100

        key = f"DC{proc['dc_setting']}_RF{proc['rf_setting']}"
        coverage_results.append({
            'recipe': key,
            'filename': proc['filename'],
            'coverage_pct': float(within),
            'pred_lower_mean': float(np.mean(pred_lower)),
            'pred_upper_mean': float(np.mean(pred_upper)),
            'actual_min': float(np.min(pwpds_values)),
            'actual_max': float(np.max(pwpds_values)),
        })

        idx_offset += n_windows

    coverage_df = pd.DataFrame(coverage_results)
    print(f"\nBounds Model Coverage (% of actual values within predicted bounds):")
    print(f"  Overall: {coverage_df['coverage_pct'].mean():.2f}%")
    print(f"  Min coverage: {coverage_df['coverage_pct'].min():.2f}%")
    print(f"\nPer-recipe coverage:")
    recipe_coverage = coverage_df.groupby('recipe')['coverage_pct'].mean().reset_index()
    for _, row in recipe_coverage.iterrows():
        print(f"  {row['recipe']:<20s}: {row['coverage_pct']:.2f}%")

    return bounds_df, coverage_df


def generate_bounds_report():
    """Generate comprehensive bounds report."""
    print("="*60)
    print("PWPDS.Data Bounds Analysis")
    print("="*60)

    # 1. Statistical bounds
    print("\n1. Statistical Bounds (3-sigma + min/max margin)")
    stat_bounds = compute_statistical_bounds(sigma_multiplier=3.0)
    print(stat_bounds[['recipe', 'dc_setting', 'rf_setting', 'mean', 'std',
                       'lower_bound', 'upper_bound', 'range']].to_string(index=False))

    stat_bounds.to_csv(os.path.join(RESULT_DIR, 'statistical_bounds.csv'), index=False)

    # 2. Train and evaluate bounds model
    print("\n2. Training Bounds Prediction Model...")
    bounds_df, coverage_df = train_bounds_model()
    coverage_df.to_csv(os.path.join(RESULT_DIR, 'bounds_coverage.csv'), index=False)

    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PWPDS.Data Bounds by DC/RF Recipe', fontsize=14)

    # Bar chart of bounds
    ax = axes[0]
    recipes = stat_bounds['recipe'].values
    x_pos = np.arange(len(recipes))
    width = 0.35

    means = stat_bounds['mean'].values
    lowers = stat_bounds['lower_bound'].values
    uppers = stat_bounds['upper_bound'].values

    ax.bar(x_pos, means - lowers, bottom=lowers, width=0.6, alpha=0.4,
           color='steelblue', label='Bounds range')
    ax.bar(x_pos, uppers - means, bottom=means, width=0.6, alpha=0.4,
           color='steelblue')
    ax.scatter(x_pos, means, color='red', zorder=5, label='Mean', s=30)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(recipes, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('PWPDS.Data')
    ax.set_title('Bounds Range per Recipe')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Coverage
    ax = axes[1]
    recipe_coverage = coverage_df.groupby('recipe')['coverage_pct'].mean().reset_index()
    recipe_coverage = recipe_coverage.sort_values('recipe')
    ax.barh(recipe_coverage['recipe'], recipe_coverage['coverage_pct'], color='steelblue')
    ax.axvline(100, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coverage %')
    ax.set_title('Model Bounds Coverage')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'bounds_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {os.path.join(RESULT_DIR, 'bounds_analysis.png')}")

    # 4. Summary table for DC/RF -> bounds mapping
    print("\n" + "="*60)
    print("FINAL BOUNDS TABLE (for anomaly detection)")
    print("="*60)
    print(f"{'DC Setting':>10s} {'RF Setting':>10s} {'Lower Bound':>15s} {'Upper Bound':>15s} {'Range':>12s}")
    print('-' * 62)
    for _, row in stat_bounds.iterrows():
        print(f"{row['dc_setting']:>10.0f} {row['rf_setting']:>10.0f} "
              f"{row['lower_bound']:>15.0f} {row['upper_bound']:>15.0f} {row['range']:>12.0f}")

    # Save final bounds as JSON for easy loading
    bounds_lookup = {}
    for _, row in stat_bounds.iterrows():
        key = f"DC{int(row['dc_setting'])}_RF{int(row['rf_setting'])}"
        bounds_lookup[key] = {
            'lower': float(row['lower_bound']),
            'upper': float(row['upper_bound']),
            'mean': float(row['mean']),
            'std': float(row['std']),
        }

    with open(os.path.join(RESULT_DIR, 'bounds_lookup.json'), 'w') as f:
        json.dump(bounds_lookup, f, indent=2)
    print(f"\nSaved bounds lookup: {os.path.join(RESULT_DIR, 'bounds_lookup.json')}")

    return stat_bounds


if __name__ == '__main__':
    generate_bounds_report()
