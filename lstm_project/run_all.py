"""
Main entry point: Run all training, evaluation, and bounds analysis.
Usage: python3 run_all.py
"""

import os
import sys
import time
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    total_start = time.time()

    print("="*70)
    print("PVD4 PWPDS.Data Prediction - Full Pipeline")
    print("="*70)
    print(f"Working directory: {BASE_DIR}")

    # Step 1: Train all 12 LSTM configurations
    print("\n" + "#"*70)
    print("# STEP 1: Training 12 LSTM Model Configurations")
    print("# (4 models x 3 augmentation levels)")
    print("#"*70)
    from train import run_all_training
    training_results = run_all_training()

    # Step 2: Full evaluation
    print("\n" + "#"*70)
    print("# STEP 2: Model Evaluation & Comparison")
    print("#"*70)
    from evaluate import run_full_evaluation
    run_full_evaluation()

    # Step 3: Bounds analysis
    print("\n" + "#"*70)
    print("# STEP 3: PWPDS.Data Bounds Analysis & Model")
    print("#"*70)
    from bounds_model import generate_bounds_report
    bounds_df = generate_bounds_report()

    # Final summary
    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\nAll results saved to: {RESULT_DIR}")
    print(f"All models saved to: {os.path.join(BASE_DIR, 'models')}")

    print("\nFiles generated:")
    for root, dirs, files in os.walk(RESULT_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size:,} bytes)")

    # Load and display final ranking
    results_path = os.path.join(RESULT_DIR, 'all_results.json')
    with open(results_path, 'r') as f:
        all_results = json.load(f)

    print(f"\n{'='*80}")
    print("LSTM MODEL RANKING (by RMSE)")
    print(f"{'='*80}")
    print(f"{'Rank':>4s} {'Config':<25s} {'Aug':>4s} {'MAE':>12s} {'RMSE':>12s} "
          f"{'R2':>10s} {'MAPE%':>10s}")
    print('-' * 80)
    for rank, r in enumerate(sorted(all_results, key=lambda x: x['RMSE']), 1):
        print(f"{rank:>4d} {r['config_name']:<25s} {r['aug_factor']:>3d}x "
              f"{r['MAE']:>12.2f} {r['RMSE']:>12.2f} {r['R2']:>10.6f} "
              f"{r['MAPE']:>9.4f}%")

    best = min(all_results, key=lambda x: x['RMSE'])
    print(f"\n>>> Best Model: {best['config_name']}")
    print(f"    RMSE: {best['RMSE']:.2f}")
    print(f"    R2:   {best['R2']:.6f}")
    print(f"    MAPE: {best['MAPE']:.4f}%")

    print("\n>>> Bounds lookup saved to: results/bounds_lookup.json")
    print(">>> Use the best model + bounds for real-time anomaly detection")


if __name__ == '__main__':
    main()
