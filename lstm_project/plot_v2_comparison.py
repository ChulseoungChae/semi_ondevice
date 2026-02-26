"""V2 Result Comparison Visualization: 1D CNN vs LSTM"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# Load results
with open(os.path.join(RESULT_DIR, 'v2_all_results.json')) as f:
    all_results = json.load(f)

# Load histories
histories = {}
for r in all_results:
    name = r['config_name']
    with open(os.path.join(RESULT_DIR, f'{name}_history.json')) as f:
        histories[name] = json.load(f)

# Sort by model type then aug factor for consistent ordering
all_results.sort(key=lambda x: (x['model_type'], x['aug_factor']))

# Color/style settings
CNN_COLOR = '#2196F3'
LSTM_COLOR = '#FF5722'
AUG_ALPHAS = {1: 1.0, 20: 0.65, 50: 0.35}
AUG_HATCHES = {1: '', 20: '//', 50: 'xx'}
AUG_LINESTYLES = {1: '-', 20: '--', 50: ':'}

fig = plt.figure(figsize=(20, 14))
fig.suptitle('V2 Model Comparison: 1D CNN vs LSTM\n(8 features, no PWPDS.Data input, predict_ahead=0)',
             fontsize=16, fontweight='bold', y=0.98)

# ── 1. RMSE Bar Chart ──
ax1 = fig.add_subplot(2, 3, 1)
names = [r['config_name'].replace('v2_', '') for r in all_results]
rmses = [r['RMSE'] for r in all_results]
colors = [CNN_COLOR if r['model_type'] == 'CNN' else LSTM_COLOR for r in all_results]
alphas = [AUG_ALPHAS[r['aug_factor']] for r in all_results]
hatches = [AUG_HATCHES[r['aug_factor']] for r in all_results]

bars = ax1.bar(range(len(names)), rmses, color=colors, edgecolor='black', linewidth=0.8)
for bar, a, h in zip(bars, alphas, hatches):
    bar.set_alpha(a)
    bar.set_hatch(h)
for i, v in enumerate(rmses):
    ax1.text(i, v + 200, f'{v:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE (lower is better)', fontweight='bold')
ax1.set_ylim(0, max(rmses) * 1.15)
# Highlight best
best_idx = np.argmin(rmses)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# ── 2. R² Bar Chart ──
ax2 = fig.add_subplot(2, 3, 2)
r2s = [r['R2'] for r in all_results]

bars = ax2.bar(range(len(names)), r2s, color=colors, edgecolor='black', linewidth=0.8)
for bar, a, h in zip(bars, alphas, hatches):
    bar.set_alpha(a)
    bar.set_hatch(h)
for i, v in enumerate(r2s):
    ax2.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('R²')
ax2.set_title('R² Score (higher is better)', fontweight='bold')
ax2.set_ylim(min(r2s) - 0.02, 1.0)
best_idx = np.argmax(r2s)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# ── 3. MAE & MAPE Grouped Bar ──
ax3 = fig.add_subplot(2, 3, 3)
maes = [r['MAE'] for r in all_results]
mapes = [r['MAPE'] for r in all_results]

x = np.arange(len(names))
w = 0.35
bars1 = ax3.bar(x - w/2, maes, w, color=colors, edgecolor='black', linewidth=0.8, label='MAE')
for bar, a, h in zip(bars1, alphas, hatches):
    bar.set_alpha(a)
    bar.set_hatch(h)

ax3_r = ax3.twinx()
bars2 = ax3_r.bar(x + w/2, mapes, w, color=[c for c in colors], edgecolor='black',
                  linewidth=0.8, alpha=0.3, label='MAPE%')

ax3.set_xticks(x)
ax3.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax3.set_ylabel('MAE')
ax3_r.set_ylabel('MAPE (%)')
ax3.set_title('MAE & MAPE', fontweight='bold')

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_r.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

# ── 4. Training Loss Curves (CNN) ──
ax4 = fig.add_subplot(2, 3, 4)
cnn_configs = [r for r in all_results if r['model_type'] == 'CNN']
for r in cnn_configs:
    h = histories[r['config_name']]
    label = f"CNN aug{r['aug_factor']}x"
    ls = AUG_LINESTYLES[r['aug_factor']]
    ax4.plot(h['val_loss'], label=f'{label} (val)', linestyle=ls, color=CNN_COLOR,
             alpha=AUG_ALPHAS[r['aug_factor']], linewidth=2)
    ax4.plot(h['train_loss'], linestyle=ls, color=CNN_COLOR,
             alpha=AUG_ALPHAS[r['aug_factor']] * 0.4, linewidth=1)

ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss (MSE)')
ax4.set_title('CNN Training Curves (bold=val, thin=train)', fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_ylim(0, 0.6)
ax4.grid(True, alpha=0.3)

# ── 5. Training Loss Curves (LSTM) ──
ax5 = fig.add_subplot(2, 3, 5)
lstm_configs = [r for r in all_results if r['model_type'] == 'LSTM']
for r in lstm_configs:
    h = histories[r['config_name']]
    label = f"LSTM aug{r['aug_factor']}x"
    ls = AUG_LINESTYLES[r['aug_factor']]
    ax5.plot(h['val_loss'], label=f'{label} (val)', linestyle=ls, color=LSTM_COLOR,
             alpha=AUG_ALPHAS[r['aug_factor']], linewidth=2)
    ax5.plot(h['train_loss'], linestyle=ls, color=LSTM_COLOR,
             alpha=AUG_ALPHAS[r['aug_factor']] * 0.4, linewidth=1)

ax5.set_xlabel('Epoch')
ax5.set_ylabel('Loss (MSE)')
ax5.set_title('LSTM Training Curves (bold=val, thin=train)', fontweight='bold')
ax5.legend(fontsize=9)
ax5.set_ylim(0, 0.6)
ax5.grid(True, alpha=0.3)

# ── 6. Summary Table ──
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

col_labels = ['Model', 'Aug', 'Params', 'RMSE', 'R²', 'MAPE%', 'Time']
table_data = []
for r in sorted(all_results, key=lambda x: x['RMSE']):
    table_data.append([
        r['model_type'],
        f"{r['aug_factor']}x",
        f"{r['total_params']:,}",
        f"{r['RMSE']:.0f}",
        f"{r['R2']:.4f}",
        f"{r['MAPE']:.4f}",
        f"{r['train_time_sec']:.1f}s",
    ])

table = ax6.table(cellText=table_data, colLabels=col_labels,
                  cellLoc='center', loc='center',
                  colColours=['#E3F2FD'] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# Highlight best row (first row = lowest RMSE)
for j in range(len(col_labels)):
    table[1, j].set_facecolor('#C8E6C9')
    table[1, j].set_text_props(fontweight='bold')
# Color model type cells
for i in range(1, len(table_data) + 1):
    model_type = table_data[i-1][0]
    c = '#BBDEFB' if model_type == 'CNN' else '#FFCCBC'
    table[i, 0].set_facecolor(c)

ax6.set_title('Results Ranked by RMSE', fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = os.path.join(RESULT_DIR, 'v2_model_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
