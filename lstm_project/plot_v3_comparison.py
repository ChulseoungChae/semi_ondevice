"""V3 Result Comparison: Full Timeline CNN vs LSTM + V2 vs V3 comparison."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'results')

with open(os.path.join(RESULT_DIR, 'v3_all_results.json')) as f:
    v3_results = json.load(f)
with open(os.path.join(RESULT_DIR, 'v2_all_results.json')) as f:
    v2_results = json.load(f)

# Load V3 histories
v3_histories = {}
for r in v3_results:
    with open(os.path.join(RESULT_DIR, f"{r['config_name']}_history.json")) as f:
        v3_histories[r['config_name']] = json.load(f)

v3_results.sort(key=lambda x: (x['model_type'], x['aug_factor']))

CNN_COLOR = '#2196F3'
LSTM_COLOR = '#FF5722'
AUG_ALPHAS = {1: 1.0, 20: 0.65, 50: 0.35}
AUG_HATCHES = {1: '', 20: '//', 50: 'xx'}
AUG_LS = {1: '-', 20: '--', 50: ':'}

fig = plt.figure(figsize=(22, 16))
fig.suptitle('V3 Model Comparison: Full Timeline (no active filtering)\n'
             '8 features, predict_ahead=0, Idle+Active all included',
             fontsize=16, fontweight='bold', y=0.99)

# ── 1. RMSE ──
ax1 = fig.add_subplot(2, 3, 1)
names = [r['config_name'].replace('v3_', '') for r in v3_results]
rmses = [r['RMSE'] for r in v3_results]
colors = [CNN_COLOR if r['model_type'] == 'CNN' else LSTM_COLOR for r in v3_results]
alphas = [AUG_ALPHAS[r['aug_factor']] for r in v3_results]
hatches = [AUG_HATCHES[r['aug_factor']] for r in v3_results]

bars = ax1.bar(range(len(names)), rmses, color=colors, edgecolor='black', linewidth=0.8)
for bar, a, h in zip(bars, alphas, hatches):
    bar.set_alpha(a)
    bar.set_hatch(h)
for i, v in enumerate(rmses):
    ax1.text(i, v + 150, f'{v:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE (lower is better)', fontweight='bold')
ax1.set_ylim(0, max(rmses) * 1.15)
best_idx = np.argmin(rmses)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# ── 2. R² ──
ax2 = fig.add_subplot(2, 3, 2)
r2s = [r['R2'] for r in v3_results]
bars = ax2.bar(range(len(names)), r2s, color=colors, edgecolor='black', linewidth=0.8)
for bar, a, h in zip(bars, alphas, hatches):
    bar.set_alpha(a)
    bar.set_hatch(h)
for i, v in enumerate(r2s):
    ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('R²')
ax2.set_title('R² Score (higher is better)', fontweight='bold')
ax2.set_ylim(min(r2s) - 0.01, max(r2s) + 0.015)
best_idx = np.argmax(r2s)
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# ── 3. V2 vs V3 Comparison ──
ax3 = fig.add_subplot(2, 3, 3)
# Match configs: same model_type + aug_factor
v2_map = {(r['model_type'], r['aug_factor']): r for r in v2_results}
v3_map = {(r['model_type'], r['aug_factor']): r for r in v3_results}
common_keys = sorted(set(v2_map.keys()) & set(v3_map.keys()))

labels = [f"{mt}\n{af}x" for mt, af in common_keys]
v2_rmses = [v2_map[k]['RMSE'] for k in common_keys]
v3_rmses = [v3_map[k]['RMSE'] for k in common_keys]

x = np.arange(len(labels))
w = 0.35
bars_v2 = ax3.bar(x - w/2, v2_rmses, w, label='V2 (active only)', color='#90CAF9', edgecolor='black', linewidth=0.8)
bars_v3 = ax3.bar(x + w/2, v3_rmses, w, label='V3 (full timeline)', color='#EF9A9A', edgecolor='black', linewidth=0.8)

for i in range(len(labels)):
    ax3.text(i - w/2, v2_rmses[i] + 200, f'{v2_rmses[i]:,.0f}', ha='center', fontsize=7.5, fontweight='bold')
    ax3.text(i + w/2, v3_rmses[i] + 200, f'{v3_rmses[i]:,.0f}', ha='center', fontsize=7.5, fontweight='bold')

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel('RMSE')
ax3.set_title('V2 (active) vs V3 (full) RMSE', fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_ylim(0, max(max(v2_rmses), max(v3_rmses)) * 1.15)

# ── 4. CNN Training Curves ──
ax4 = fig.add_subplot(2, 3, 4)
for r in v3_results:
    if r['model_type'] != 'CNN':
        continue
    h = v3_histories[r['config_name']]
    ls = AUG_LS[r['aug_factor']]
    a = AUG_ALPHAS[r['aug_factor']]
    ax4.plot(h['val_loss'], label=f"CNN aug{r['aug_factor']}x (val)",
             linestyle=ls, color=CNN_COLOR, alpha=a, linewidth=2)
    ax4.plot(h['train_loss'], linestyle=ls, color=CNN_COLOR, alpha=a*0.4, linewidth=1)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss (MSE)')
ax4.set_title('CNN Training Curves (bold=val, thin=train)', fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_ylim(0, 0.3)
ax4.grid(True, alpha=0.3)

# ── 5. LSTM Training Curves ──
ax5 = fig.add_subplot(2, 3, 5)
for r in v3_results:
    if r['model_type'] != 'LSTM':
        continue
    h = v3_histories[r['config_name']]
    ls = AUG_LS[r['aug_factor']]
    a = AUG_ALPHAS[r['aug_factor']]
    ax5.plot(h['val_loss'], label=f"LSTM aug{r['aug_factor']}x (val)",
             linestyle=ls, color=LSTM_COLOR, alpha=a, linewidth=2)
    ax5.plot(h['train_loss'], linestyle=ls, color=LSTM_COLOR, alpha=a*0.4, linewidth=1)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Loss (MSE)')
ax5.set_title('LSTM Training Curves (bold=val, thin=train)', fontweight='bold')
ax5.legend(fontsize=9)
ax5.set_ylim(0, 0.3)
ax5.grid(True, alpha=0.3)

# ── 6. Summary Table ──
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

col_labels = ['Model', 'Aug', 'Params', 'RMSE', 'R²', 'MAPE%', 'Time']
table_data = []
for r in sorted(v3_results, key=lambda x: x['RMSE']):
    table_data.append([
        r['model_type'], f"{r['aug_factor']}x", f"{r['total_params']:,}",
        f"{r['RMSE']:,.0f}", f"{r['R2']:.4f}", f"{r['MAPE']:.4f}", f"{r['train_time_sec']:.1f}s",
    ])

table = ax6.table(cellText=table_data, colLabels=col_labels,
                  cellLoc='center', loc='center',
                  colColours=['#E3F2FD'] * len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)
for j in range(len(col_labels)):
    table[1, j].set_facecolor('#C8E6C9')
    table[1, j].set_text_props(fontweight='bold')
for i in range(1, len(table_data) + 1):
    c = '#BBDEFB' if table_data[i-1][0] == 'CNN' else '#FFCCBC'
    table[i, 0].set_facecolor(c)

ax6.set_title('V3 Results Ranked by RMSE', fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = os.path.join(RESULT_DIR, 'v3_model_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
