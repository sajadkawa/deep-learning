import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 0])
H = np.array([[0,1],[1,1],[1,1],[1,0]])

c1, c0 = '#e05252', '#4a9eda'   # red = class 1, blue = class 0
boundary_col = '#f0a500'

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
fig.patch.set_facecolor('#0f1117')
for ax in axes:
    ax.set_facecolor('#161b27')
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3f50')

# ── helpers ──────────────────────────────────────────────────────────────────
def plot_point(ax, x, y_coord, label, color, marker, note, offset=(14, 8)):
    ax.scatter(x, y_coord, color=color, marker=marker, s=220, zorder=6,
               edgecolors='white', linewidths=1.0)
    ax.annotate(note, (x, y_coord), textcoords='offset points',
                xytext=offset, fontsize=8.5, color='#d0d4e0')

# ── LEFT: Original Input Space ───────────────────────────────────────────────
ax = axes[0]

labels_left = ['(0,0)\ny=0', '(0,1)\ny=1', '(1,0)\ny=1', '(1,1)\ny=0']
colors_left = [c0, c1, c1, c0]
markers_left = ['s', 'o', 'o', 's']
offsets_left = [(-38, 8), (-38, 8), (14, 8), (14, 8)]

for i in range(4):
    plot_point(ax, X[i,0], X[i,1], y[i], colors_left[i],
               markers_left[i], labels_left[i], offsets_left[i])

# Show that no line works — draw a crossed-out attempt
ax.plot([-0.3, 1.3], [1.3, -0.3], color='#888', linewidth=1.2,
        linestyle=':', alpha=0.5, zorder=2)
ax.text(1.1, 1.25, '?', color='#888', fontsize=13, alpha=0.6)

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(-0.55, 1.55)
ax.set_xlabel('x₁', color='#c0c4d0', fontsize=12, labelpad=6)
ax.set_ylabel('x₂', color='#c0c4d0', fontsize=12, labelpad=6)
ax.set_title('Input Space  (x₁, x₂)', color='white', fontsize=11, pad=10)
ax.tick_params(colors='#888', labelsize=9)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['0', '1'], color='#aaa')
ax.set_yticklabels(['0', '1'], color='#aaa')

ax.text(0.5, -0.48, '✗  Not linearly separable',
        ha='center', color='#e05252', fontsize=9, style='italic')

# ── RIGHT: Hidden Feature Space ──────────────────────────────────────────────
ax = axes[1]

# Region shading
h1v = np.linspace(-0.55, 1.55, 300)
h2_boundary = 1.5 - h1v
ax.fill_between(h1v, h2_boundary, 1.55, alpha=0.13, color=c1, zorder=1)
ax.fill_between(h1v, -0.55, h2_boundary, alpha=0.13, color=c0, zorder=1)

# Decision boundary
h1_line = np.linspace(-0.3, 1.55, 200)
ax.plot(h1_line, 1.5 - h1_line, color=boundary_col, linewidth=2.0,
        linestyle='--', zorder=4, label='h₁ + h₂ = 1.5')

# Points — (0,1) and (1,0) both map to (1,1): draw as concentric rings
labels_right = ['(0,0)→(0,1)', '(0,1)→(1,1)', '(1,0)→(1,1)', '(1,1)→(1,0)']
colors_right = [c0, c1, c1, c0]
markers_right = ['s', 'o', 'o', 's']
offsets_right = [(-72, 8), (14, 14), (14, -20), (14, 8)]

for i in range(4):
    hx, hy = H[i]
    ax.scatter(hx, hy, color=colors_right[i], marker=markers_right[i],
               s=220, zorder=6, edgecolors='white', linewidths=1.0)
    ax.annotate(labels_right[i], (hx, hy), textcoords='offset points',
                xytext=offsets_right[i], fontsize=8.5, color='#d0d4e0')

# Extra outer ring on (1,1) to show two points collapsed there
ax.scatter(1, 1, color='none', marker='o', s=420, zorder=5,
           edgecolors=boundary_col, linewidths=1.5, linestyle='--')

# Collapse annotation
ax.annotate('two points\ncollapse here',
            xy=(1.0, 1.0), xytext=(0.18, 1.38),
            fontsize=8, color=boundary_col, style='italic',
            arrowprops=dict(arrowstyle='->', color=boundary_col,
                            lw=1.3, connectionstyle='arc3,rad=-0.2'))

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(-0.55, 1.55)
ax.set_xlabel('h₁  (OR neuron)', color='#c0c4d0', fontsize=12, labelpad=6)
ax.set_ylabel('h₂  (NAND neuron)', color='#c0c4d0', fontsize=12, labelpad=6)
ax.set_title('Hidden Feature Space  (h₁, h₂)', color='white', fontsize=11, pad=10)
ax.tick_params(colors='#888', labelsize=9)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['0', '1'], color='#aaa')
ax.set_yticklabels(['0', '1'], color='#aaa')
ax.legend(fontsize=8.5, facecolor='#161b27', edgecolor='#3a3f50',
          labelcolor='#d0d4e0', loc='upper left')

ax.text(0.5, -0.48, '✓  Linearly separable — one line suffices',
        ha='center', color='#4caf7d', fontsize=9, style='italic')

# ── shared legend ─────────────────────────────────────────────────────────────
p1 = mpatches.Patch(color=c1, label='Class 1  (XOR = 1)')
p0 = mpatches.Patch(color=c0, label='Class 0  (XOR = 0)')
fig.legend(handles=[p1, p0], loc='lower center', ncol=2,
           facecolor='#0f1117', edgecolor='#3a3f50', labelcolor='#d0d4e0',
           fontsize=9.5, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Feature Space Warping — the Hidden Layer Transforms the Problem',
             color='white', fontsize=12, y=1.01)

plt.tight_layout(rect=[0, 0.07, 1, 1])

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'xor_feature_space.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved: {os.path.abspath(out)}')
