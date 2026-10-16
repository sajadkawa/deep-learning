import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

X = np.array([[0,0],[0,1],[1,0],[1,1]])
H = np.array([[0,1],[1,1],[1,1],[1,0]])

c1, c0       = '#1d4ed8', '#b91c1c'   # XOR=1 blue, XOR=0 red — matches all XOR diagrams
boundary_col = '#ca8a04'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')

# ── LEFT: Original Input Space ───────────────────────────────────────────────
ax = axes[0]

colors_left  = [c0, c1, c1, c0]
markers_left = ['X', 'o', 'o', 'X']
labels_left  = ['(0,0)\ny=0', '(0,1)\ny=1', '(1,0)\ny=1', '(1,1)\ny=0']
offsets_left = [(-42, 8), (-42, 8), (14, 8), (14, 8)]

for i in range(4):
    ax.scatter(X[i,0], X[i,1], color=colors_left[i], marker=markers_left[i],
               s=200, zorder=6, edgecolors='#374151', linewidths=0.8)
    ax.annotate(labels_left[i], (X[i,0], X[i,1]), textcoords='offset points',
                xytext=offsets_left[i], fontsize=8.5, color='#374151')

ax.plot([-0.3, 1.3], [1.3, -0.3], color='#9ca3af', linewidth=1.2,
        linestyle=':', alpha=0.6, zorder=2)
ax.text(1.1, 1.25, '?', color='#9ca3af', fontsize=13, alpha=0.7)

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(-0.55, 1.55)
ax.set_xlabel('x₁', color='#374151', fontsize=12, labelpad=6)
ax.set_ylabel('x₂', color='#374151', fontsize=12, labelpad=6)
ax.set_title('Input Space  (x₁, x₂)', color='#111111', fontsize=11, pad=10)
ax.tick_params(colors='#374151', labelsize=9)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.text(0.5, -0.48, '✗  Not linearly separable',
        ha='center', color='#b91c1c', fontsize=9, style='italic')

# ── RIGHT: Hidden Feature Space ──────────────────────────────────────────────
ax = axes[1]

h1v = np.linspace(-0.55, 1.55, 300)
h2_boundary = 1.5 - h1v
ax.fill_between(h1v, h2_boundary, 1.55,  alpha=0.08, color=c1, zorder=1)
ax.fill_between(h1v, -0.55, h2_boundary, alpha=0.08, color=c0, zorder=1)

h1_line = np.linspace(-0.3, 1.55, 200)
ax.plot(h1_line, 1.5 - h1_line, color=boundary_col, linewidth=2.0,
        linestyle='--', zorder=4, label='h₁ + h₂ = 1.5')

colors_right  = [c0, c1, c1, c0]
markers_right = ['X', 'o', 'o', 'X']
labels_right  = ['(0,0)→(0,1)', '(0,1)→(1,1)', '(1,0)→(1,1)', '(1,1)→(1,0)']
offsets_right = [(-72, 8), (14, 14), (14, -20), (14, 8)]

for i in range(4):
    hx, hy = H[i]
    ax.scatter(hx, hy, color=colors_right[i], marker=markers_right[i],
               s=200, zorder=6, edgecolors='#374151', linewidths=0.8)
    ax.annotate(labels_right[i], (hx, hy), textcoords='offset points',
                xytext=offsets_right[i], fontsize=8.5, color='#374151')

ax.scatter(1, 1, color='none', marker='o', s=420, zorder=5,
           edgecolors=boundary_col, linewidths=1.5)
ax.annotate('two points\ncollapse here',
            xy=(1.0, 1.0), xytext=(0.18, 1.38),
            fontsize=8, color=boundary_col, style='italic',
            arrowprops=dict(arrowstyle='->', color=boundary_col,
                            lw=1.3, connectionstyle='arc3,rad=-0.2'))

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(-0.55, 1.55)
ax.set_xlabel('h₁  (OR neuron)',   color='#374151', fontsize=12, labelpad=6)
ax.set_ylabel('h₂  (NAND neuron)', color='#374151', fontsize=12, labelpad=6)
ax.set_title('Hidden Feature Space  (h₁, h₂)', color='#111111', fontsize=11, pad=10)
ax.tick_params(colors='#374151', labelsize=9)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.legend(fontsize=8.5, facecolor='white', edgecolor='#d1d5db',
          labelcolor='#111111', loc='upper left')
ax.text(0.5, -0.48, '✓  Linearly separable — one line suffices',
        ha='center', color='#15803d', fontsize=9, style='italic')

# ── shared legend ─────────────────────────────────────────────────────────────
p1 = mpatches.Patch(color=c1, label='Class 1  (XOR = 1)')
p0 = mpatches.Patch(color=c0, label='Class 0  (XOR = 0)')
fig.legend(handles=[p1, p0], loc='lower center', ncol=2,
           facecolor='white', edgecolor='#d1d5db', labelcolor='#111111',
           fontsize=9.5, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Feature Space Warping — the Hidden Layer Transforms the Problem',
             color='#111111', fontsize=12, y=1.01)

plt.tight_layout(rect=[0, 0.07, 1, 1])
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'xor_feature_space.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {os.path.abspath(out)}')
