import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('#f8f8f8')

filter_colors = ['#e74c3c', '#3498db', '#2ecc71']  # 3 weight values shown as colours

def draw_grid_with_weights(ax, rows, cols, x0, y0, cell=0.75, patches=None, weight_colors=None):
    """Draw grid. patches = list of (r,c) top-left corners of 2x2 patches to highlight."""
    for r in range(rows):
        for c in range(cols):
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor='#e8e8e8', edgecolor='#aaa', linewidth=0.8, zorder=1)
            ax.add_patch(rect)

    if patches and weight_colors:
        for idx, (pr, pc) in enumerate(patches):
            color = weight_colors[idx % len(weight_colors)]
            for dr in range(2):
                for dc in range(2):
                    rect = plt.Rectangle((x0 + (pc + dc) * cell, y0 - (pr + dr) * cell), cell, cell,
                                          facecolor=color, edgecolor='#333', linewidth=1.5,
                                          alpha=0.5, zorder=2)
                    ax.add_patch(rect)

# ---- Left: FC — different weights at each position ----
ax = axes[0]
ax.set_xlim(0, 9)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('#f8f8f8')
ax.set_title('Fully Connected\nDifferent weights at every position', fontsize=11, fontweight='bold', color='#222')

# 4x4 grid
cell = 0.85
x0, y0 = 0.5, 7.0
for r in range(4):
    for c in range(4):
        # Each 2x2 patch gets a unique colour
        unique_colors = ['#ffb3b3', '#b3d9ff', '#b3ffb3', '#ffe0b3',
                         '#e0b3ff', '#b3fff0', '#ffd9b3', '#d9b3ff',
                         '#b3ffcc']
        idx = r * 4 + c
        color = unique_colors[idx % len(unique_colors)]
        rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                              facecolor=color, edgecolor='#888', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + c * cell + cell/2, y0 - r * cell + cell/2,
                f'w{idx+1}', ha='center', va='center', fontsize=7, color='#333', fontweight='bold')

ax.text(x0 + 2 * cell, y0 + 0.35, '4×4 input', ha='center', fontsize=9, color='#333', fontweight='bold')
ax.text(x0 + 2 * cell, y0 - 4 * cell - 0.3, '16 unique weights\n(one per pixel)', ha='center', fontsize=10,
        color='#cc3333', bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffe8e8', edgecolor='#cc3333'))

# ---- Right: CNN — same filter weights reused ----
ax2 = axes[1]
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_facecolor('#f8f8f8')
ax2.set_title('Convolutional Layer\nSame filter weights reused at every position', fontsize=11, fontweight='bold', color='#222')

# Show 4x4 grid with 2x2 filter applied at 3 positions
x0, y0 = 0.3, 7.0
cell = 0.85

# Base grid
for r in range(4):
    for c in range(4):
        rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                              facecolor='#e8e8e8', edgecolor='#aaa', linewidth=0.8, zorder=1)
        ax2.add_patch(rect)

# Highlight 3 patch positions with same colour scheme
patch_positions = [(0, 0), (0, 2), (2, 1)]
patch_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
for idx, (pr, pc) in enumerate(patch_positions):
    for dr in range(2):
        for dc in range(2):
            rect = plt.Rectangle((x0 + (pc + dc) * cell, y0 - (pr + dr) * cell), cell, cell,
                                  facecolor=patch_colors[idx], edgecolor='#333', linewidth=1.8,
                                  alpha=0.6, zorder=2)
            ax2.add_patch(rect)

ax2.text(x0 + 2 * cell, y0 + 0.35, '4×4 input', ha='center', fontsize=9, color='#333', fontweight='bold')

# Draw the filter box (same for all)
filter_x, filter_y = 5.8, 6.8
filter_cell = 0.75
filter_weights = [[1, 0], [1, 1]]
filter_colors_map = {1: '#ff6b6b', 0: '#ffffff'}
for r in range(2):
    for c in range(2):
        w = filter_weights[r][c]
        rect = plt.Rectangle((filter_x + c * filter_cell, filter_y - r * filter_cell),
                              filter_cell, filter_cell,
                              facecolor=filter_colors_map[w], edgecolor='#333', linewidth=1.5, zorder=3)
        ax2.add_patch(rect)
        ax2.text(filter_x + c * filter_cell + filter_cell/2,
                 filter_y - r * filter_cell + filter_cell/2,
                 str(w), ha='center', va='center', fontsize=11, fontweight='bold', color='#333')

ax2.text(filter_x + filter_cell, filter_y + 0.35, 'Filter\n(4 weights)', ha='center', fontsize=9,
         color='#333', fontweight='bold')

# Arrows from filter to each patch
for idx, (pr, pc) in enumerate(patch_positions):
    px = x0 + (pc + 1) * cell
    py = y0 - (pr + 1) * cell + cell/2
    ax2.annotate('', xy=(px, py), xytext=(filter_x + filter_cell, filter_y - filter_cell/2),
                arrowprops=dict(arrowstyle='->', color=patch_colors[idx], lw=1.5, linestyle='dashed'))

ax2.text(x0 + 2 * cell, y0 - 4 * cell - 0.3, '4 weights total\n(reused at every position)', ha='center', fontsize=10,
         color='#1a7a1a', bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8e8', edgecolor='#1a7a1a'))

plt.tight_layout(pad=1.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_parameter_sharing.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_parameter_sharing.png")
