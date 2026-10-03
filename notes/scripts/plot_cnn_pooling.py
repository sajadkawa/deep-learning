import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor('#f8f8f8')

fm = np.array([
    [6, 4, 3, 7],
    [4, 3, 5, 4],
    [3, 5, 5, 3],
    [5, 5, 3, 6]
])

region_colors = [['#ffb3b3', '#b3d9ff'], ['#b3ffb3', '#ffe0b3']]
region_edge   = [['#cc3333', '#2255aa'], ['#1a7a1a', '#cc6600']]

cell = 1.0

def draw_pooling(ax, fm, pool_type, title):
    ax.set_facecolor('#f8f8f8')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', color='#222', pad=10)

    x0, y0 = 0.4, 5.8   # input grid top-left corner

    # --- Input 4×4 grid ---
    for r in range(4):
        for c in range(4):
            ri, ci = r // 2, c // 2
            color = region_colors[ri][ci]
            edge  = region_edge[ri][ci]
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor=color, edgecolor=edge, linewidth=2.2, zorder=2)
            ax.add_patch(rect)
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2,
                    str(fm[r, c]), ha='center', va='center',
                    fontsize=13, fontweight='bold', color='#222', zorder=3)

    # Thick region borders
    for ri in range(2):
        for ci in range(2):
            rect_b = plt.Rectangle((x0 + ci * 2 * cell, y0 - (ri * 2 + 1) * cell),
                                    2 * cell, 2 * cell,
                                    facecolor='none', edgecolor=region_edge[ri][ci],
                                    linewidth=3.5, zorder=4)
            ax.add_patch(rect_b)

    ax.text(x0 + 2 * cell, y0 + 0.5, 'Input feature map (4×4)',
            ha='center', fontsize=10, fontweight='bold', color='#333')
    ax.text(x0 + 2 * cell, y0 - 4 * cell - 0.4, '2×2 pool regions, stride = 2',
            ha='center', fontsize=9, color='#555')

    # --- Arrow ---
    arrow_x0 = x0 + 4 * cell + 0.3
    arrow_x1 = arrow_x0 + 0.9
    arrow_y  = y0 - 1.5 * cell
    ax.annotate('', xy=(arrow_x1, arrow_y), xytext=(arrow_x0, arrow_y),
                arrowprops=dict(arrowstyle='->', color='#444', lw=2.5))

    # --- Output 2×2 grid ---
    ox0 = arrow_x1 + 0.3
    oy0 = y0 - cell * 0.5   # vertically centred relative to input

    for ri in range(2):
        for ci in range(2):
            region = fm[ri * 2:(ri + 1) * 2, ci * 2:(ci + 1) * 2]
            val = region.max() if pool_type == 'max' else region.mean()
            color = region_colors[ri][ci]
            edge  = region_edge[ri][ci]
            rect = plt.Rectangle((ox0 + ci * cell * 1.3, oy0 - ri * cell * 1.3),
                                  cell * 1.3, cell * 1.3,
                                  facecolor=color, edgecolor=edge, linewidth=2.5, zorder=2)
            ax.add_patch(rect)
            label = str(int(val)) if pool_type == 'max' else f'{val:.1f}'
            ax.text(ox0 + ci * cell * 1.3 + cell * 0.65,
                    oy0 - ri * cell * 1.3 + cell * 0.65,
                    label, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#222', zorder=3)

    ax.text(ox0 + cell * 1.3, oy0 + 0.5, 'Output (2×2)',
            ha='center', fontsize=10, fontweight='bold', color='#333')

    # --- Computation detail box — below output grid, well clear ---
    if pool_type == 'max':
        lines = [
            'max(6, 4, 4, 3) = 6',
            'max(3, 7, 5, 4) = 7',
            'max(3, 5, 5, 5) = 5',
            'max(5, 3, 3, 6) = 6',
        ]
    else:
        lines = [
            'avg(6, 4, 4, 3) = 4.25',
            'avg(3, 7, 5, 4) = 4.75',
            'avg(3, 5, 5, 5) = 4.50',
            'avg(5, 3, 3, 6) = 4.25',
        ]

    detail_x = ox0 + cell * 1.3
    detail_y = oy0 - 2 * cell * 1.3 - 0.6
    ax.text(detail_x, detail_y, '\n'.join(lines),
            ha='center', va='top', fontsize=9, color='#333', linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f8f8', edgecolor='#aaa', linewidth=1.2))

    ax.set_xlim(-0.2, ox0 + 2 * cell * 1.3 + 0.5)
    ax.set_ylim(detail_y - 2.5, y0 + 0.9)

draw_pooling(axes[0], fm, 'max',
             '2×2 Max Pooling  (stride = 2)\nKeeps the strongest activation in each region')
draw_pooling(axes[1], fm, 'avg',
             '2×2 Average Pooling  (stride = 2)\nAverages all activations in each region')

plt.tight_layout(pad=2.0)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_pooling.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_pooling.png")
