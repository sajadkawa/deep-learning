import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(13, 8))
fig.patch.set_facecolor('white')

fm = np.array([
    [6, 4, 3, 7],
    [4, 3, 5, 4],
    [3, 5, 5, 3],
    [5, 5, 3, 6]
])

region_colors = [['#fee2e2', '#dbeafe'], ['#dcfce7', '#fef3c7']]
region_edge   = [['#dc2626', '#2563eb'], ['#15803d', '#b45309']]

cell_in  = 0.9   # input cell size
cell_out = 1.1   # output cell size (larger for readability)

def draw_pooling(ax, fm, pool_type, title):
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', color='#111111', pad=12)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10)

    # ── Input grid (top, centred) ──────────────────────────────
    x0 = 0.3
    y0 = 9.0   # top of input grid

    ax.text(x0 + 2 * cell_in, y0 + 0.3, 'Input feature map (4×4)',
            ha='center', fontsize=9.5, fontweight='bold', color='#111111')

    for r in range(4):
        for c in range(4):
            ri, ci = r // 2, c // 2
            rect = plt.Rectangle((x0 + c * cell_in, y0 - (r+1) * cell_in),
                                  cell_in, cell_in,
                                  facecolor=region_colors[ri][ci],
                                  edgecolor=region_edge[ri][ci],
                                  linewidth=1.5, zorder=2)
            ax.add_patch(rect)
            ax.text(x0 + c * cell_in + cell_in/2,
                    y0 - (r+1) * cell_in + cell_in/2,
                    str(fm[r, c]), ha='center', va='center',
                    fontsize=12, fontweight='bold', color='#111111', zorder=3)

    # thick 2×2 region borders
    for ri in range(2):
        for ci in range(2):
            rect_b = plt.Rectangle((x0 + ci * 2 * cell_in, y0 - (ri * 2 + 2) * cell_in),
                                    2 * cell_in, 2 * cell_in,
                                    facecolor='none', edgecolor=region_edge[ri][ci],
                                    linewidth=3.0, zorder=4)
            ax.add_patch(rect_b)

    ax.text(x0 + 2 * cell_in, y0 - 4 * cell_in - 0.2,
            '4 non-overlapping 2×2 regions  (stride = 2)',
            ha='center', fontsize=8.5, color='#6b7280')

    # ── Arrow ─────────────────────────────────────────────────
    arrow_x = x0 + 2 * cell_in
    arrow_y_top = y0 - 4 * cell_in - 0.55
    arrow_y_bot = arrow_y_top - 0.7
    ax.annotate('', xy=(arrow_x, arrow_y_bot), xytext=(arrow_x, arrow_y_top),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=2.0))
    op = 'max' if pool_type == 'max' else 'avg'
    ax.text(arrow_x + 0.2, (arrow_y_top + arrow_y_bot) / 2,
            f'{op} pool', ha='left', va='center', fontsize=8.5, color='#374151')

    # ── Output grid (below arrow) ──────────────────────────────
    oy0 = arrow_y_bot - 0.15

    ax.text(x0 + cell_out, oy0 + 0.25, 'Output (2×2)',
            ha='center', fontsize=9.5, fontweight='bold', color='#111111')

    comp_lines = []
    for ri in range(2):
        for ci in range(2):
            region = fm[ri*2:(ri+1)*2, ci*2:(ci+1)*2].flatten()
            val = region.max() if pool_type == 'max' else region.mean()
            label = str(int(val)) if pool_type == 'max' else f'{val:.2f}'

            rect = plt.Rectangle((x0 + ci * cell_out, oy0 - (ri+1) * cell_out),
                                  cell_out, cell_out,
                                  facecolor=region_colors[ri][ci],
                                  edgecolor=region_edge[ri][ci],
                                  linewidth=2.5, zorder=2)
            ax.add_patch(rect)
            ax.text(x0 + ci * cell_out + cell_out/2,
                    oy0 - (ri+1) * cell_out + cell_out/2,
                    label, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='#111111', zorder=3)

            vals_str = ', '.join(str(v) for v in region)
            if pool_type == 'max':
                comp_lines.append(f'max({vals_str}) = {int(val)}')
            else:
                comp_lines.append(f'avg({vals_str}) = {val:.2f}')

    # ── Computation box (to the right of output grid) ─────────
    comp_x = x0 + 2 * cell_out + 0.4
    comp_y = oy0 - cell_out   # vertically centred on output grid
    ax.text(comp_x, comp_y, '\n'.join(comp_lines),
            ha='left', va='center', fontsize=8.5, color='#374151',
            linespacing=1.8, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9fafb',
                      edgecolor='#d1d5db', linewidth=1.2))

draw_pooling(axes[0], fm, 'max',
             'Max Pooling  (2×2, stride = 2)\nKeeps the strongest activation in each region')
draw_pooling(axes[1], fm, 'avg',
             'Average Pooling  (2×2, stride = 2)\nAverages all activations in each region')

out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_pooling.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_pooling.png")
