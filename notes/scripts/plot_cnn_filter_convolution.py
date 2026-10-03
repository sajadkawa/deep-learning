import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor('#f8f8f8')

inp = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 0],
    [1, 0, 1, 2, 1],
    [2, 1, 0, 1, 2],
    [0, 1, 2, 1, 0]
])

kernel = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
])

output = np.array([
    [ 8, 12,  8],
    [ 4,  8, 10],
    [ 4,  4,  8]
])

# Position (0,1) — top-left at row 0, col 1 → output value 12
pos_r, pos_c = 0, 1

def draw_matrix(ax, data, x0, y0, cell=0.8, highlight=None, hl_color='#ffcc44',
                title='', fontsize=11, val_fontsize=12):
    rows, cols = data.shape
    for r in range(rows):
        for c in range(cols):
            color = hl_color if (highlight and (r, c) in highlight) else '#e8e8e8'
            edge = '#333' if (highlight and (r, c) in highlight) else '#aaa'
            lw = 2.2 if (highlight and (r, c) in highlight) else 0.8
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor=color, edgecolor=edge, linewidth=lw, zorder=2)
            ax.add_patch(rect)
            ax.text(x0 + c * cell + cell/2, y0 - r * cell + cell/2,
                    str(data[r, c]), ha='center', va='center',
                    fontsize=val_fontsize, fontweight='bold', color='#222', zorder=3)
    if title:
        ax.text(x0 + cols * cell / 2, y0 + 0.35, title, ha='center', fontsize=fontsize,
                fontweight='bold', color='#333')

# ---- Panel 1: Input with highlighted patch ----
ax = axes[0]
ax.set_xlim(-0.2, 5.5)
ax.set_ylim(-0.5, 5.5)
ax.axis('off')
ax.set_facecolor('#f8f8f8')
hl = {(pos_r + dr, pos_c + dc) for dr in range(3) for dc in range(3)}
draw_matrix(ax, inp, 0, 4.5, cell=0.85, highlight=hl, hl_color='#ffcc44', title='Input (5×5)')
# Border around patch
rect_b = plt.Rectangle((pos_c * 0.85, 4.5 - (pos_r + 2) * 0.85), 3 * 0.85, 3 * 0.85,
                         facecolor='none', edgecolor='#cc6600', linewidth=2.5, linestyle='--', zorder=5)
ax.add_patch(rect_b)
ax.text(pos_c * 0.85 + 1.5 * 0.85, 4.5 - (pos_r + 2) * 0.85 - 0.3,
        'current\npatch', ha='center', fontsize=8, color='#cc6600')

# ---- Panel 2: Kernel ----
ax2 = axes[1]
ax2.set_xlim(-0.5, 4.5)
ax2.set_ylim(-0.5, 4.5)
ax2.axis('off')
ax2.set_facecolor('#f8f8f8')
draw_matrix(ax2, kernel, 0.5, 3.5, cell=0.85, hl_color='#b3e6b3', title='Filter (3×3)')

# Element-wise products
patch = inp[pos_r:pos_r+3, pos_c:pos_c+3]
products = patch * kernel
ax2.text(1.775, 0.8, 'Element-wise\nproducts:', ha='center', fontsize=9, color='#555')
prod_str = '\n'.join([' '.join([f'{products[r,c]:+d}' for c in range(3)]) for r in range(3)])
ax2.text(1.775, -0.1, prod_str, ha='center', fontsize=10, color='#1a5c8a',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f4ff', edgecolor='#4a90d9'))

# ---- Panel 3: Output with highlighted position ----
ax3 = axes[2]
ax3.set_xlim(-0.5, 4.5)
ax3.set_ylim(-0.5, 4.5)
ax3.axis('off')
ax3.set_facecolor('#f8f8f8')
draw_matrix(ax3, output, 0.5, 3.5, cell=0.85, highlight={(pos_r, pos_c)}, hl_color='#ff9944', title='Output (3×3)')

# Annotation showing the sum
ax3.text(1.775, 0.8, f'Sum of products:\n1+2+0+2+3+0+0+0+2 = 12', ha='center', fontsize=9, color='#cc6600',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff4e0', edgecolor='#cc6600'))

# Arrows between panels
for ax_from, ax_to, x_from, x_to, y_pos in [
    (axes[0], axes[1], 4.3, -0.3, 2.5),
    (axes[1], axes[2], 4.0, -0.3, 2.5)
]:
    pass  # arrows via fig coords are complex; use text instead

axes[0].text(4.6, 2.2, '*', ha='center', fontsize=20, color='#555')
axes[1].text(4.2, 2.2, '=', ha='center', fontsize=20, color='#555')

fig.suptitle('Convolution: Filter Slides Over Input, Producing One Output Value Per Position',
             fontsize=11, fontweight='bold', y=1.01, color='#222')

plt.tight_layout(pad=1.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_filter_convolution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_filter_convolution.png")
