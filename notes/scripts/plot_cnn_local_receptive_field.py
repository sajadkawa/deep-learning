import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('white')

def draw_grid(ax, rows, cols, x0, y0, cell=0.7, highlight=None):
    for r in range(rows):
        for c in range(cols):
            is_hl = highlight and (r, c) in highlight
            color = '#dbeafe' if is_hl else '#f3f4f6'
            edge  = '#2563eb' if is_hl else '#d1d5db'
            lw    = 2.0 if is_hl else 0.8
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor=color, edgecolor=edge, linewidth=lw, zorder=2)
            ax.add_patch(rect)

# ---- Left: FC neuron ----
ax = axes[0]
ax.set_xlim(0, 9)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('white')
ax.set_title('Fully Connected Neuron\n(sees all 25 pixels)', fontsize=11, fontweight='bold', color='#111111')

draw_grid(ax, 5, 5, 0.5, 5.5, cell=0.65, highlight={(r,c) for r in range(5) for c in range(5)})

circ = plt.Circle((7.0, 3.2), 0.35, facecolor='#fef3c7', edgecolor='#b45309', linewidth=1.5, zorder=4)
ax.add_patch(circ)
ax.text(7.0, 3.2, 'n', ha='center', va='center', fontsize=10, color='#111111', fontweight='bold')

for r in range(5):
    for c in range(5):
        px = 0.5 + c * 0.65 + 0.325
        py = 5.5 - r * 0.65 + 0.325
        ax.plot([px, 6.65], [py, 3.2], color='#bfdbfe', linewidth=0.25, alpha=0.6, zorder=1)

ax.text(3.6, 0.8, '25 connections\n(one per pixel)', ha='center', fontsize=10, color='#dc2626',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626'))

# ---- Right: Local receptive field ----
ax2 = axes[1]
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 7)
ax2.axis('off')
ax2.set_facecolor('white')
ax2.set_title('CNN Neuron with Local Receptive Field\n(sees only 3×3 = 9 pixels)', fontsize=11, fontweight='bold', color='#111111')

patch_cells = {(r, c) for r in range(1, 4) for c in range(1, 4)}
draw_grid(ax2, 5, 5, 0.5, 5.5, cell=0.65, highlight=patch_cells)

rect_border = plt.Rectangle((0.5 + 1 * 0.65, 5.5 - 3 * 0.65), 3 * 0.65, 3 * 0.65,
                              facecolor='none', edgecolor='#b45309', linewidth=2.5, zorder=5,
                              linestyle='--')
ax2.add_patch(rect_border)
ax2.text(0.5 + 1 * 0.65 + 1.5 * 0.65, 5.5 - 3 * 0.65 - 0.25,
         '3×3 receptive\nfield', ha='center', fontsize=8, color='#b45309')

circ2 = plt.Circle((7.0, 3.2), 0.35, facecolor='#dcfce7', edgecolor='#15803d', linewidth=1.5, zorder=4)
ax2.add_patch(circ2)
ax2.text(7.0, 3.2, 'n', ha='center', va='center', fontsize=10, color='#111111', fontweight='bold')

for r in range(1, 4):
    for c in range(1, 4):
        px = 0.5 + c * 0.65 + 0.325
        py = 5.5 - r * 0.65 + 0.325
        ax2.plot([px, 6.65], [py, 3.2], color='#b45309', linewidth=1.0, alpha=0.7, zorder=3)

ax2.text(3.6, 0.8, '9 connections\n(one per patch pixel)', ha='center', fontsize=10, color='#15803d',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#dcfce7', edgecolor='#15803d'))

plt.tight_layout(pad=1.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_local_receptive_field.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_local_receptive_field.png")
