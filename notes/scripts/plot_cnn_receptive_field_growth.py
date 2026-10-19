import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(15, 7))
fig.patch.set_facecolor('white')
fig.suptitle('Receptive Field Grows with Depth\n(each 3×3 conv adds 2 to the effective receptive field)',
             fontsize=12, fontweight='bold', color='#111111', y=0.98)

cell = 0.62
grid_size = 7

rf_configs = [
    dict(rf=3, color='#dbeafe', title='After Layer 1\n(one 3×3 conv)',
         rf_label='Receptive field: 3×3',
         params_label='9 params per filter',
         extra_label=''),
    dict(rf=5, color='#dcfce7', title='After Layer 2\n(two 3×3 convs)',
         rf_label='Receptive field: 5×5',
         params_label='9 + 9 = 18 params',
         extra_label=''),
    dict(rf=7, color='#fef3c7', title='After Layer 3\n(three 3×3 convs)',
         rf_label='Receptive field: 7×7',
         params_label='9 + 9 + 9 = 27 params',
         extra_label='vs one 7×7 filter: 49 params\nSame RF — 45% fewer params'),
]

for ax, cfg in zip(axes, rf_configs):
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title(cfg['title'], fontsize=11, fontweight='bold', color='#111111', pad=8)

    center = grid_size // 2
    rf_half = cfg['rf'] // 2

    # Grid
    for r in range(grid_size):
        for c in range(grid_size):
            in_rf    = (abs(r - center) <= rf_half and abs(c - center) <= rf_half)
            is_center = (r == center and c == center)

            if is_center:
                fc, ec, lw = '#fee2e2', '#dc2626', 2.5
            elif in_rf:
                fc, ec, lw = cfg['color'], '#374151', 1.5
            else:
                fc, ec, lw = '#f3f4f6', '#d1d5db', 0.7

            rect = plt.Rectangle((c * cell, (grid_size - 1 - r) * cell), cell, cell,
                                  facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
            ax.add_patch(rect)

    # RF border (dashed red rectangle)
    c0 = center - rf_half
    r0_bottom = center - rf_half
    rect_b = plt.Rectangle((c0 * cell, r0_bottom * cell),
                             cfg['rf'] * cell, cfg['rf'] * cell,
                             facecolor='none', edgecolor='#dc2626',
                             linewidth=3.0, linestyle='--', zorder=5)
    ax.add_patch(rect_b)

    # RF label — below grid
    ax.text(grid_size * cell / 2, -0.5, cfg['rf_label'],
            ha='center', va='top', fontsize=10.5, color='#dc2626', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#fee2e2', edgecolor='#dc2626', linewidth=1.5))

    # Params label — above grid
    ax.text(grid_size * cell / 2, grid_size * cell + 0.25, cfg['params_label'],
            ha='center', va='bottom', fontsize=9, color='#374151',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9fafb', edgecolor='#d1d5db'))

    # Extra label (only panel 3) — well below RF label
    if cfg['extra_label']:
        ax.text(grid_size * cell / 2, -1.5, cfg['extra_label'],
                ha='center', va='top', fontsize=9, color='#15803d', linespacing=1.5,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#dcfce7', edgecolor='#15803d', linewidth=1.5))

    # Set limits with enough room for all labels
    ax.set_xlim(-0.4, grid_size * cell + 0.4)
    ax.set_ylim(-2.5, grid_size * cell + 0.8)

plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_receptive_field_growth.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_receptive_field_growth.png")
