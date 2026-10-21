import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor('white')
fig.suptitle('Effect of Stride and Padding on Output Size\n(3×3 filter applied to 5×5 input)',
             fontsize=13, fontweight='bold', color='#222', y=0.98)

cell = 0.7

def draw_panel(ax, inp_size, pad, stride, filter_size=3, title='', formula=''):
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', color='#222', pad=10)

    total = inp_size + 2 * pad
    out_size = (inp_size + 2 * pad - filter_size) // stride + 1

    # Fixed layout: input grid starts at (0.3, top), output grid to the right
    x0 = 0.3
    y0 = total * cell + 1.2   # top of input grid

    # --- Draw padded input ---
    for r in range(total):
        for c in range(total):
            is_pad = (r < pad or r >= inp_size + pad or c < pad or c >= inp_size + pad)
            color = '#ccd4ff' if is_pad else '#f3f4f6'
            edge = '#7788cc' if is_pad else '#aaa'
            lw = 1.2 if is_pad else 0.8
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor=color, edgecolor=edge, linewidth=lw, zorder=2)
            ax.add_patch(rect)
            if is_pad:
                ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2, '0',
                        ha='center', va='center', fontsize=8, color='#5566aa', zorder=3)

    # --- Draw first filter position (yellow) ---
    for dr in range(filter_size):
        for dc in range(filter_size):
            r, c = pad + dr, pad + dc
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor='#ffdd55', edgecolor='#cc8800',
                                  linewidth=2.0, alpha=0.75, zorder=3)
            ax.add_patch(rect)

    # --- Draw second filter position if stride > 1 (orange) ---
    if stride > 1:
        c2_start = pad + stride
        if c2_start + filter_size <= total:
            for dr in range(filter_size):
                for dc in range(filter_size):
                    r, c = pad + dr, c2_start + dc
                    rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                          facecolor='#ff9944', edgecolor='#cc4400',
                                          linewidth=2.0, alpha=0.75, zorder=3)
                    ax.add_patch(rect)

    # Input label
    ax.text(x0 + total * cell / 2, y0 + 0.45,
            f'Input {inp_size}×{inp_size}' + (f'  (+ {pad} zero pad each side)' if pad else ''),
            ha='center', fontsize=9.5, fontweight='bold', color='#333')

    # Padding legend
    if pad > 0:
        legend_x = x0 + total * cell + 0.2
        rect_leg = plt.Rectangle((legend_x, y0 - 0.5 * cell), 0.4, 0.4,
                                   facecolor='#ccd4ff', edgecolor='#7788cc', linewidth=1.2)
        ax.add_patch(rect_leg)
        ax.text(legend_x + 0.5, y0 - 0.3 * cell, '= zero pad', ha='left', fontsize=8, color='#5566aa')

    # --- Arrow ---
    arrow_x_from = x0 + total * cell + (0.7 if pad == 0 else 1.2)
    arrow_x_to = arrow_x_from + 0.7
    arrow_y = y0 - total * cell / 2 + cell / 2
    ax.annotate('', xy=(arrow_x_to, arrow_y), xytext=(arrow_x_from, arrow_y),
                arrowprops=dict(arrowstyle='->', color='#444', lw=2.0))
    ax.text((arrow_x_from + arrow_x_to) / 2, arrow_y + 0.25, 'conv',
            ha='center', fontsize=8.5, color='#444')

    # --- Output grid ---
    ox0 = arrow_x_to + 0.3
    oy0 = y0 - (total - out_size) * cell / 2   # vertically centred
    for r in range(out_size):
        for c in range(out_size):
            rect = plt.Rectangle((ox0 + c * cell, oy0 - r * cell), cell, cell,
                                  facecolor='#dcfce7', edgecolor='#15803d', linewidth=1.5, zorder=2)
            ax.add_patch(rect)

    ax.text(ox0 + out_size * cell / 2, oy0 + 0.45,
            f'Output {out_size}×{out_size}',
            ha='center', fontsize=9.5, fontweight='bold', color='#15803d')

    # --- Formula box at bottom ---
    mid_x = (x0 + total * cell / 2 + ox0 + out_size * cell / 2) / 2
    ax.text(mid_x, 0.5, formula, ha='center', fontsize=9.5, color='#333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9c3', edgecolor='#ca8a04', linewidth=1.2))

    # Set axis limits with enough room
    ax.set_xlim(-0.2, ox0 + out_size * cell + 0.8)
    ax.set_ylim(0.0, y0 + 0.9)

draw_panel(axes[0, 0], 5, 0, 1, title='Stride = 1,  No Padding',
           formula='(5 − 3) / 1 + 1 = 3   →   output: 3×3')
draw_panel(axes[0, 1], 5, 0, 2, title='Stride = 2,  No Padding',
           formula='(5 − 3) / 2 + 1 = 2   →   output: 2×2')
draw_panel(axes[1, 0], 5, 1, 1, title='Stride = 1,  Padding = 1  ("same")',
           formula='(5 + 2 − 3) / 1 + 1 = 5   →   output: 5×5')
draw_panel(axes[1, 1], 5, 1, 2, title='Stride = 2,  Padding = 1',
           formula='(5 + 2 − 3) / 2 + 1 = 3   →   output: 3×3')

plt.tight_layout(pad=2.5, rect=[0, 0, 1, 0.96])
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_stride_padding.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_stride_padding.png")
