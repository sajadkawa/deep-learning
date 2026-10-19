import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.patch.set_facecolor('white')

# ---- Left panel: K filters → K feature maps (2D, clear layout) ----
ax = axes[0]
ax.set_xlim(0, 11)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('white')
ax.set_title('K Filters → K Feature Maps', fontsize=12, fontweight='bold', color='#111111', pad=10)

# Input image (top-left)
np.random.seed(7)
for r in range(4):
    for c in range(4):
        val = np.random.rand() * 0.5 + 0.3
        rect = plt.Rectangle((0.3 + c * 0.6, 7.5 - r * 0.6), 0.57, 0.57,
                              facecolor=(val, val, val), edgecolor='#888', linewidth=0.6)
        ax.add_patch(rect)
ax.text(0.3 + 2 * 0.6, 9.9, 'Input  (H × W × 1)', ha='center', fontsize=10, fontweight='bold', color='#111111')

# Arrow from input to filters
ax.annotate('', xy=(3.2, 8.1), xytext=(2.8, 8.1),
            arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=2.0))

filter_colors = ['#dc2626', '#2563eb', '#15803d', '#b45309']
filter_labels = ['Filter 1  (edges)', 'Filter 2  (corners)', 'Filter 3  (textures)', 'Filter 4  (blobs)']

# Each filter row: filter box | arrow | feature map
row_y = [9.0, 7.2, 5.4, 3.6]   # well-spaced rows

for k in range(4):
    y = row_y[k]
    fc = filter_colors[k]

    # Filter box (2×2 mini grid)
    for r in range(2):
        for c in range(2):
            rect = plt.Rectangle((3.3 + c * 0.45, y - r * 0.45), 0.42, 0.42,
                                  facecolor=fc, edgecolor='white', linewidth=1.2, alpha=0.3)
            ax.add_patch(rect)

    # Filter label — to the right of the box, same row
    ax.text(3.3 + 2 * 0.45 + 0.15, y - 0.22, filter_labels[k],
            ha='left', va='center', fontsize=9, color=fc, fontweight='bold')

    # Arrow
    ax.annotate('', xy=(6.8, y - 0.22), xytext=(5.8, y - 0.22),
                arrowprops=dict(arrowstyle='->', color=fc, lw=1.8))

    # Feature map (3×3 mini grid)
    np.random.seed(k * 13 + 5)
    for r in range(3):
        for c in range(3):
            val = np.random.rand()
            cmap_color = plt.colormaps['RdYlGn'](val)[:3]
            rect = plt.Rectangle((6.9 + c * 0.5, y + 0.25 - r * 0.5), 0.47, 0.47,
                                  facecolor=cmap_color, edgecolor='#888', linewidth=0.5)
            ax.add_patch(rect)

    # Feature map label — below each map
    ax.text(6.9 + 1.5 * 0.5, y + 0.25 - 3 * 0.5 - 0.2, f'Feature map {k+1}',
            ha='center', fontsize=8, color='#374151')

ax.text(6.9 + 1.5 * 0.5, 9.9, "Feature Maps  (H' × W' × K)", ha='center', fontsize=10, fontweight='bold', color='#111111')

# ---- Right panel: Volume interpretation (2D schematic, no 3D) ----
ax2 = axes[1]
ax2.set_xlim(0, 8)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_facecolor('white')
ax2.set_title("Output Volume  (H' × W' × K)", fontsize=12, fontweight='bold', color='#111111', pad=10)

# Draw stacked slabs as offset rectangles (isometric-style 2D)
slab_w = 3.5
slab_h = 3.5
dx = 0.35   # horizontal offset per layer
dy = 0.35   # vertical offset per layer
x_base = 1.5
y_base = 2.5

for k in range(4):
    xk = x_base + k * dx
    yk = y_base + k * dy
    fc = filter_colors[k]
    fill_colors = ['#fee2e2', '#dbeafe', '#dcfce7', '#fef3c7']
    border_colors = ['#dc2626', '#2563eb', '#15803d', '#b45309']

    # Main face
    rect = mpatches.FancyBboxPatch((xk, yk), slab_w, slab_h,
                                    boxstyle='square,pad=0',
                                    facecolor=fill_colors[k], edgecolor=border_colors[k],
                                    linewidth=2.0, zorder=k + 2)
    ax2.add_patch(rect)

    # Label inside slab
    ax2.text(xk + slab_w / 2, yk + slab_h / 2,
             f'Feature\nMap {k + 1}',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color='#111111', zorder=k + 10)

# Dimension arrows
top_k = 3
xk_top = x_base + top_k * dx
yk_top = y_base + top_k * dy

# Width arrow (bottom)
ax2.annotate('', xy=(x_base + slab_w, y_base - 0.4),
             xytext=(x_base, y_base - 0.4),
             arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.5))
ax2.text(x_base + slab_w / 2, y_base - 0.65, "Width (W')", ha='center', fontsize=9, color='#374151')

# Height arrow (left side)
ax2.annotate('', xy=(x_base - 0.4, y_base + slab_h),
             xytext=(x_base - 0.4, y_base),
             arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.5))
ax2.text(x_base - 0.55, y_base + slab_h / 2, "Height (H')",
         ha='right', va='center', fontsize=9, color='#374151', rotation=90)

# Depth arrow (diagonal)
ax2.annotate('', xy=(x_base + top_k * dx + 0.3, y_base + top_k * dy + slab_h + 0.3),
             xytext=(x_base + 0.3, y_base + slab_h + 0.3),
             arrowprops=dict(arrowstyle='->', color='#374151', lw=1.5))
ax2.text(x_base + (top_k * dx) / 2 + 0.5, y_base + slab_h + 0.55,
         'Depth = K filters', ha='center', fontsize=9, color='#374151')

# Key insight box
ax2.text(3.5, 1.2,
         'Each filter learns a different pattern.\nK filters → K feature maps → depth K.',
         ha='center', va='center', fontsize=9.5, color='#374151',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#dbeafe', edgecolor='#2563eb', linewidth=1.5))

plt.tight_layout(pad=2.0)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_multiple_filters.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_multiple_filters.png")
