import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor('#f8f8f8')

def draw_image_grid(ax, data, x0, y0, cell=0.7, title=''):
    rows, cols = data.shape
    vmin, vmax = data.min(), data.max()
    for r in range(rows):
        for c in range(cols):
            val = (data[r, c] - vmin) / (vmax - vmin + 1e-9)
            color = (val, val, val)
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor=color, edgecolor='#888', linewidth=0.5, zorder=2)
            ax.add_patch(rect)
    if title:
        ax.text(x0 + cols * cell / 2, y0 + 0.3, title, ha='center', fontsize=9,
                fontweight='bold', color='#333')

def draw_feature_map(ax, data, x0, y0, cell=0.7, title='', highlight=None):
    rows, cols = data.shape
    vmin, vmax = data.min(), data.max()
    for r in range(rows):
        for c in range(cols):
            val = (data[r, c] - vmin) / (vmax - vmin + 1e-9)
            color = plt.cm.RdYlGn(val)
            edge = '#ff6600' if (highlight and (r, c) in highlight) else '#888'
            lw = 2.5 if (highlight and (r, c) in highlight) else 0.5
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor=color, edgecolor=edge, linewidth=lw, zorder=2)
            ax.add_patch(rect)
            ax.text(x0 + c * cell + cell/2, y0 - r * cell + cell/2,
                    f'{data[r,c]:.0f}', ha='center', va='center', fontsize=8, color='#222')
    if title:
        ax.text(x0 + cols * cell / 2, y0 + 0.3, title, ha='center', fontsize=9,
                fontweight='bold', color='#333')

# Vertical edge filter (Sobel-like)
v_filter = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]])

# Image 1: vertical edge on the LEFT
img1 = np.array([
    [200, 200, 50, 50, 50],
    [200, 200, 50, 50, 50],
    [200, 200, 50, 50, 50],
    [200, 200, 50, 50, 50],
    [200, 200, 50, 50, 50]
], dtype=float)

# Image 2: vertical edge on the RIGHT
img2 = np.array([
    [50, 50, 50, 200, 200],
    [50, 50, 50, 200, 200],
    [50, 50, 50, 200, 200],
    [50, 50, 50, 200, 200],
    [50, 50, 50, 200, 200]
], dtype=float)

def convolve(img, k):
    out = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            out[i, j] = np.sum(img[i:i+3, j:j+3] * k)
    return out

fm1 = convolve(img1, v_filter)
fm2 = convolve(img2, v_filter)

# ---- Panel 1: Image with edge on left ----
ax = axes[0]
ax.set_xlim(0, 5)
ax.set_ylim(-1.5, 5)
ax.axis('off')
ax.set_facecolor('#f8f8f8')
draw_image_grid(ax, img1, 0.1, 4.5, cell=0.75, title='Image A: edge on left')
# Mark edge location
ax.axvline(x=0.1 + 2 * 0.75, color='#ff6600', linewidth=2.5, linestyle='--', ymin=0.25, ymax=0.95)
ax.text(0.1 + 2 * 0.75, -0.3, 'edge\nhere', ha='center', fontsize=8, color='#ff6600')

# ---- Panel 2: Filter ----
ax2 = axes[1]
ax2.set_xlim(0, 5)
ax2.set_ylim(-1.5, 5)
ax2.axis('off')
ax2.set_facecolor('#f8f8f8')
ax2.set_title('Vertical Edge Filter\n(same filter, applied everywhere)', fontsize=9, fontweight='bold', color='#333')

filter_cell = 0.9
fx, fy = 1.1, 4.2
for r in range(3):
    for c in range(3):
        val = v_filter[r, c]
        color = '#ff6b6b' if val > 0 else ('#4a90d9' if val < 0 else '#eeeeee')
        rect = plt.Rectangle((fx + c * filter_cell, fy - r * filter_cell), filter_cell, filter_cell,
                              facecolor=color, edgecolor='#333', linewidth=1.5, zorder=3)
        ax2.add_patch(rect)
        ax2.text(fx + c * filter_cell + filter_cell/2, fy - r * filter_cell + filter_cell/2,
                 str(int(val)), ha='center', va='center', fontsize=12, fontweight='bold', color='white')

ax2.text(fx + 1.5 * filter_cell, fy + 0.35, '3×3 filter', ha='center', fontsize=9, color='#333')
ax2.text(2.5, 0.5, 'Detects: bright on left,\ndark on right', ha='center', fontsize=9, color='#555',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#aaa'))

# ---- Panel 3: Both feature maps ----
ax3 = axes[2]
ax3.set_xlim(0, 6)
ax3.set_ylim(-0.5, 8)
ax3.axis('off')
ax3.set_facecolor('#f8f8f8')
ax3.set_title('Feature Maps — Same Filter, Different Images\nHigh activation where edge is detected', fontsize=9, fontweight='bold', color='#333')

# Feature map for img1 (edge left → high activation at col 0)
draw_feature_map(ax3, fm1, 0.2, 7.2, cell=0.75, title='Feature map A', highlight={(r,0) for r in range(3)})
# Feature map for img2 (edge right → high activation at col 2)
draw_feature_map(ax3, fm2, 0.2, 3.8, cell=0.75, title='Feature map B', highlight={(r,2) for r in range(3)})

ax3.text(1.325, 0.3, 'Same filter detects the same\nfeature regardless of position', ha='center', fontsize=9,
         color='#1a7a1a', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8e8', edgecolor='#1a7a1a'))

plt.tight_layout(pad=1.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_translation_invariance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_translation_invariance.png")
