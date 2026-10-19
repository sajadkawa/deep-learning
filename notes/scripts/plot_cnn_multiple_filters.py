import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.patch.set_facecolor('white')

dx = 0.28
dy = 0.14

def draw_box(ax, x0, y0, W, H, D, front, top, right, border='#374151', zorder=3):
    sx, sy = D * dx, D * dy
    ax.add_patch(plt.Polygon(
        [[x0, y0], [x0+W, y0], [x0+W, y0+H], [x0, y0+H]],
        facecolor=front, edgecolor=border, linewidth=1.2, zorder=zorder))
    ax.add_patch(plt.Polygon(
        [[x0, y0+H], [x0+W, y0+H], [x0+W+sx, y0+H+sy], [x0+sx, y0+H+sy]],
        facecolor=top, edgecolor=border, linewidth=1.2, zorder=zorder))
    ax.add_patch(plt.Polygon(
        [[x0+W, y0], [x0+W+sx, y0+sy], [x0+W+sx, y0+H+sy], [x0+W, y0+H]],
        facecolor=right, edgecolor=border, linewidth=1.2, zorder=zorder))

# ── Left panel: 3D input + K 3D filters → K feature maps ─────────────────────
ax = axes[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('white')
ax.set_title('K Filters (each F×F×C_in) → K Feature Maps', fontsize=11,
             fontweight='bold', color='#111111', pad=10)

# Input volume — 3D box
draw_box(ax, 0.3, 4.5, 1.4, 1.4, 2.0,
         front='#dbeafe', top='#bfdbfe', right='#93c5fd', border='#2563eb')
ax.text(0.3 + 0.7 + 2.0*dx/2, 4.5 + 1.4 + 2.0*dy + 0.2,
        'Input\nH × W × C_in',
        ha='center', fontsize=9, fontweight='bold', color='#111111')

# Label between input and filters
ax.text(2.55, 5.5, 'applied\nto each', ha='center', va='center',
        fontsize=7.5, color='#6b7280', fontstyle='italic')

# K filters — each shown as a small 3D box with same depth as input
filter_colors = [
    ('#fee2e2', '#fecaca', '#fca5a5', '#dc2626'),
    ('#dbeafe', '#bfdbfe', '#93c5fd', '#2563eb'),
    ('#dcfce7', '#bbf7d0', '#86efac', '#15803d'),
    ('#fef3c7', '#fde68a', '#fcd34d', '#b45309'),
]
filter_labels = ['Filter 1', 'Filter 2', 'Filter 3', 'Filter 4']
row_y = [9.0, 7.2, 5.4, 3.6]

for k in range(4):
    y = row_y[k]
    fr, tp, ri, bd = filter_colors[k]

    # 3D filter box — same depth (C_in) as input
    draw_box(ax, 3.1, y - 0.3, 0.55, 0.55, 2.0,
             front=fr, top=tp, right=ri, border=bd, zorder=4)

    # depth = C_in label on first filter only
    if k == 0:
        ax.text(3.1 + 0.55 + 2.0*dx + 0.05, y - 0.3 + 0.55 + 2.0*dy,
                'depth = C_in', ha='left', va='bottom',
                fontsize=7.5, color='#6b7280', fontstyle='italic')

    # filter label
    ax.text(3.1 + 0.55 + 2.0*dx + 0.1, y - 0.02,
            f'{filter_labels[k]}  (F × F × C_in)',
            ha='left', va='center', fontsize=8.5, color=bd, fontweight='bold')

    # arrow to feature map
    ax.annotate('', xy=(7.8, y - 0.02), xytext=(6.9, y - 0.02),
                arrowprops=dict(arrowstyle='->', color=bd, lw=1.8))

    # feature map (flat — depth 1)
    np.random.seed(k * 13 + 5)
    for r in range(3):
        for c in range(3):
            val = np.random.rand()
            cmap_color = plt.colormaps['RdYlGn'](val)[:3]
            rect = plt.Rectangle((7.9 + c * 0.5, y + 0.25 - r * 0.5), 0.47, 0.47,
                                  facecolor=cmap_color, edgecolor='#d1d5db', linewidth=0.5)
            ax.add_patch(rect)

    ax.text(7.9 + 0.75, y + 0.25 - 3*0.5 - 0.2, f'Feature map {k+1}',
            ha='center', fontsize=8, color='#374151')

ax.text(7.9 + 0.75, 9.9, "H' × W' × 1  each",
        ha='center', fontsize=9, fontweight='bold', color='#111111')

# ── Right panel: stacked feature maps → output volume ────────────────────────
ax2 = axes[1]
ax2.set_xlim(0, 8)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_facecolor('white')
ax2.set_title("K Feature Maps Stacked → Output Volume  (H' × W' × K)",
              fontsize=11, fontweight='bold', color='#111111', pad=10)

slab_w = 3.5
slab_h = 3.5
sdx = 0.35
sdy = 0.35
x_base = 1.5
y_base = 2.5

fill_colors   = ['#fee2e2', '#dbeafe', '#dcfce7', '#fef3c7']
border_colors = ['#dc2626', '#2563eb', '#15803d', '#b45309']

for k in range(4):
    xk = x_base + k * sdx
    yk = y_base + k * sdy
    rect = mpatches.FancyBboxPatch((xk, yk), slab_w, slab_h,
                                    boxstyle='square,pad=0',
                                    facecolor=fill_colors[k],
                                    edgecolor=border_colors[k],
                                    linewidth=2.0, zorder=k + 2)
    ax2.add_patch(rect)
    ax2.text(xk + slab_w/2, yk + slab_h/2,
             f'Feature\nMap {k+1}',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color='#111111', zorder=k + 10)

top_k = 3
# Width arrow
ax2.annotate('', xy=(x_base + slab_w, y_base - 0.4),
             xytext=(x_base, y_base - 0.4),
             arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.5))
ax2.text(x_base + slab_w/2, y_base - 0.65, "W'",
         ha='center', fontsize=10, color='#374151', fontweight='bold')

# Height arrow
ax2.annotate('', xy=(x_base - 0.4, y_base + slab_h),
             xytext=(x_base - 0.4, y_base),
             arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.5))
ax2.text(x_base - 0.55, y_base + slab_h/2, "H'",
         ha='right', va='center', fontsize=10, color='#374151',
         fontweight='bold', rotation=90)

# Depth arrow
ax2.annotate('', xy=(x_base + top_k*sdx + 0.3, y_base + top_k*sdy + slab_h + 0.3),
             xytext=(x_base + 0.3, y_base + slab_h + 0.3),
             arrowprops=dict(arrowstyle='->', color='#374151', lw=1.5))
ax2.text(x_base + (top_k*sdx)/2 + 0.5, y_base + slab_h + 0.55,
         'K  (one slice per filter)', ha='center', fontsize=9, color='#374151',
         fontweight='bold')

ax2.text(3.5, 1.2,
         'Each filter learns a different pattern.\nK filters → K feature maps → output depth K.',
         ha='center', va='center', fontsize=9.5, color='#374151',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#dbeafe',
                   edgecolor='#2563eb', linewidth=1.5))

plt.tight_layout(pad=2.0)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_multiple_filters.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_multiple_filters.png")
