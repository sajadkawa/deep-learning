import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left — grid cell with anchor boxes
ax = axes[0]
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Anchor boxes at one grid cell', fontsize=11, fontweight='bold')

# image
img = mpatches.FancyBboxPatch((0.2, 0.2), 5.6, 5.6,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor='#888888', facecolor='#eeeeee')
ax.add_patch(img)

# grid lines
for i in range(1, 4):
    ax.plot([0.2 + i*5.6/4, 0.2 + i*5.6/4], [0.2, 5.8], color='#bbbbbb', lw=0.8)
    ax.plot([0.2, 5.8], [0.2 + i*5.6/4, 0.2 + i*5.6/4], color='#bbbbbb', lw=0.8)

cx, cy = 3.0, 3.0  # centre of highlighted cell

# anchor boxes — different aspect ratios, all centred at (cx, cy)
anchors = [
    (2.4, 2.4, '#4C72B0', 'square'),
    (3.6, 1.6, '#C44E52', 'wide'),
    (1.6, 3.6, '#55A868', 'tall'),
]
for w, h, col, label in anchors:
    rect = mpatches.Rectangle((cx - w/2, cy - h/2), w, h,
        linewidth=2.2, edgecolor=col, facecolor=col, alpha=0.12)
    ax.add_patch(rect)

# centre dot
ax.plot(cx, cy, 'ko', markersize=5)

# legend
for w, h, col, label in anchors:
    ax.plot([], [], color=col, lw=2.2, label=label)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

# Right — anchor box coordinates
ax2 = axes[1]
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 6)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Bounding box parameterisation', fontsize=11, fontweight='bold')

# draw one box with annotations
bx, by, bw, bh = 1.5, 1.8, 3.0, 2.4
rect = mpatches.Rectangle((bx, by), bw, bh,
    linewidth=2.5, edgecolor='#4C72B0', facecolor='#4C72B0', alpha=0.15)
ax2.add_patch(rect)

# centre point
cx2, cy2 = bx + bw/2, by + bh/2
ax2.plot(cx2, cy2, 'o', color='#4C72B0', markersize=7)

# annotations
ax2.annotate('', xy=(bx + bw, cy2), xytext=(bx, cy2),
             arrowprops=dict(arrowstyle='<->', color='#C44E52', lw=1.8))
ax2.text(cx2, cy2 + 0.25, 'w', ha='center', fontsize=11,
         color='#C44E52', fontweight='bold')

ax2.annotate('', xy=(cx2 + 0.15, by + bh), xytext=(cx2 + 0.15, by),
             arrowprops=dict(arrowstyle='<->', color='#55A868', lw=1.8))
ax2.text(cx2 + 0.45, cy2, 'h', ha='center', fontsize=11,
         color='#55A868', fontweight='bold')

ax2.plot(bx, by, 's', color='#888888', markersize=7)
ax2.text(bx - 0.15, by - 0.3, '(x, y)', ha='center', fontsize=9.5, color='#888888')
ax2.text(cx2, cy2 - 0.35, '(cx, cy)', ha='center', fontsize=9, color='#4C72B0')

ax2.text(3.0, 0.4, 'Box = (x, y, w, h)  or  (cx, cy, w, h)',
         ha='center', fontsize=9.5, color='#333333',
         bbox=dict(facecolor='#f5f5f5', edgecolor='#cccccc', pad=4))

plt.suptitle('Anchor Boxes and Bounding Box Parameterisation',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'anchor_boxes.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print("Saved anchor_boxes.png")
