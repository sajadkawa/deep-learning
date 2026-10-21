import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')

# Left — grid cell with anchor boxes
ax = axes[0]
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('white')
ax.set_title('Anchor boxes at one grid cell', fontsize=11, fontweight='bold', color='#111111')

img = mpatches.FancyBboxPatch((0.2, 0.2), 5.6, 5.6,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor='#9ca3af', facecolor='#f3f4f6')
ax.add_patch(img)

for i in range(1, 4):
    ax.plot([0.2 + i*5.6/4, 0.2 + i*5.6/4], [0.2, 5.8], color='#d1d5db', lw=0.8)
    ax.plot([0.2, 5.8], [0.2 + i*5.6/4, 0.2 + i*5.6/4], color='#d1d5db', lw=0.8)

cx, cy = 3.0, 3.0

anchors = [
    (2.4, 2.4, '#2563eb', '#dbeafe', 'square'),
    (3.6, 1.6, '#dc2626', '#fee2e2', 'wide'),
    (1.6, 3.6, '#15803d', '#dcfce7', 'tall'),
]
for w, h, col, fill, label in anchors:
    rect = mpatches.Rectangle((cx - w/2, cy - h/2), w, h,
        linewidth=2.2, edgecolor=col, facecolor=fill, alpha=0.3)
    ax.add_patch(rect)

ax.plot(cx, cy, 'o', color='#111111', markersize=5)

for w, h, col, fill, label in anchors:
    ax.plot([], [], color=col, lw=2.2, label=label)
ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)

# Right — bounding box parameterisation
ax2 = axes[1]
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 6)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_facecolor('white')
ax2.set_title('Bounding box parameterisation', fontsize=11, fontweight='bold', color='#111111')

bx, by, bw, bh = 1.5, 1.8, 3.0, 2.4
rect = mpatches.Rectangle((bx, by), bw, bh,
    linewidth=2.5, edgecolor='#2563eb', facecolor='#dbeafe', alpha=0.3)
ax2.add_patch(rect)

cx2, cy2 = bx + bw/2, by + bh/2
ax2.plot(cx2, cy2, 'o', color='#2563eb', markersize=7)

ax2.annotate('', xy=(bx + bw, cy2), xytext=(bx, cy2),
             arrowprops=dict(arrowstyle='<->', color='#dc2626', lw=1.8))
ax2.text(cx2, cy2 + 0.25, 'w', ha='center', fontsize=11,
         color='#dc2626', fontweight='bold')

ax2.annotate('', xy=(cx2 + 0.15, by + bh), xytext=(cx2 + 0.15, by),
             arrowprops=dict(arrowstyle='<->', color='#15803d', lw=1.8))
ax2.text(cx2 + 0.45, cy2, 'h', ha='center', fontsize=11,
         color='#15803d', fontweight='bold')

ax2.plot(bx, by, 's', color='#9ca3af', markersize=7)
ax2.text(bx - 0.15, by - 0.3, '(x, y)', ha='center', fontsize=9.5, color='#6b7280')
ax2.text(cx2, cy2 - 0.35, '(cx, cy)', ha='center', fontsize=9, color='#2563eb')

ax2.text(3.0, 0.4, 'Box = (x, y, w, h)  or  (cx, cy, w, h)',
         ha='center', fontsize=9.5, color='#111111',
         bbox=dict(facecolor='#fef9c3', edgecolor='#ca8a04', pad=4))

plt.suptitle('Anchor Boxes and Bounding Box Parameterisation',
             fontsize=12, fontweight='bold', color='#111111', y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'anchor_boxes.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved anchor_boxes.png")
