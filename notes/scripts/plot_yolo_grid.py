import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('white')

S = 4

# Left — image divided into S×S grid
ax = axes[0]
ax.set_xlim(0, S)
ax.set_ylim(0, S)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('white')
ax.set_title('S×S grid over image', fontsize=11, fontweight='bold', color='#111111')

img = mpatches.FancyBboxPatch((0, 0), S, S,
    boxstyle="square,pad=0", linewidth=1.5,
    edgecolor='#9ca3af', facecolor='#f3f4f6')
ax.add_patch(img)

for i in range(1, S):
    ax.plot([i, i], [0, S], color='#d1d5db', lw=1.0)
    ax.plot([0, S], [i, i], color='#d1d5db', lw=1.0)

highlight = mpatches.Rectangle((1, 2), 1, 1,
    linewidth=2.5, edgecolor='#dc2626', facecolor='#fee2e2', alpha=0.5)
ax.add_patch(highlight)

obj = mpatches.Rectangle((0.7, 1.3), 1.8, 1.5,
    linewidth=2.5, edgecolor='#2563eb', facecolor='none')
ax.add_patch(obj)

obj_cx, obj_cy = 0.7 + 1.8/2, 1.3 + 1.5/2
ax.plot(obj_cx, obj_cy, 'o', color='#dc2626', markersize=8, zorder=5)
ax.text(obj_cx + 0.12, obj_cy + 0.12, 'centre\n(responsible cell)',
        fontsize=7.5, color='#dc2626')

ax.text(S/2, -0.35, 'Cell containing object centre is responsible for detection',
        ha='center', fontsize=8, color='#374151', style='italic')

# Right — what each cell predicts
ax2 = axes[1]
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 7)
ax2.axis('off')
ax2.set_facecolor('white')
ax2.set_title('Each cell predicts B boxes + class scores', fontsize=11, fontweight='bold', color='#111111')

cell = mpatches.FancyBboxPatch((0.3, 4.5), 1.2, 1.2,
    boxstyle="round,pad=0.1", linewidth=2,
    edgecolor='#dc2626', facecolor='#fee2e2')
ax2.add_patch(cell)
ax2.text(0.9, 5.1, 'cell\n(i,j)', ha='center', va='center',
         fontsize=9, fontweight='bold', color='#dc2626')

ax2.annotate('', xy=(2.2, 5.8), xytext=(1.5, 5.8),
             arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.5))
ax2.annotate('', xy=(2.2, 5.1), xytext=(1.5, 5.1),
             arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.5))

boxes = [
    (2.2, 5.5, '#2563eb', '#dbeafe', 'Box 1: (cx, cy, w, h, conf)'),
    (2.2, 4.8, '#15803d', '#dcfce7', 'Box 2: (cx, cy, w, h, conf)'),
]
for bx, by, col, fill, label in boxes:
    rect = mpatches.FancyBboxPatch((bx, by), 3.5, 0.55,
        boxstyle="round,pad=0.05", linewidth=1.5,
        edgecolor=col, facecolor=fill)
    ax2.add_patch(rect)
    ax2.text(bx + 1.75, by + 0.27, label, ha='center', va='center',
             fontsize=8.5, color='#111111', fontweight='bold')

ax2.annotate('', xy=(2.2, 3.8), xytext=(1.5, 4.2),
             arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.5))

class_box = mpatches.FancyBboxPatch((2.2, 3.3), 3.5, 0.55,
    boxstyle="round,pad=0.05", linewidth=1.5,
    edgecolor='#b45309', facecolor='#fef3c7')
ax2.add_patch(class_box)
ax2.text(3.95, 3.57, 'Class scores: P(cat), P(dog), ...',
         ha='center', va='center', fontsize=8.5,
         color='#111111', fontweight='bold')

ax2.text(3.0, 2.6,
         'Output tensor: S × S × (B×5 + C)',
         ha='center', fontsize=9.5, color='#111111',
         bbox=dict(facecolor='#fef9c3', edgecolor='#ca8a04', pad=5))
ax2.text(3.0, 1.9,
         'S=grid size,  B=boxes per cell,  C=classes',
         ha='center', fontsize=8.5, color='#374151')
ax2.text(3.0, 1.3,
         'Example: S=7, B=2, C=20  ->  7×7×30 output',
         ha='center', fontsize=8.5, color='#374151')

ax2.text(3.0, 0.5,
         'One forward pass — no region proposals',
         ha='center', fontsize=9, color='#111111', fontweight='bold',
         bbox=dict(facecolor='#fee2e2', edgecolor='#dc2626', pad=4))

plt.suptitle('YOLO — Single-Pass Detection with Grid Cells',
             fontsize=12, fontweight='bold', color='#111111', y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'yolo_grid.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved yolo_grid.png")
