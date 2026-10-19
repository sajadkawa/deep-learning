import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.patch.set_facecolor('white')

# ---- Left panel: FC on image ----
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor('white')
ax.set_title('Fully Connected on a 4×4 Image\n(16 inputs × 4 neurons = 64 weights)', fontsize=10, fontweight='bold', color='#111111')

img_vals = np.array([[120, 80, 40, 10],
                     [200, 150, 90, 30],
                     [180, 130, 70, 20],
                     [100, 60, 30, 5]])
for r in range(4):
    for c in range(4):
        val = img_vals[r, c] / 255.0
        color = (val, val, val)
        rect = plt.Rectangle((0.2 + c * 0.7, 5.5 - r * 0.7), 0.65, 0.65,
                              facecolor=color, edgecolor='#d1d5db', linewidth=0.8)
        ax.add_patch(rect)

ax.text(1.6, 7.6, '4×4 image\n(16 pixels)', ha='center', fontsize=8, color='#374151')

ax.annotate('', xy=(3.2, 6.2), xytext=(2.8, 6.2),
            arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.5))
ax.text(3.0, 6.55, 'flatten', ha='center', fontsize=7, color='#374151')

node_y = [7.2 - i * 0.28 for i in range(16)]
for i, y in enumerate(node_y):
    circ = plt.Circle((3.6, y), 0.10, facecolor='#dbeafe', edgecolor='#2563eb', linewidth=0.8, zorder=3)
    ax.add_patch(circ)

ax.text(3.6, 7.55, '16\nnodes', ha='center', fontsize=7, color='#2563eb')

out_y = [7.0 - i * 0.7 for i in range(4)]
for y in out_y:
    circ = plt.Circle((6.0, y), 0.18, facecolor='#fef3c7', edgecolor='#b45309', linewidth=0.8, zorder=3)
    ax.add_patch(circ)

ax.text(6.0, 7.55, '4\nneurons', ha='center', fontsize=7, color='#b45309')

for iy in node_y:
    for oy in out_y:
        ax.plot([3.70, 5.82], [iy, oy], color='#d1d5db', linewidth=0.3, alpha=0.5, zorder=1)

ax.text(4.8, 2.2, '64 weights\n(16 × 4)', ha='center', fontsize=9, color='#dc2626',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626'))

ax.text(4.8, 1.1, 'For a 224×224 image:\n224×224×512 = 25.7M weights\nper layer', ha='center', fontsize=8, color='#dc2626',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626'))

# ---- Right panel: spatial structure lost ----
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_facecolor('white')
ax2.set_title('Spatial Structure Is Lost After Flattening', fontsize=10, fontweight='bold', color='#111111')

for r in range(4):
    for c in range(4):
        val = img_vals[r, c] / 255.0
        color = (val, val, val)
        rect = plt.Rectangle((0.5 + c * 0.85, 4.5 - r * 0.85), 0.8, 0.8,
                              facecolor=color, edgecolor='#d1d5db', linewidth=0.8)
        ax2.add_patch(rect)

ax2.text(2.2, 8.0, 'Spatial grid\n(neighbours matter)', ha='center', fontsize=8.5, color='#15803d', fontweight='bold')

ax2.annotate('', xy=(1.35 + 0.85, 4.5 - 0 * 0.85 + 0.4), xytext=(1.35, 4.5 - 0 * 0.85 + 0.4),
            arrowprops=dict(arrowstyle='<->', color='#15803d', lw=1.5))
ax2.text(1.78, 5.35, 'adjacent', ha='center', fontsize=7, color='#15803d')

ax2.annotate('', xy=(6.0, 6.2), xytext=(4.2, 6.2),
            arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.5))
ax2.text(5.1, 6.55, 'flatten', ha='center', fontsize=7, color='#374151')

flat_y = [7.4 - i * 0.38 for i in range(16)]
for i, y in enumerate(flat_y):
    val = img_vals[i // 4, i % 4] / 255.0
    color = (val, val, val)
    rect = plt.Rectangle((6.2, y - 0.14), 0.5, 0.28,
                          facecolor=color, edgecolor='#d1d5db', linewidth=0.8)
    ax2.add_patch(rect)

ax2.text(6.45, 7.75, 'flat\nvector', ha='center', fontsize=7, color='#374151')

ax2.text(6.45, 1.5, 'Pixel 1 and pixel 16\nlook equally "close"\nto the network', ha='center', fontsize=8.5, color='#dc2626',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#dc2626'))

plt.tight_layout(pad=1.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_fc_problem.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_fc_problem.png")
