import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#f8f8f8')
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_facecolor('#f8f8f8')

levels = [
    {'x': 0.4, 'label': 'Input\n(pixels)', 'sublabel': 'Raw pixel values\n0–255', 'color': '#95a5a6'},
    {'x': 3.2, 'label': 'Layer 1\n(edges)', 'sublabel': 'Horizontal edges\nVertical edges\nDiagonals', 'color': '#3498db'},
    {'x': 6.0, 'label': 'Layer 2\n(shapes)', 'sublabel': 'Corners\nCurves\nTextures', 'color': '#9b59b6'},
    {'x': 8.8, 'label': 'Layer 3\n(parts)', 'sublabel': 'Eyes, nose\nWheels, windows\nLeaves, petals', 'color': '#e67e22'},
    {'x': 11.6, 'label': 'Output\n(objects)', 'sublabel': 'Cat\nCar\nFlower', 'color': '#27ae60'},
]

box_w = 2.2
box_h = 4.2

for lv in levels:
    # Box
    rect = mpatches.FancyBboxPatch((lv['x'], 0.8), box_w, box_h,
                                    boxstyle='round,pad=0.1',
                                    facecolor=lv['color'], edgecolor='white',
                                    linewidth=2, alpha=0.85, zorder=2)
    ax.add_patch(rect)
    # Title
    ax.text(lv['x'] + box_w/2, 0.8 + box_h - 0.35, lv['label'],
            ha='center', va='top', fontsize=10, fontweight='bold', color='white', zorder=3)
    # Sublabel
    ax.text(lv['x'] + box_w/2, 0.8 + box_h/2 - 0.2, lv['sublabel'],
            ha='center', va='center', fontsize=8.5, color='white', zorder=3,
            linespacing=1.5)

# Arrows between levels
for i in range(len(levels) - 1):
    x_from = levels[i]['x'] + box_w + 0.05
    x_to = levels[i+1]['x'] - 0.05
    y_mid = 0.8 + box_h / 2
    ax.annotate('', xy=(x_to, y_mid), xytext=(x_from, y_mid),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.0))

# Small filter visualisations above each layer (except input and output)
filter_examples = [
    # Layer 1: horizontal and vertical edge filters
    {'x': 3.2, 'filters': [
        np.array([[1,1,1],[0,0,0],[-1,-1,-1]]),
        np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    ]},
    # Layer 2: corner-like patterns
    {'x': 6.0, 'filters': [
        np.array([[1,1,0],[1,0,-1],[0,-1,-1]]),
        np.array([[0,1,1],[-1,0,1],[-1,-1,0]])
    ]},
]

for fe in filter_examples:
    for fi, filt in enumerate(fe['filters']):
        fx = fe['x'] + 0.15 + fi * 1.1
        fy = 5.35
        cell = 0.28
        for r in range(3):
            for c in range(3):
                val = filt[r, c]
                color = '#ffffff' if val > 0 else ('#222222' if val < 0 else '#888888')
                rect = plt.Rectangle((fx + c * cell, fy - r * cell), cell, cell,
                                      facecolor=color, edgecolor='#aaa', linewidth=0.5, zorder=4)
                ax.add_patch(rect)

ax.text(7.0, 0.35, 'Each layer detects increasingly complex features — built from the features of the layer below',
        ha='center', fontsize=9.5, color='#444', style='italic')

plt.tight_layout(pad=0.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_feature_hierarchy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_feature_hierarchy.png")
