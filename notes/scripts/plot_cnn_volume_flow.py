import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor('#f8f8f8')
ax.set_facecolor('#f8f8f8')
ax.axis('off')
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.set_title('Volume Flow Through a CNN\n(spatial dimensions shrink, depth grows)',
             fontsize=13, fontweight='bold', color='#222', pad=10)

# Each stage: (x_center, width, height, color, top_label, bottom_label, dim_label)
# Width represents spatial size, height represents depth (channels)
stages = [
    dict(xc=1.2,  w=1.6, h=2.0, color='#95a5a6',
         top='Input',    bot='6×6×1',   dim='H=6\nW=6\nC=1'),
    dict(xc=4.0,  w=1.2, h=2.8, color='#3498db',
         top='Conv1\n+ ReLU', bot='4×4×8',  dim='H=4\nW=4\nC=8'),
    dict(xc=6.6,  w=0.8, h=2.8, color='#2980b9',
         top='MaxPool1', bot='2×2×8',   dim='H=2\nW=2\nC=8'),
    dict(xc=9.0,  w=0.5, h=3.8, color='#e67e22',
         top='Conv2\n+ ReLU', bot='1×1×16', dim='H=1\nW=1\nC=16'),
    dict(xc=11.2, w=0.3, h=3.8, color='#d35400',
         top='MaxPool2', bot='1×1×16',  dim='H=1\nW=1\nC=16'),
    dict(xc=13.2, w=0.25, h=1.2, color='#9b59b6',
         top='Flatten',  bot='16',       dim='16'),
    dict(xc=15.0, w=0.25, h=0.7, color='#27ae60',
         top='FC\n+ Softmax', bot='2',  dim='2'),
]

y_center = 3.0

for s in stages:
    x0 = s['xc'] - s['w'] / 2
    y0 = y_center - s['h'] / 2

    rect = mpatches.FancyBboxPatch((x0, y0), s['w'], s['h'],
                                    boxstyle='square,pad=0',
                                    facecolor=s['color'], edgecolor='white',
                                    linewidth=2.0, alpha=0.85, zorder=2)
    ax.add_patch(rect)

    # Top label (above box)
    ax.text(s['xc'], y0 + s['h'] + 0.25, s['top'],
            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#222')

    # Bottom label (below box)
    ax.text(s['xc'], y0 - 0.25, s['bot'],
            ha='center', va='top', fontsize=9, color='#444')

    # Dimension label inside box (only if tall enough)
    if s['h'] >= 1.0:
        ax.text(s['xc'], y_center, s['dim'],
                ha='center', va='center', fontsize=7.5, color='white',
                fontweight='bold', linespacing=1.4, zorder=3)

# Arrows between stages
for i in range(len(stages) - 1):
    x_from = stages[i]['xc'] + stages[i]['w'] / 2 + 0.08
    x_to   = stages[i + 1]['xc'] - stages[i + 1]['w'] / 2 - 0.08
    ax.annotate('', xy=(x_to, y_center), xytext=(x_from, y_center),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))

# Operation labels on arrows
op_labels = ['3×3 conv\n8 filters', '2×2\nmax pool', '2×2 conv\n16 filters', '2×2\nmax pool', 'reshape', 'W·x+b']
for i, label in enumerate(op_labels):
    xm = (stages[i]['xc'] + stages[i + 1]['xc']) / 2
    ax.text(xm, y_center + 0.45, label,
            ha='center', va='bottom', fontsize=7.5, color='#555', linespacing=1.3)

# Key insight
ax.text(8.0, 0.4,
        'Spatial size shrinks (pooling)   ·   Depth grows (more filters)   ·   Features become more abstract',
        ha='center', fontsize=9.5, color='#333', style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#aaa'))

plt.tight_layout(pad=1.0)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_volume_flow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_volume_flow.png")
