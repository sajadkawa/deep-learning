import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(15, 5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 5)
ax.axis('off')

C_CONV = '#90CAF9'
C_POOL = '#A5D6A7'
C_FC   = '#FFCC80'
C_OUT  = '#EF9A9A'
C_TEXT = '#212121'

def box(ax, x, y, w, h, color, lines, fontsize=8):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.04', facecolor=color, edgecolor='#455A64', linewidth=1.1)
    ax.add_patch(rect)
    n = len(lines)
    for i, line in enumerate(lines):
        offset = (n-1-i)*0.22 - (n-1)*0.11
        ax.text(x+w/2, y+h/2+offset, line, ha='center', va='center', fontsize=fontsize,
                fontweight='bold' if i == 0 else 'normal', color=C_TEXT)

def arrow(ax, x1, x2, y=2.5):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.4))

box(ax, 0.1, 1.8, 1.0, 1.4, '#E0E0E0', ['Input', '224x224x3'], fontsize=8)
arrow(ax, 1.1, 1.4)

blocks = [
    ('Block 1', '2x Conv 3x3\n64 filters', C_CONV, '112x112x64'),
    ('Block 2', '2x Conv 3x3\n128 filters', C_CONV, '56x56x128'),
    ('Block 3', '3x Conv 3x3\n256 filters', C_CONV, '28x28x256'),
    ('Block 4', '3x Conv 3x3\n512 filters', C_CONV, '14x14x512'),
    ('Block 5', '3x Conv 3x3\n512 filters', C_CONV, '7x7x512'),
]

x = 1.4
for i, (title, sub, color, out_size) in enumerate(blocks):
    box(ax, x, 1.5, 1.5, 2.0, color, [title, sub, '+ MaxPool'], fontsize=7.5)
    ax.text(x+0.75, 1.3, out_size, ha='center', va='top', fontsize=7, color='#546E7A')
    x += 1.5
    if i < len(blocks)-1:
        arrow(ax, x, x+0.2)
        x += 0.2

arrow(ax, x, x+0.3)
x += 0.3
box(ax, x, 1.8, 0.9, 1.4, '#E0E0E0', ['Flatten', '25,088'], fontsize=7.5)
x += 0.9
arrow(ax, x, x+0.2)
x += 0.2

fc_layers = [
    ('FC1', '4,096\n+ReLU+Drop', C_FC),
    ('FC2', '4,096\n+ReLU+Drop', C_FC),
    ('FC3', '1,000\n+Softmax',   C_OUT),
]
for i, (title, sub, color) in enumerate(fc_layers):
    box(ax, x, 1.8, 0.9, 1.4, color, [title, sub], fontsize=7.5)
    x += 0.9
    if i < len(fc_layers)-1:
        arrow(ax, x, x+0.2)
        x += 0.2

ax.text(7.5, 4.7, 'VGG16 Architecture', ha='center', va='center',
        fontsize=13, fontweight='bold', color=C_TEXT)

legend_items = [
    mpatches.Patch(facecolor=C_CONV, edgecolor='#455A64', label='Conv block (3x3 filters + MaxPool)'),
    mpatches.Patch(facecolor=C_FC,   edgecolor='#455A64', label='Fully connected + ReLU + Dropout'),
    mpatches.Patch(facecolor=C_OUT,  edgecolor='#455A64', label='Output (Softmax, 1,000 classes)'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=3,
          fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'vgg16_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved vgg16_architecture.png')
