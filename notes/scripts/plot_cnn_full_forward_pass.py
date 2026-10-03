import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor('#f8f8f8')
ax.set_facecolor('#f8f8f8')
ax.axis('off')
ax.set_xlim(0, 18)
ax.set_ylim(0, 8)
ax.set_title('Full CNN Forward Pass — Worked Example', fontsize=13, fontweight='bold', color='#222', pad=10)

y_center = 5.0   # centre of all boxes

stages = [
    dict(x=0.3,  w=2.0, h=4.5, color='#7f8c8d',
         label='Input\n6×6×1',
         sub='6×6 image\n1 channel',
         detail=''),
    dict(x=3.1,  w=2.0, h=3.8, color='#2980b9',
         label='Conv\n4×4×1',
         sub='3×3 filter\nstride=1, pad=0',
         detail='(6−3)/1+1 = 4'),
    dict(x=5.9,  w=2.0, h=3.8, color='#27ae60',
         label='ReLU\n4×4×1',
         sub='max(0, z)',
         detail='all values ≥ 0\nno change'),
    dict(x=8.7,  w=2.0, h=3.0, color='#e67e22',
         label='MaxPool\n2×2×1',
         sub='2×2 pool\nstride=2',
         detail='4/2 = 2'),
    dict(x=11.5, w=1.6, h=2.0, color='#8e44ad',
         label='Flatten\n4×1',
         sub='2×2 → 4',
         detail='[6, 7, 5, 6]'),
    dict(x=13.9, w=1.8, h=3.0, color='#c0392b',
         label='FC\n2×1',
         sub='4 → 2 neurons',
         detail='z = [7.0, 6.0]'),
    dict(x=16.5, w=1.2, h=3.0, color='#16a085',
         label='Softmax\n2×1',
         sub='probabilities',
         detail='[0.73, 0.27]'),
]

for s in stages:
    y0 = y_center - s['h'] / 2

    # Box
    rect = mpatches.FancyBboxPatch((s['x'], y0), s['w'], s['h'],
                                    boxstyle='round,pad=0.1',
                                    facecolor=s['color'], edgecolor='white',
                                    linewidth=2.0, alpha=0.88, zorder=2)
    ax.add_patch(rect)

    # Layer name — top of box
    ax.text(s['x'] + s['w'] / 2, y0 + s['h'] - 0.3,
            s['label'], ha='center', va='top',
            fontsize=9.5, fontweight='bold', color='white', zorder=3, linespacing=1.3)

    # Sub-label — middle of box
    ax.text(s['x'] + s['w'] / 2, y_center,
            s['sub'], ha='center', va='center',
            fontsize=8.5, color='#f0f0f0', zorder=3, linespacing=1.4)

    # Detail — bottom of box (only if box is tall enough)
    if s['detail'] and s['h'] >= 2.5:
        ax.text(s['x'] + s['w'] / 2, y0 + 0.35,
                s['detail'], ha='center', va='bottom',
                fontsize=8, color='#ffffaa', zorder=3, linespacing=1.3)

# Arrows between stages
for i in range(len(stages) - 1):
    x_from = stages[i]['x'] + stages[i]['w'] + 0.06
    x_to   = stages[i + 1]['x'] - 0.06
    ax.annotate('', xy=(x_to, y_center), xytext=(x_from, y_center),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2.0))

# Parameter count — below each box, with enough clearance
param_data = [
    (0,  '—'),
    (1,  '9 weights\n+ 1 bias\n= 10 params'),
    (2,  '0 params'),
    (3,  '0 params'),
    (4,  '0 params'),
    (5,  '4×2 weights\n+ 2 biases\n= 10 params'),
    (6,  '0 params'),
]

for idx, txt in param_data:
    s = stages[idx]
    y_note = y_center - s['h'] / 2 - 0.25
    ax.text(s['x'] + s['w'] / 2, y_note, txt,
            ha='center', va='top', fontsize=8, color='#555', linespacing=1.3)

# Total params box at the very bottom
ax.text(9.0, 0.55,
        'Total trainable parameters:  10 (conv layer)  +  10 (FC layer)  =  20',
        ha='center', va='center', fontsize=10, color='#222',
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#f0f0f0', edgecolor='#888', linewidth=1.5))

plt.tight_layout(pad=1.0)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_full_forward_pass.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_full_forward_pass.png")
