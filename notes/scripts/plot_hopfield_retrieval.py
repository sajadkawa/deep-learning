import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(11, 4), facecolor='#0d1117')
fig.patch.set_facecolor('#0d1117')

titles = ['Noisy input', 'Updating...', 'Recovered pattern']
states = [
    [-1, -1, +1, -1],   # corrupted x1 (s1 flipped)
    [+1, -1, +1, -1],   # after fixing s1
    [+1, -1, +1, -1],   # x1 recovered
]
notes = [
    's1 = -1  <- wrong',
    's1 -> +1  (net1 = 0.5 > 0)',
    'x1 = (+1,-1,+1,-1)  recovered',
]
colors_on  = '#1f6feb'
colors_off = '#30363d'
highlight  = '#f78166'

for col, ax in enumerate(axes):
    ax.set_facecolor('#0d1117')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.2, 2.0)
    ax.axis('off')
    ax.set_title(titles[col], color='#e6edf3', fontsize=11, pad=8)

    for i, val in enumerate(states[col]):
        # Highlight the neuron being corrected in the middle panel
        if col == 1 and i == 0:
            ec = '#f78166'
            lw = 2.5
        else:
            ec = 'none'
            lw = 0
        fc = colors_on if val == +1 else colors_off
        circle = plt.Circle((i * 1.0, 0.8), 0.35, color=fc,
                             ec=ec, linewidth=lw, zorder=3)
        ax.add_patch(circle)
        label = '+1' if val == +1 else '-1'
        ax.text(i * 1.0, 0.8, label, color='#e6edf3', fontsize=10,
                ha='center', va='center', fontweight='bold', zorder=4)
        ax.text(i * 1.0, 0.25, 's' + str(i+1), color='#8b949e', fontsize=9,
                ha='center', va='center')

    ax.text(1.5, -0.55, notes[col], color='#8b949e', fontsize=9,
            ha='center', va='center')

# Arrows between panels
for x in [3.85, 7.7]:
    fig.text(x / 11.0, 0.54, '→', color='#8b949e', fontsize=20,
             ha='center', va='center', transform=fig.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.suptitle('Hopfield retrieval — energy descent from noisy input to stored pattern',
             color='#e6edf3', fontsize=11, y=0.98)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'hopfield_retrieval.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Saved hopfield_retrieval.png")
