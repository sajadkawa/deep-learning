import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
fig.patch.set_facecolor('white')

titles = ['Noisy input', 'Updating...', 'Recovered pattern']
states = [
    [-1, -1, +1, -1],
    [+1, -1, +1, -1],
    [+1, -1, +1, -1],
]
notes = [
    's1 = -1  ← wrong',
    's1 → +1  (net1 = 0.5 > 0)',
    'x1 = (+1,-1,+1,-1)  recovered',
]

for col, ax in enumerate(axes):
    ax.set_facecolor('white')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.2, 2.0)
    ax.axis('off')
    ax.set_title(titles[col], color='#111111', fontsize=11, pad=8)

    for i, val in enumerate(states[col]):
        if col == 1 and i == 0:
            ec = '#dc2626'
            lw = 2.5
        else:
            ec = '#2563eb'
            lw = 1.5
        fc = '#dbeafe' if val == +1 else '#f3f4f6'
        circle = plt.Circle((i * 1.0, 0.8), 0.35, facecolor=fc,
                             edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(circle)
        label = '+1' if val == +1 else '-1'
        ax.text(i * 1.0, 0.8, label, color='#111111', fontsize=10,
                ha='center', va='center', fontweight='bold', zorder=4)
        ax.text(i * 1.0, 0.25, 's' + str(i+1), color='#374151', fontsize=9,
                ha='center', va='center')

    ax.text(1.5, -0.55, notes[col], color='#374151', fontsize=9,
            ha='center', va='center')

for x in [3.85, 7.7]:
    fig.text(x / 11.0, 0.54, '→', color='#9ca3af', fontsize=20,
             ha='center', va='center', transform=fig.transFigure)

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.suptitle('Hopfield retrieval — energy descent from noisy input to stored pattern',
             color='#111111', fontsize=11, y=0.98)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'hopfield_retrieval.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved hopfield_retrieval.png")
