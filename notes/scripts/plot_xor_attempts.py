import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
fig.patch.set_facecolor('white')

points = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

attempts = [
    ('Attempt 1', np.array([-0.5, 1.5]), np.array([1.5, -0.5]), 'Misclassifies (1,1)'),
    ('Attempt 2', np.array([0.5, 0.5]),  np.array([-0.5, 1.5]), 'Misclassifies (0,0)'),
    ('Attempt 3', np.array([-0.5, 1.5]), np.array([0.5, 0.5]),  'Misclassifies both'),
]

for ax, (title, lx, ly, note) in zip(axes, attempts):
    ax.set_facecolor('white')
    for x1, x2, y in points:
        color  = '#1d4ed8' if y == 1 else '#b91c1c'
        marker = 'o'       if y == 1 else 'X'
        ax.scatter(x1, x2, c=color, marker=marker, s=120, zorder=4,
                   edgecolors='#374151', linewidths=0.8)
        ax.text(x1 + 0.08, x2 + 0.08, f'({x1},{x2})', color='#374151', fontsize=8)

    ax.plot(lx, ly, color='#16a34a', lw=2, linestyle='--')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x₁', color='#374151', fontsize=9)
    ax.set_ylabel('x₂', color='#374151', fontsize=9)
    ax.tick_params(colors='#374151')
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')
    ax.set_title(title, color='#111111', fontsize=10, pad=6)
    ax.text(0.5, -0.42, note, ha='center', color='#dc2626', fontsize=8.5)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1d4ed8',
           markeredgecolor='#374151', markersize=9, label='XOR = 1', linestyle='None'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#b91c1c',
           markeredgecolor='#374151', markersize=9, label='XOR = 0', linestyle='None'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           facecolor='white', edgecolor='#d1d5db', labelcolor='#111111',
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('XOR is Not Linearly Separable — No Single Line Works',
             color='#111111', fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'xor_attempts.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
