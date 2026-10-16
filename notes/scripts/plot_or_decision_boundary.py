import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

points = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
for x1, x2, y in points:
    color  = '#1d4ed8' if y == 1 else '#b91c1c'
    marker = 'o'       if y == 1 else 'X'
    ax.scatter(x1, x2, c=color, marker=marker, s=180, zorder=4,
               edgecolors='#374151', linewidths=0.8)
    ax.text(x1 + 0.06, x2 + 0.07, f'({x1},{x2})', color='#111111', fontsize=11)

x_line = np.linspace(-0.3, 1.4, 200)
y_line = 1 - x_line
ax.plot(x_line, y_line, color='#16a34a', lw=2, linestyle='--', zorder=3)
ax.text(0.78, 0.52, 'x₁ + x₂ = 1\n(decision boundary)', color='#16a34a',
        fontsize=9, rotation=-45, ha='center')

ax.set_xlim(-0.4, 1.6)
ax.set_ylim(-0.4, 1.6)
ax.set_xlabel('x₁', color='#374151', fontsize=11)
ax.set_ylabel('x₂', color='#374151', fontsize=11)
ax.tick_params(colors='#374151')
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1d4ed8',
           markeredgecolor='#374151', markersize=10, label='OR = 1', linestyle='None'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='#b91c1c',
           markeredgecolor='#374151', markersize=10, label='OR = 0', linestyle='None'),
]
ax.legend(handles=legend_elements, facecolor='white', edgecolor='#d1d5db',
          labelcolor='#111111', fontsize=9, loc='upper left')

ax.set_title('OR — Learned Decision Boundary', color='#111111', fontsize=12, pad=8)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'or_decision_boundary.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
