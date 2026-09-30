import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.patch.set_facecolor('#0d1117')

points = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

attempts = [
    ('Attempt 1', np.array([-0.5, 1.5]), np.array([1.5, -0.5]),
     'Misclassifies (1,1)'),
    ('Attempt 2', np.array([0.5, 0.5]), np.array([-0.5, 1.5]),
     'Misclassifies (0,0)'),
    ('Attempt 3', np.array([-0.5, 1.5]), np.array([0.5, 0.5]),
     'Misclassifies both'),
]

for ax, (title, lx, ly, note) in zip(axes, attempts):
    ax.set_facecolor('#0d1117')
    for x1, x2, y in points:
        color = '#f0883e' if y == 1 else '#388bfd'
        marker = 'o' if y == 1 else 's'
        ax.scatter(x1, x2, c=color, marker=marker, s=120, zorder=4,
                   edgecolors='white', linewidths=0.8)
        ax.text(x1 + 0.08, x2 + 0.08, f'({x1},{x2})', color='#8b949e', fontsize=8)

    ax.plot(lx, ly, color='#3fb950', lw=2, linestyle='--')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x₁', color='#8b949e', fontsize=9)
    ax.set_ylabel('x₂', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(title, color='#e6edf3', fontsize=10, pad=6)
    ax.text(0.5, -0.42, note, ha='center', color='#f85149', fontsize=8.5)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0883e',
           markersize=9, label='XOR = 1 (class 1)', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#388bfd',
           markersize=9, label='XOR = 0 (class 0)', linestyle='None'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3',
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('XOR is Not Linearly Separable — No Single Line Works', color='#e6edf3',
             fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'xor_attempts.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
