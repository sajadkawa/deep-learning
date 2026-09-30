import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

points = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
for x1, x2, y in points:
    color  = '#f0883e' if y == 1 else '#388bfd'
    marker = 'o'       if y == 1 else 's'
    ax.scatter(x1, x2, c=color, marker=marker, s=180, zorder=4,
               edgecolors='white', linewidths=0.8)
    label = '●' if y == 1 else '○'
    ax.text(x1 + 0.06, x2 + 0.07, f'{label}({x1},{x2})',
            color='#e6edf3', fontsize=11)

ax.set_xlim(-0.4, 1.6)
ax.set_ylim(-0.4, 1.6)
ax.set_xlabel('x₁', color='#8b949e', fontsize=11)
ax.set_ylabel('x₂', color='#8b949e', fontsize=11)
ax.tick_params(colors='#8b949e')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0883e',
           markersize=10, label='● class 1  (XOR = 1)', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#388bfd',
           markersize=10, label='○ class 0  (XOR = 0)', linestyle='None'),
]
ax.legend(handles=legend_elements, facecolor='#161b22', edgecolor='#30363d',
          labelcolor='#e6edf3', fontsize=9, loc='upper left')

ax.set_title('XOR — Coordinate Plane', color='#e6edf3', fontsize=12, pad=8)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'xor_coordinate_plane.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
