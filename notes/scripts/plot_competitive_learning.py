import matplotlib.pyplot as plt
import numpy as np
import os

rng = np.random.default_rng(42)

clusters = [
    (1.5, 7.0, '#388bfd', 'o', 'Cluster 1\n(●)'),
    (6.5, 7.5, '#3fb950', 's', 'Cluster 2\n(■)'),
    (3.5, 2.5, '#f0883e', '^', 'Cluster 3\n(▲)'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('#0d1117')

for ax, (title, show_prototypes) in zip(axes, [
    ('Before Competitive Learning\n(random prototypes)', False),
    ('After Competitive Learning\n(prototypes = cluster centroids)', True),
]):
    ax.set_facecolor('#0d1117')

    for cx, cy, color, marker, _ in clusters:
        pts = rng.normal(loc=[cx, cy], scale=0.8, size=(18, 2))
        ax.scatter(pts[:, 0], pts[:, 1], c=color, marker=marker,
                   s=50, alpha=0.7, zorder=3, edgecolors='none')

    if show_prototypes:
        for cx, cy, color, marker, label in clusters:
            ax.scatter(cx, cy, c=color, marker='*', s=350, zorder=5,
                       edgecolors='white', linewidths=0.8)
            ax.text(cx + 0.3, cy + 0.4, 'prototype', color=color, fontsize=8)
    else:
        # Random initial prototypes
        rand_pts = [(2.5, 5.0), (5.0, 3.5), (4.5, 7.0)]
        for (rx, ry), (_, _, color, _, _) in zip(rand_pts, clusters):
            ax.scatter(rx, ry, c=color, marker='*', s=350, zorder=5,
                       edgecolors='white', linewidths=0.8, alpha=0.5)

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Feature 1', color='#8b949e', fontsize=9)
    ax.set_ylabel('Feature 2', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(title, color='#e6edf3', fontsize=10, pad=8)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#388bfd',
           markersize=9, label='Cluster 1', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#3fb950',
           markersize=9, label='Cluster 2', linestyle='None'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#f0883e',
           markersize=9, label='Cluster 3', linestyle='None'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='white',
           markersize=11, label='Prototype (winner neuron)', linestyle='None'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3',
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Competitive Learning — Winner-Takes-All Clustering', color='#e6edf3',
             fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'competitive_learning.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
