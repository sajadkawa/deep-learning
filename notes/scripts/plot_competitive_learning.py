import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

rng = np.random.default_rng(42)

clusters = [
    (1.5, 7.0, '#dbeafe', '#2563eb', 'o', 'Cluster 1'),
    (6.5, 7.5, '#dcfce7', '#15803d', 's', 'Cluster 2'),
    (3.5, 2.5, '#fef3c7', '#b45309', '^', 'Cluster 3'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')

for ax, (title, show_prototypes) in zip(axes, [
    ('Before Competitive Learning\n(random prototypes)', False),
    ('After Competitive Learning\n(prototypes = cluster centroids)', True),
]):
    ax.set_facecolor('white')

    for cx, cy, fill, border, marker, _ in clusters:
        pts = rng.normal(loc=[cx, cy], scale=0.8, size=(18, 2))
        ax.scatter(pts[:, 0], pts[:, 1], c=fill, marker=marker,
                   s=60, alpha=0.9, zorder=3, edgecolors=border, linewidths=0.8)

    if show_prototypes:
        for cx, cy, fill, border, marker, label in clusters:
            ax.scatter(cx, cy, c=fill, marker='*', s=350, zorder=5,
                       edgecolors=border, linewidths=1.2)
            ax.text(cx + 0.3, cy + 0.4, 'prototype', color=border, fontsize=8)
    else:
        rand_pts = [(2.5, 5.0), (5.0, 3.5), (4.5, 7.0)]
        for (rx, ry), (_, _, fill, border, _, _) in zip(rand_pts, clusters):
            ax.scatter(rx, ry, c=fill, marker='*', s=350, zorder=5,
                       edgecolors=border, linewidths=1.2, alpha=0.6)

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Feature 1', color='#374151', fontsize=9)
    ax.set_ylabel('Feature 2', color='#374151', fontsize=9)
    ax.tick_params(colors='#374151')
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')
    ax.set_title(title, color='#111111', fontsize=10, pad=8)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#dbeafe',
           markeredgecolor='#2563eb', markersize=9, label='Cluster 1', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#dcfce7',
           markeredgecolor='#15803d', markersize=9, label='Cluster 2', linestyle='None'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#fef3c7',
           markeredgecolor='#b45309', markersize=9, label='Cluster 3', linestyle='None'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#f3f4f6',
           markeredgecolor='#374151', markersize=11, label='Prototype (winner neuron)', linestyle='None'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           facecolor='white', edgecolor='#d1d5db', labelcolor='#111111',
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Competitive Learning — Winner-Takes-All Clustering', color='#111111',
             fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'competitive_learning.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
