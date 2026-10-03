import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0d1117')
ax.set_facecolor('#0d1117')
ax.set_aspect('equal')
ax.axis('off')

N = 4
angles = [np.pi / 2 + 2 * np.pi * i / N for i in range(N)]
radius = 1.8
positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
labels = ['s₁', 's₂', 's₃', 's₄']

# Draw connections (all pairs, both directions shown as one line)
for i in range(N):
    for j in range(i + 1, N):
        x_vals = [positions[i][0], positions[j][0]]
        y_vals = [positions[i][1], positions[j][1]]
        ax.plot(x_vals, y_vals, color='#8b949e', linewidth=1.2, zorder=1)
        # Label the weight at midpoint
        mx = (positions[i][0] + positions[j][0]) / 2
        my = (positions[i][1] + positions[j][1]) / 2
        ax.text(mx, my, f'w{i+1}{j+1}', color='#8b949e', fontsize=8,
                ha='center', va='center',
                bbox=dict(facecolor='#0d1117', edgecolor='none', pad=1))

# Draw neurons
node_radius = 0.28
for i, (x, y) in enumerate(positions):
    circle = plt.Circle((x, y), node_radius, color='#1f6feb', zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, labels[i], color='#e6edf3', fontsize=12,
            ha='center', va='center', fontweight='bold', zorder=4)

# Annotations
ax.text(0, -2.6, 'Symmetric weights:  wᵢⱼ = wⱼᵢ', color='#8b949e',
        fontsize=10, ha='center', va='center')
ax.text(0, -3.0, 'No self-connections:  wᵢᵢ = 0', color='#8b949e',
        fontsize=10, ha='center', va='center')

ax.set_xlim(-2.8, 2.8)
ax.set_ylim(-3.3, 2.8)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'hopfield_architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Saved hopfield_architecture.png")
