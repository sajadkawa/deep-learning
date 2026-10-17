import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_aspect('equal')
ax.axis('off')

N = 4
angles = [np.pi / 2 + 2 * np.pi * i / N for i in range(N)]
radius = 1.8
positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
labels = ['s₁', 's₂', 's₃', 's₄']

for i in range(N):
    for j in range(i + 1, N):
        x_vals = [positions[i][0], positions[j][0]]
        y_vals = [positions[i][1], positions[j][1]]
        ax.plot(x_vals, y_vals, color='#d1d5db', linewidth=1.2, zorder=1)
        mx = (positions[i][0] + positions[j][0]) / 2
        my = (positions[i][1] + positions[j][1]) / 2
        ax.text(mx, my, f'w{i+1}{j+1}', color='#6b7280', fontsize=8,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=1))

node_radius = 0.28
for i, (x, y) in enumerate(positions):
    circle = plt.Circle((x, y), node_radius, facecolor='#dbeafe',
                         edgecolor='#2563eb', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, labels[i], color='#111111', fontsize=12,
            ha='center', va='center', fontweight='bold', zorder=4)

ax.text(0, -2.6, 'Symmetric weights:  wᵢⱼ = wⱼᵢ', color='#374151',
        fontsize=10, ha='center', va='center')
ax.text(0, -3.0, 'No self-connections:  wᵢᵢ = 0', color='#374151',
        fontsize=10, ha='center', va='center')

ax.set_xlim(-2.8, 2.8)
ax.set_ylim(-3.3, 2.8)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'hopfield_architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved hopfield_architecture.png")
