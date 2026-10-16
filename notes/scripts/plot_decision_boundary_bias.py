import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('white')

x = np.linspace(-1.5, 1.5, 200)

for ax, title, lines, note in [
    (axes[0], 'No Bias  (b = 0)',
     [(-1.0, 0.0, '#1d4ed8'), (-0.5, 0.0, '#15803d'), (0.5, 0.0, '#c2410c')],
     'All lines pass through origin\n— cannot shift freely'),
    (axes[1], 'With Bias  (b ≠ 0)',
     [(-1.0, 0.8, '#1d4ed8'), (-0.5, -0.3, '#15803d'), (0.5, 0.5, '#c2410c')],
     'Lines can shift anywhere\n— full expressive power'),
]:
    ax.set_facecolor('white')
    ax.axhline(0, color='#e5e7eb', lw=0.8)
    ax.axvline(0, color='#e5e7eb', lw=0.8)

    for slope, intercept, color in lines:
        y = slope * x + intercept
        ax.plot(x, y, color=color, lw=2)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x₁', color='#374151', fontsize=10)
    ax.set_ylabel('x₂', color='#374151', fontsize=10)
    ax.tick_params(colors='#374151')
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')
    ax.set_title(title, color='#111111', fontsize=11, fontweight='bold', pad=8)
    ax.text(0, -1.35, note, ha='center', va='bottom', color='#6b7280',
            fontsize=8.5, fontweight='bold')

fig.suptitle('Decision Boundary: Effect of the Bias Term',
             color='#111111', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'decision_boundary_bias.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved decision_boundary_bias.png")
