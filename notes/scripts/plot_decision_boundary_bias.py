import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('#0d1117')

x = np.linspace(-1.5, 1.5, 200)

for ax, title, lines, note in [
    (axes[0], 'No Bias  (b = 0)',
     [(-1.0, 0.0, '#388bfd'), (-0.5, 0.0, '#3fb950'), (0.5, 0.0, '#f0883e')],
     'All lines pass through origin\n— cannot shift freely'),
    (axes[1], 'With Bias  (b ≠ 0)',
     [(-1.0, 0.8, '#388bfd'), (-0.5, -0.3, '#3fb950'), (0.5, 0.5, '#f0883e')],
     'Lines can shift anywhere\n— full expressive power'),
]:
    ax.set_facecolor('#0d1117')
    ax.axhline(0, color='#30363d', lw=0.8)
    ax.axvline(0, color='#30363d', lw=0.8)

    for slope, intercept, color in lines:
        y = slope * x + intercept
        ax.plot(x, y, color=color, lw=1.8)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x₁', color='#8b949e', fontsize=10)
    ax.set_ylabel('x₂', color='#8b949e', fontsize=10)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(title, color='#e6edf3', fontsize=11, pad=8)
    ax.text(0, -1.35, note, ha='center', va='bottom', color='#8b949e', fontsize=8.5)

fig.suptitle('Decision Boundary: Effect of the Bias Term', color='#e6edf3', fontsize=13, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'decision_boundary_bias.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
