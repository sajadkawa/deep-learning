import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#1e1e1e')

d = np.linspace(0, 5, 300)

sigmas = [2.0, 0.5]
colors = ['#4fc3f7', '#ce93d8']
labels = ['σ = 2.0  (ordering phase — wide neighbourhood)',
          'σ = 0.5  (convergence phase — narrow neighbourhood)']

for sigma, color, label in zip(sigmas, colors, labels):
    h = np.exp(-d**2 / (2 * sigma**2))
    ax.plot(d, h, color=color, linewidth=2.2, label=label)

# Integer grid distance markers
for di in range(6):
    ax.axvline(di, color='#444444', linewidth=0.6, linestyle='--')

ax.set_xlabel('Grid distance d from BMU', color='#cccccc', fontsize=11)
ax.set_ylabel('h(d)  — neighbourhood strength', color='#cccccc', fontsize=11)
ax.tick_params(colors='#aaaaaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#555555')

ax.set_xlim(0, 5)
ax.set_ylim(-0.02, 1.08)
ax.set_xticks(range(6))
ax.set_xticklabels([f'd={i}' for i in range(6)], color='#aaaaaa', fontsize=9)

ax.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='#555555', labelcolor='#eeeeee')

ax.text(0.02, 0.97, 'h(d) = exp(−d² / 2σ²)', transform=ax.transAxes,
        fontsize=10, color='#aaaaaa', va='top', style='italic')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'som_neighbourhood.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved som_neighbourhood.png")
