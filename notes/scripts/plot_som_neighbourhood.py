import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(6, 3.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

d = np.linspace(0, 5, 300)

sigmas = [2.0, 0.5]
colors = ['#2563eb', '#dc2626']
labels = ['σ = 2.0  (ordering phase — wide neighbourhood)',
          'σ = 0.5  (convergence phase — narrow neighbourhood)']

for sigma, color, label in zip(sigmas, colors, labels):
    h = np.exp(-d**2 / (2 * sigma**2))
    ax.plot(d, h, color=color, linewidth=2.2, label=label)

for di in range(6):
    ax.axvline(di, color='#e5e7eb', linewidth=0.6, linestyle='--')

ax.set_xlabel('Grid distance d from BMU', color='#374151', fontsize=11)
ax.set_ylabel('h(d)  — neighbourhood strength', color='#374151', fontsize=11)
ax.tick_params(colors='#374151')
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')

ax.set_xlim(0, 5)
ax.set_ylim(-0.02, 1.08)
ax.set_xticks(range(6))
ax.set_xticklabels([f'd={i}' for i in range(6)], color='#374151', fontsize=9)

ax.legend(fontsize=9, facecolor='white', edgecolor='#d1d5db', labelcolor='#111111')

ax.text(0.02, 0.97, 'h(d) = exp(−d² / 2σ²)', transform=ax.transAxes,
        fontsize=10, color='#6b7280', va='top', style='italic')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'som_neighbourhood.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved som_neighbourhood.png")
