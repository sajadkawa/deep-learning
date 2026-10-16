import os
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 500)
energy = 0.5 * (x**2 - 2)**2 - 0.3 * x

fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(x, energy, color='#2563eb', linewidth=2)

minima_x = [x[np.argmin(energy[:250])], x[250 + np.argmin(energy[250:])]]
for mx in minima_x:
    my = 0.5 * (mx**2 - 2)**2 - 0.3 * mx
    ax.plot(mx, my, 'o', color='#15803d', markersize=8, zorder=5)
    ax.annotate('Low energy\n(stable state)', xy=(mx, my),
                xytext=(mx + 0.3, my + 1.2),
                fontsize=8, color='#15803d',
                arrowprops=dict(arrowstyle='->', color='#15803d', lw=1))

ax.set_xlabel('Network State', color='#374151')
ax.set_ylabel('Energy', color='#374151')
ax.tick_params(colors='#374151')
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Hopfield Network Energy Landscape', color='#111111', fontsize=11)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'energy_landscape.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
