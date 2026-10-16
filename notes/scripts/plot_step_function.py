import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(5, 3))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

z = np.linspace(-3, 3, 1000)
f = np.where(z >= 0, 1.0, 0.0)

ax.plot(z[z < 0],  f[z < 0],  color='#1d4ed8', lw=2.5)
ax.plot(z[z >= 0], f[z >= 0], color='#1d4ed8', lw=2.5)
ax.plot(0, 1, 'o', color='#1d4ed8', markersize=7, zorder=5)
ax.plot(0, 0, 'o', color='white', markersize=7,
        markeredgecolor='#1d4ed8', markeredgewidth=2, zorder=5)

ax.axhline(0, color='#e5e7eb', lw=0.8)
ax.axvline(0, color='#e5e7eb', lw=0.8)

ax.set_xlim(-3, 3)
ax.set_ylim(-0.3, 1.4)
ax.set_xlabel('z  (weighted sum)', color='#374151', fontsize=10)
ax.set_ylabel('f(z)', color='#374151', fontsize=10)
ax.tick_params(colors='#374151')
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')

ax.text(-1.5, 0.15, 'z < 0  →  f(z) = 0', color='#374151', fontsize=9, fontweight='bold')
ax.text(0.2,  0.85, 'z ≥ 0  →  f(z) = 1', color='#374151', fontsize=9, fontweight='bold')

ax.set_title('Step Function (Heaviside)', color='#111111', fontsize=12, fontweight='bold', pad=8)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'step_function.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved step_function.png")
