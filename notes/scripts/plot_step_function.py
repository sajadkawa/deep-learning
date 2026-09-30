import matplotlib.pyplot as plt
import numpy as np
import os

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

z = np.linspace(-3, 3, 1000)
f = np.where(z >= 0, 1.0, 0.0)

ax.plot(z[z < 0], f[z < 0], color='#388bfd', lw=2.5)
ax.plot(z[z >= 0], f[z >= 0], color='#388bfd', lw=2.5)
ax.plot(0, 1, 'o', color='#388bfd', markersize=7, zorder=5)
ax.plot(0, 0, 'o', color='#0d1117', markersize=7, markeredgecolor='#388bfd',
        markeredgewidth=2, zorder=5)

ax.axhline(0, color='#30363d', lw=0.8)
ax.axvline(0, color='#30363d', lw=0.8)

ax.set_xlim(-3, 3)
ax.set_ylim(-0.3, 1.4)
ax.set_xlabel('z  (weighted sum)', color='#8b949e', fontsize=10)
ax.set_ylabel('f(z)', color='#8b949e', fontsize=10)
ax.tick_params(colors='#8b949e')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')

ax.text(-1.5, 0.15, 'z < 0  →  f(z) = 0', color='#8b949e', fontsize=9)
ax.text(0.2, 0.85, 'z ≥ 0  →  f(z) = 1', color='#8b949e', fontsize=9)

ax.set_title('Step Function (Heaviside)', color='#e6edf3', fontsize=12, pad=8)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'step_function.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
