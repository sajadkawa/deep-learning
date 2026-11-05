import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('white')

# ── Panel 1: Constraint regions ────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('white')

# Loss contours — ellipse centred away from origin (the unconstrained minimum)
w1 = np.linspace(-1.6, 1.6, 400)
w2 = np.linspace(-1.6, 1.6, 400)
W1, W2 = np.meshgrid(w1, w2)
# Unconstrained minimum at (1.1, 0.6)
L = 2.5 * (W1 - 1.1)**2 + 4.0 * (W2 - 0.6)**2
ax.contour(W1, W2, L, levels=[0.3, 0.7, 1.2, 1.9, 2.8],
           colors='#9ca3af', linewidths=1.0, linestyles='--')
ax.plot(1.1, 0.6, 'o', color='#6b7280', markersize=5, zorder=5)
ax.text(1.15, 0.65, 'unconstrained\nminimum', fontsize=7.5, color='#6b7280')

# L2 constraint — circle of radius r
r = 0.72
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(r * np.cos(theta), r * np.sin(theta),
        color='#2563eb', linewidth=2.0, label='L2 constraint (circle)')
ax.fill(r * np.cos(theta), r * np.sin(theta),
        color='#2563eb', alpha=0.06)

# L2 solution — exact tangency point where loss contour touches circle
ax.plot(0.6017, 0.3954, 'o', color='#2563eb', markersize=7, zorder=6)
ax.annotate('L2 solution\n(non-zero weights)',
            xy=(0.6017, 0.3954), xytext=(0.75, -0.55),
            fontsize=8, color='#2563eb',
            arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.2))

# L1 constraint — diamond of radius r
d = 0.72
diamond_x = [d, 0, -d, 0, d]
diamond_y = [0, d, 0, -d, 0]
ax.plot(diamond_x, diamond_y,
        color='#dc2626', linewidth=2.0, label='L1 constraint (diamond)')
ax.fill(diamond_x, diamond_y, color='#dc2626', alpha=0.06)

# L1 solution — corner closest to unconstrained minimum (0.72, 0): w2=0, sparse
ax.plot(0.72, 0.0, 'o', color='#dc2626', markersize=7, zorder=6)
ax.annotate('L1 solution\n(w₂ = 0, sparse)',
            xy=(0.72, 0.0), xytext=(0.6, -0.9),
            fontsize=8, color='#dc2626',
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.2))

ax.axhline(0, color='#d1d5db', linewidth=0.8)
ax.axvline(0, color='#d1d5db', linewidth=0.8)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel('w₁', color='#374151', fontsize=10)
ax.set_ylabel('w₂', color='#374151', fontsize=10)
ax.set_title('Constraint Regions', color='#111111', fontsize=11, pad=10)
ax.tick_params(colors='#374151', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')
ax.legend(fontsize=8.5, facecolor='white', edgecolor='#d1d5db',
          labelcolor='#111111', loc='lower right')
ax.set_aspect('equal')

# ── Panel 2: Penalty shape ─────────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('white')

w = np.linspace(-2, 2, 400)
l2_penalty = w ** 2
l1_penalty = np.abs(w)

ax.plot(w, l2_penalty, color='#2563eb', linewidth=2.5, label='L2 penalty:  w²')
ax.plot(w, l1_penalty, color='#dc2626', linewidth=2.5, label='L1 penalty:  |w|')

# Gradient annotations
ax.annotate('', xy=(1.4, 1.4 ** 2 + 0.15), xytext=(1.1, 1.1 ** 2 + 0.15),
            arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.4))
ax.text(1.45, 2.1, 'gradient = 2w\n(grows with w)', fontsize=8, color='#2563eb')

ax.annotate('', xy=(-1.4, 1.4 + 0.15), xytext=(-1.1, 1.1 + 0.15),
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.4))
ax.text(-2.0, 1.85, 'gradient = ±1\n(constant)', fontsize=8, color='#dc2626')

ax.axvline(0, color='#d1d5db', linewidth=0.8)
ax.axhline(0, color='#d1d5db', linewidth=0.8)
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-0.2, 4.2)
ax.set_xlabel('weight value  w', color='#374151', fontsize=10)
ax.set_ylabel('penalty', color='#374151', fontsize=10)
ax.set_title('Penalty Shape', color='#111111', fontsize=11, pad=10)
ax.tick_params(colors='#374151', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9, facecolor='white', edgecolor='#d1d5db', labelcolor='#111111')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'l1_l2_regularisation.png'),
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
