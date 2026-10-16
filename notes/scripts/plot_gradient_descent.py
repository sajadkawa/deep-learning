import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')
    ax.tick_params(colors='#374151', labelsize=8)
    ax.xaxis.label.set_color('#374151')
    ax.yaxis.label.set_color('#374151')

w = np.linspace(-2.5, 2.5, 400)
L = w ** 2

def run_gd(w0, eta, steps):
    pts = [w0]
    for _ in range(steps):
        pts.append(pts[-1] - eta * 2 * pts[-1])
    return np.array(pts)

# ── Panel 1: good convergence ─────────────────────────────────────────────────
ax = axes[0]
ax.plot(w, L, color='#2563eb', linewidth=2)

pts_w = run_gd(1.8, 0.35, 6)
pts_L = pts_w ** 2

for i in range(len(pts_w) - 1):
    ax.annotate('', xy=(pts_w[i+1], pts_L[i+1]), xytext=(pts_w[i], pts_L[i]),
                arrowprops=dict(arrowstyle='->', color='#15803d', lw=1.8,
                                connectionstyle='arc3,rad=-0.25'))
ax.plot(pts_w, pts_L, 'o', color='#15803d', markersize=6, zorder=6)
for i, (pw, pl) in enumerate(zip(pts_w, pts_L)):
    ax.text(pw + 0.08, pl + 0.15, f'w{i}', color='#374151', fontsize=7)

ax.text(0.0, 5.8, 'w ← w − η·∂L/∂w', color='#ca8a04', fontsize=8,
        ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef9c3', edgecolor='#ca8a04'))
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.3, 7)
ax.set_xlabel('w', fontsize=9); ax.set_ylabel('Loss L', fontsize=9)
ax.set_title('η just right — converges', color='#111111', fontsize=9, pad=8)
ax.legend(handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor='#15803d',
          markersize=7, label='η = 0.35')],
          fontsize=7, facecolor='white', edgecolor='#d1d5db', labelcolor='#374151')

# ── Panel 2: η too large — overshoots ────────────────────────────────────────
ax = axes[1]
ax.plot(w, L, color='#2563eb', linewidth=2)

pts_w = run_gd(1.8, 0.9, 5)
pts_L = pts_w ** 2

for i in range(len(pts_w) - 1):
    ax.annotate('', xy=(pts_w[i+1], pts_L[i+1]), xytext=(pts_w[i], pts_L[i]),
                arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.8,
                                connectionstyle='arc3,rad=0.3'))
ax.plot(pts_w, pts_L, 'o', color='#b45309', markersize=7, zorder=6)
for i, (pw, pl) in enumerate(zip(pts_w, pts_L)):
    ax.text(pw + 0.08, pl + 0.15, f'w{i}', color='#374151', fontsize=7)

ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.3, 7)
ax.set_xlabel('w', fontsize=9); ax.set_ylabel('Loss L', fontsize=9)
ax.set_title('η too large — overshoots', color='#111111', fontsize=9, pad=8)
ax.legend(handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor='#b45309',
          markersize=7, label='η = 0.9')],
          fontsize=7, facecolor='white', edgecolor='#d1d5db', labelcolor='#374151')

# ── Panel 3: η too small — slow ───────────────────────────────────────────────
ax = axes[2]
ax.plot(w, L, color='#2563eb', linewidth=2)

pts_w = run_gd(1.8, 0.05, 20)
pts_L = pts_w ** 2

ax.plot(pts_w, pts_L, 'o-', color='#dc2626', markersize=4, linewidth=1.2, zorder=6)
ax.plot(pts_w[0],  pts_L[0],  'o', color='#dc2626', markersize=8, zorder=7)
ax.plot(pts_w[-1], pts_L[-1], 'o', color='#dc2626', markersize=8, zorder=7)
ax.text(pts_w[0] + 0.1, pts_L[0] + 0.2, 'start', color='#374151', fontsize=7)
ax.text(pts_w[-1] + 0.1, pts_L[-1] + 0.2, 'still far', color='#374151', fontsize=7)

ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.3, 7)
ax.set_xlabel('w', fontsize=9); ax.set_ylabel('Loss L', fontsize=9)
ax.set_title('η too small — very slow', color='#111111', fontsize=9, pad=8)
ax.legend(handles=[Line2D([0],[0], marker='o', color='w', markerfacecolor='#dc2626',
          markersize=7, label='η = 0.05')],
          fontsize=7, facecolor='white', edgecolor='#d1d5db', labelcolor='#374151')

plt.tight_layout(pad=1.5)
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'gradient_descent.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved gradient_descent.png")
