import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#0d1117')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.xaxis.label.set_color('#aaa')
    ax.yaxis.label.set_color('#aaa')

w = np.linspace(-3, 3, 400)
L = w ** 2  # simple parabola: minimum at w=0

# ── Panel 1: one gradient descent step ──────────────────────────────────────
ax = axes[0]
ax.plot(w, L, color='#58a6ff', linewidth=2)

w0 = 2.2
L0 = w0 ** 2
grad0 = 2 * w0          # dL/dw = 2w
eta = 0.4
w1 = w0 - eta * grad0
L1 = w1 ** 2

ax.plot(w0, L0, 'o', color='#f78166', markersize=9, zorder=5, label='current position')
ax.plot(w1, L1, 'o', color='#3fb950', markersize=9, zorder=5, label='after one step')
ax.annotate('', xy=(w1, L1 + 0.3), xytext=(w0, L0 + 0.3),
            arrowprops=dict(arrowstyle='->', color='#e3b341', lw=1.8))
ax.text((w0 + w1) / 2, L0 + 0.7, 'w ← w − η·∂L/∂w', color='#e3b341',
        fontsize=8, ha='center')
ax.set_xlabel('w', fontsize=9)
ax.set_ylabel('Loss L', fontsize=9)
ax.set_title('One gradient descent step', color='#e6edf3', fontsize=9, pad=8)
ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#444', labelcolor='#aaa')

# ── Panel 2: learning rate too large ────────────────────────────────────────
ax = axes[1]
ax.plot(w, L, color='#58a6ff', linewidth=2)

eta_large = 1.8
pts = [2.2]
for _ in range(5):
    pts.append(pts[-1] - eta_large * 2 * pts[-1])
pts_w = np.array(pts)
pts_L = pts_w ** 2

ax.plot(pts_w, pts_L, 'o-', color='#f78166', markersize=7, linewidth=1.4,
        label='η = 1.8 (too large)')
for i in range(len(pts_w) - 1):
    ax.annotate('', xy=(pts_w[i + 1], pts_L[i + 1]),
                xytext=(pts_w[i], pts_L[i]),
                arrowprops=dict(arrowstyle='->', color='#f78166', lw=1.2))
ax.set_xlabel('w', fontsize=9)
ax.set_ylabel('Loss L', fontsize=9)
ax.set_title('η too large — overshoots', color='#e6edf3', fontsize=9, pad=8)
ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#444', labelcolor='#aaa')
ax.set_ylim(-0.5, 10)

# ── Panel 3: η too small vs just right ──────────────────────────────────────
ax = axes[2]
ax.plot(w, L, color='#58a6ff', linewidth=2)

for eta_val, color, label in [(0.05, '#e3b341', 'η = 0.05 (too small)'),
                               (0.4,  '#3fb950', 'η = 0.4  (just right)')]:
    pts = [2.2]
    for _ in range(14):
        pts.append(pts[-1] - eta_val * 2 * pts[-1])
    pts_w = np.array(pts)
    pts_L = pts_w ** 2
    ax.plot(pts_w, pts_L, 'o-', color=color, markersize=4, linewidth=1.4,
            label=label, alpha=0.9)

ax.set_xlabel('w', fontsize=9)
ax.set_ylabel('Loss L', fontsize=9)
ax.set_title('η too small vs just right', color='#e6edf3', fontsize=9, pad=8)
ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#444', labelcolor='#aaa')

plt.tight_layout(pad=1.5)
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'gradient_descent.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("saved gradient_descent.png")
