import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor('#1e1e1e')

for ax in axes:
    ax.set_facecolor('#2d2d2d')
    ax.tick_params(colors='#aaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

# ── Panel 1: Geometric intuition — number line ──────────────────────────────
ax = axes[0]
ax.set_xlim(-0.5, 7)
ax.set_ylim(-1.5, 3)
ax.axis('off')
ax.set_title('Geometric Intuition', color='white', fontsize=11)

# number line
ax.annotate('', xy=(6.8, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.2))

# small error: ŷ=3, y=4
for x, label, col in [(3.0, 'ŷ = 3', '#ffb74d'), (4.0, 'y = 4', '#81c784')]:
    ax.plot(x, 0, 'o', color=col, markersize=9, zorder=3)
    ax.text(x, 0.25, label, color=col, fontsize=9, ha='center')

ax.annotate('', xy=(4.0, -0.6), xytext=(3.0, -0.6),
            arrowprops=dict(arrowstyle='<->', color='#4fc3f7', lw=1.5))
ax.text(3.5, -0.95, 'error = 1\nL = 1²= 1', color='#4fc3f7', fontsize=8.5, ha='center')

# large error: ŷ=1, y=4
for x, label, col in [(1.0, 'ŷ = 1', '#ff8a65'), (4.0, 'y = 4', '#81c784')]:
    ax.plot(x, 1.6, 'o', color=col, markersize=9, zorder=3)
    ax.text(x, 1.85, label, color=col, fontsize=9, ha='center')

ax.annotate('', xy=(4.0, 1.0), xytext=(1.0, 1.0),
            arrowprops=dict(arrowstyle='<->', color='#ce93d8', lw=1.5))
ax.text(2.5, 0.65, 'error = 3\nL = 3² = 9', color='#ce93d8', fontsize=8.5, ha='center')

ax.text(6.9, 0, 'value', color='#aaa', fontsize=8, va='center')

# ── Panel 2: MSE parabola ────────────────────────────────────────────────────
ax = axes[1]
error = np.linspace(-4, 4, 400)
mse   = error ** 2
ax.plot(error, mse, color='#4fc3f7', linewidth=2)
ax.axhline(0, color='#555', linewidth=0.8)
ax.axvline(0, color='#555', linewidth=0.8, linestyle='--')

ax.plot(1, 1, 'o', color='#81c784', markersize=7)
ax.text(1.15, 1.3, 'error=1\nL=1', color='#81c784', fontsize=8)

ax.plot(3, 9, 'o', color='#ffb74d', markersize=7)
ax.text(3.1, 8.0, 'error=3\nL=9', color='#ffb74d', fontsize=8)

ax.set_title('L = (y − ŷ)²', color='white', fontsize=11)
ax.set_xlabel('error  (y − ŷ)', color='#aaa', fontsize=9)
ax.set_ylabel('Loss L', color='#aaa', fontsize=9)
ax.set_xlim(-4, 4)
ax.set_ylim(-0.5, 16)

# ── Panel 3: Gradient of MSE ─────────────────────────────────────────────────
ax = axes[2]
y_hat = np.linspace(-2, 4, 400)
y_true = 1.0
grad  = 2 * (y_hat - y_true)

ax.plot(y_hat, grad, color='#ce93d8', linewidth=2)
ax.axhline(0, color='#555', linewidth=0.8)
ax.axvline(y_true, color='#81c784', linewidth=1.2, linestyle='--')
ax.text(y_true + 0.05, 3.8, 'y = 1\n(truth)', color='#81c784', fontsize=8)

# annotate gradient at two points
for yh, col in [(3.5, '#ffb74d'), (-0.5, '#4fc3f7')]:
    g = 2 * (yh - y_true)
    ax.plot(yh, g, 'o', color=col, markersize=7)
    ax.annotate(f'ŷ={yh}\n∂L/∂ŷ={g:.1f}',
                xy=(yh, g), xytext=(yh - 1.2, g - 0.8),
                color=col, fontsize=8,
                arrowprops=dict(arrowstyle='->', color=col, lw=1))

ax.set_title('Gradient  ∂L/∂ŷ = 2(ŷ − y)', color='white', fontsize=11)
ax.set_xlabel('ŷ  (prediction)', color='#aaa', fontsize=9)
ax.set_ylabel('∂L/∂ŷ', color='#aaa', fontsize=9)
ax.set_xlim(-2, 4)
ax.set_ylim(-5, 7)

fig.suptitle('Mean Squared Error — Geometry, Shape, and Gradient', color='white', fontsize=12, y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'mse_loss.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved mse_loss.png")
