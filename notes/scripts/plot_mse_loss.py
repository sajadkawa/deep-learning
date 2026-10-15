import numpy as np
import matplotlib.pyplot as plt
import os

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_facecolor('white')

for ax in axes:
    ax.set_facecolor('white')
    ax.tick_params(colors='#374151', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')

# ── Panel 1: Geometric intuition ─────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(-0.5, 7)
ax.set_ylim(-1.5, 3)
ax.axis('off')
ax.set_title('Geometric Intuition', color='#111111', fontsize=11, fontweight='bold')

# number line
ax.annotate('', xy=(6.8, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.2))

# small error: ŷ=3, y=4
for x, label, col in [(3.0, 'ŷ = 3', '#b45309'), (4.0, 'y = 4', '#15803d')]:
    ax.plot(x, 0, 'o', color=col, markersize=9, zorder=3)
    ax.text(x, 0.25, label, color=col, fontsize=9, ha='center', fontweight='bold')

ax.annotate('', xy=(4.0, -0.6), xytext=(3.0, -0.6),
            arrowprops=dict(arrowstyle='<->', color='#1d4ed8', lw=1.5))
ax.text(3.5, -1.1, 'error = 1\nL = 1² = 1', color='#1d4ed8', fontsize=8.5, ha='center')

# large error: ŷ=1, y=4
for x, label, col in [(1.0, 'ŷ = 1', '#b45309'), (4.0, 'y = 4', '#15803d')]:
    ax.plot(x, 1.6, 'o', color=col, markersize=9, zorder=3)
    ax.text(x, 1.85, label, color=col, fontsize=9, ha='center', fontweight='bold')

ax.annotate('', xy=(4.0, 1.0), xytext=(1.0, 1.0),
            arrowprops=dict(arrowstyle='<->', color='#7e22ce', lw=1.5))
ax.text(2.5, 0.6, 'error = 3\nL = 3² = 9', color='#7e22ce', fontsize=8.5, ha='center')

ax.text(6.9, 0, 'value', color='#6b7280', fontsize=8, va='center')

# ── Panel 2: MSE parabola ─────────────────────────────────────────────────────
ax = axes[1]
error = np.linspace(-4, 4, 400)
mse   = error ** 2
ax.plot(error, mse, color='#1d4ed8', linewidth=2)
ax.axhline(0, color='#d1d5db', linewidth=0.8)
ax.axvline(0, color='#d1d5db', linewidth=0.8, linestyle='--')

ax.plot(1, 1, 'o', color='#15803d', markersize=7)
ax.text(1.2, 2.2, 'error=1\nL=1', color='#15803d', fontsize=8, fontweight='bold')

ax.plot(3, 9, 'o', color='#b45309', markersize=7)
ax.text(1.8, 10.5, 'error=3\nL=9', color='#b45309', fontsize=8, fontweight='bold')

ax.set_title('L = (y − ŷ)²', color='#111111', fontsize=11, fontweight='bold')
ax.set_xlabel('error  (y − ŷ)', color='#374151', fontsize=9)
ax.set_ylabel('Loss L', color='#374151', fontsize=9)
ax.set_xlim(-4, 4)
ax.set_ylim(-0.5, 16)

# ── Panel 3: Gradient of MSE ──────────────────────────────────────────────────
ax = axes[2]
y_hat  = np.linspace(-2, 4, 400)
y_true = 1.0
grad   = 2 * (y_hat - y_true)

ax.plot(y_hat, grad, color='#7e22ce', linewidth=2)
ax.axhline(0, color='#d1d5db', linewidth=0.8)
ax.axvline(y_true, color='#15803d', linewidth=1.2, linestyle='--')
ax.text(y_true + 0.05, 3.8, 'y = 1\n(truth)', color='#15803d', fontsize=8, fontweight='bold')

for yh, col in [(3.5, '#b45309'), (-0.5, '#1d4ed8')]:
    g = 2 * (yh - y_true)
    ax.plot(yh, g, 'o', color=col, markersize=7)
    ax.annotate(f'ŷ={yh}\n∂L/∂ŷ={g:.1f}',
                xy=(yh, g), xytext=(yh - 1.2, g - 0.8),
                color=col, fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=col, lw=1))

ax.set_title('Gradient  ∂L/∂ŷ = 2(ŷ − y)', color='#111111', fontsize=11, fontweight='bold')
ax.set_xlabel('ŷ  (prediction)', color='#374151', fontsize=9)
ax.set_ylabel('∂L/∂ŷ', color='#374151', fontsize=9)
ax.set_xlim(-2, 4)
ax.set_ylim(-5, 7)

fig.suptitle('Mean Squared Error — Geometry, Shape, and Gradient',
             color='#111111', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'mse_loss.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved mse_loss.png")
