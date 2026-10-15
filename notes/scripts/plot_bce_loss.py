import numpy as np
import matplotlib.pyplot as plt
import os

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor('white')

y_hat = np.linspace(0.001, 0.999, 400)

for ax in axes:
    ax.set_facecolor('white')
    ax.tick_params(colors='#374151', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')

# ── Left: y = 1  →  L = -log(ŷ) ─────────────────────────────────────────────
ax = axes[0]
loss = -np.log(y_hat)
ax.plot(y_hat, loss, color='#1d4ed8', linewidth=2)
ax.axhline(0, color='#d1d5db', linewidth=0.8)

for yh, xoff, yoff in [(0.9, 0.03, 0.15), (0.5, 0.03, 0.15), (0.1, 0.03, 0.15)]:
    l = -np.log(yh)
    ax.plot(yh, l, 'o', color='#b45309', markersize=6)
    ax.text(yh + xoff, l + yoff, f'ŷ={yh}\nL={l:.2f}',
            color='#b45309', fontsize=7.5, fontweight='bold')

ax.set_title('BCE  —  y = 1,   L = −log(ŷ)', color='#111111', fontsize=11, fontweight='bold')
ax.set_xlabel('ŷ  (predicted probability)', color='#374151', fontsize=9)
ax.set_ylabel('Loss L', color='#374151', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 5)

# ── Right: y = 0  →  L = -log(1 - ŷ) ────────────────────────────────────────
ax = axes[1]
loss = -np.log(1 - y_hat)
ax.plot(y_hat, loss, color='#15803d', linewidth=2)
ax.axhline(0, color='#d1d5db', linewidth=0.8)

for yh, xoff, yoff in [(0.1, 0.03, 0.15), (0.5, 0.03, 0.15), (0.9, -0.22, 0.15)]:
    l = -np.log(1 - yh)
    ax.plot(yh, l, 'o', color='#b45309', markersize=6)
    ax.text(yh + xoff, l + yoff, f'ŷ={yh}\nL={l:.2f}',
            color='#b45309', fontsize=7.5, fontweight='bold')

ax.set_title('BCE  —  y = 0,   L = −log(1 − ŷ)', color='#111111', fontsize=11, fontweight='bold')
ax.set_xlabel('ŷ  (predicted probability)', color='#374151', fontsize=9)
ax.set_ylabel('Loss L', color='#374151', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 5)

fig.suptitle('Binary Cross-Entropy Loss', color='#111111', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'bce_loss.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved bce_loss.png")
