import numpy as np
import matplotlib.pyplot as plt
import os

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor('#1e1e1e')

y_hat = np.linspace(0.001, 0.999, 400)

for ax in axes:
    ax.set_facecolor('#2d2d2d')
    ax.tick_params(colors='#aaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

# Left: y = 1  →  L = -log(ŷ)
ax = axes[0]
loss = -np.log(y_hat)
ax.plot(y_hat, loss, color='#4fc3f7', linewidth=2)
ax.axhline(0, color='#555', linewidth=0.8)

# annotate key points
for yh, label in [(0.9, 'confident\ncorrect'), (0.5, 'uncertain'), (0.1, 'confident\nwrong')]:
    l = -np.log(yh)
    ax.plot(yh, l, 'o', color='#ffb74d', markersize=6)
    ax.text(yh + 0.03, l + 0.1, f'ŷ={yh}\nL={l:.2f}', color='#ffb74d', fontsize=7.5)

ax.set_title('BCE  —  y = 1,   L = −log(ŷ)', color='white', fontsize=11)
ax.set_xlabel('ŷ  (predicted probability)', color='#aaa', fontsize=9)
ax.set_ylabel('Loss L', color='#aaa', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 5)

# Right: y = 0  →  L = -log(1 - ŷ)
ax = axes[1]
loss = -np.log(1 - y_hat)
ax.plot(y_hat, loss, color='#81c784', linewidth=2)
ax.axhline(0, color='#555', linewidth=0.8)

for yh, label in [(0.1, 'confident\ncorrect'), (0.5, 'uncertain'), (0.9, 'confident\nwrong')]:
    l = -np.log(1 - yh)
    ax.plot(yh, l, 'o', color='#ffb74d', markersize=6)
    ax.text(yh + 0.02, l + 0.1, f'ŷ={yh}\nL={l:.2f}', color='#ffb74d', fontsize=7.5)

ax.set_title('BCE  —  y = 0,   L = −log(1 − ŷ)', color='white', fontsize=11)
ax.set_xlabel('ŷ  (predicted probability)', color='#aaa', fontsize=9)
ax.set_ylabel('Loss L', color='#aaa', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(-0.1, 5)

fig.suptitle('Binary Cross-Entropy Loss', color='white', fontsize=12, y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'bce_loss.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved bce_loss.png")
