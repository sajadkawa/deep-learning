import numpy as np
import matplotlib.pyplot as plt
import os

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor('#1e1e1e')

classes = ['Class 1', 'Class 2', 'Class 3']
raw     = [2.0, 1.0, 0.5]
exp_z   = np.exp(raw)
probs   = exp_z / exp_z.sum()
color   = '#4fc3f7'

for ax in axes:
    ax.set_facecolor('#2d2d2d')
    ax.tick_params(colors='#aaa', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

# left: raw scores
axes[0].bar(classes, raw, color=color, alpha=0.85, width=0.5)
axes[0].set_title("Raw Scores  z", color='white', fontsize=11)
axes[0].set_ylabel("Score", color='#aaa', fontsize=9)
axes[0].set_ylim(0, 2.8)
for i, v in enumerate(raw):
    axes[0].text(i, v + 0.05, f"{v:.1f}", ha='center', color='white', fontsize=10)

# right: softmax probabilities
axes[1].bar(classes, probs, color='#81c784', alpha=0.85, width=0.5)
axes[1].set_title("Softmax Probabilities", color='white', fontsize=11)
axes[1].set_ylabel("Probability", color='#aaa', fontsize=9)
axes[1].set_ylim(0, 0.80)
axes[1].axhline(1/3, color='#666', linewidth=0.8, linestyle='--')
axes[1].text(2.55, 1/3 + 0.01, 'uniform (0.333)', color='#888', fontsize=8)
for i, v in enumerate(probs):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', color='white', fontsize=10)

fig.suptitle("Softmax: Raw Scores → Probability Distribution", color='white', fontsize=12, y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'softmax.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved softmax.png")
