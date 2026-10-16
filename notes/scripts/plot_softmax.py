import numpy as np
import matplotlib.pyplot as plt
import os

classes = ['Class 1', 'Class 2', 'Class 3']
raw     = [2.0, 1.0, 0.5]
exp_z   = np.exp(raw)
probs   = exp_z / exp_z.sum()

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
fig.patch.set_facecolor('white')

for ax in axes:
    ax.set_facecolor('white')
    ax.tick_params(colors='#374151', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')

# left: raw scores
axes[0].bar(classes, raw, color='#93c5fd', edgecolor='#2563eb', linewidth=0.8, width=0.5)
axes[0].set_title("Raw Scores  z", color='#111111', fontsize=11)
axes[0].set_ylabel("Score", color='#374151', fontsize=9)
axes[0].set_ylim(0, 2.8)
for i, v in enumerate(raw):
    axes[0].text(i, v + 0.05, f"{v:.1f}", ha='center', color='#111111', fontsize=10)

# right: softmax probabilities
axes[1].bar(classes, probs, color='#86efac', edgecolor='#15803d', linewidth=0.8, width=0.5)
axes[1].set_title("Softmax Probabilities", color='#111111', fontsize=11)
axes[1].set_ylabel("Probability", color='#374151', fontsize=9)
axes[1].set_ylim(0, 0.80)
axes[1].axhline(1/3, color='#d1d5db', linewidth=0.8, linestyle='--')
axes[1].text(2.55, 1/3 + 0.01, 'uniform (0.333)', color='#9ca3af', fontsize=8)
for i, v in enumerate(probs):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', color='#111111', fontsize=10)

fig.suptitle("Softmax: Raw Scores → Probability Distribution",
             color='#111111', fontsize=12, y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'softmax.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved softmax.png")
