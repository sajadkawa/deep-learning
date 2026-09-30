import os
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 51)

# Underfitting: both losses high and flat
train_under = 0.85 * np.exp(-epochs / 200) + 0.72
val_under   = 0.85 * np.exp(-epochs / 200) + 0.75

# Overfitting: training keeps falling, validation rises after epoch ~15
train_over = 0.9 * np.exp(-epochs / 18) + 0.05
val_over   = 0.9 * np.exp(-epochs / 40) + 0.05 + np.maximum(0, (epochs - 15) * 0.012)

# Good fit: both decrease and converge with small gap
train_good = 0.85 * np.exp(-epochs / 15) + 0.08
val_good   = 0.85 * np.exp(-epochs / 15) + 0.14

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, train, val, title in zip(
    axes,
    [train_under, train_over, train_good],
    [val_under,   val_over,   val_good],
    ["Underfitting (High Bias)", "Overfitting (High Variance)", "Good Fit"],
):
    ax.plot(epochs, train, color="#2196F3", linewidth=2, label="Training loss")
    ax.plot(epochs, val,   color="#F44336", linewidth=2, linestyle="--", label="Validation loss")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "images", "learning_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", os.path.abspath(out))
