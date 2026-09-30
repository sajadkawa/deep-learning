import os
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 61)

train = 0.9 * np.exp(-epochs / 18) + 0.05
val   = 0.9 * np.exp(-epochs / 40) + 0.05 + np.maximum(0, (epochs - 20) * 0.012)

best_epoch = epochs[np.argmin(val)]

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(epochs, train, color="#2196F3", linewidth=2, label="Training loss")
ax.plot(epochs, val,   color="#F44336", linewidth=2, linestyle="--", label="Validation loss")
ax.axvline(best_epoch, color="gray", linewidth=1.5, linestyle=":")
ax.text(best_epoch + 0.8, 0.72, "Stop here\n(best validation loss)", fontsize=9, color="gray")

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_ylim(0, 0.9)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "images", "early_stopping.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", os.path.abspath(out))
