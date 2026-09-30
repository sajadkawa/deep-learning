import os
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.1, 10, 300)

bias2    = 1.2 / x
variance = 0.05 * x ** 1.5
total    = bias2 + variance + 0.1   # 0.1 = irreducible noise

optimal  = x[np.argmin(total)]

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, bias2,    color="#2196F3", linewidth=2, label="Bias²")
ax.plot(x, variance, color="#F44336", linewidth=2, label="Variance")
ax.plot(x, total,    color="#4CAF50", linewidth=2, linestyle="--", label="Total Error")
ax.set_xlabel("Model Complexity →")
ax.set_ylabel("Error")
ax.set_xlim(0.1, 10)
ax.set_ylim(0, 2.2)

ax.axvline(optimal, color="gray", linewidth=1, linestyle=":")
ax.text(optimal + 0.15, 2.0, "Sweet spot", fontsize=9, color="gray")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "images", "bias_variance_tradeoff.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", os.path.abspath(out))
