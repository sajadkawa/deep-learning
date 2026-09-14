import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# OR truth table
points = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 1,
}

# Final learned weights from note 02
w1, w2, b = 0.2, 0.2, 0.3
threshold = 0.5

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Perceptron — OR Function", fontsize=14, fontweight="bold")

# ── Plot 1: Data points only ──────────────────────────────────────────────────
ax1 = axes[0]
ax1.set_title("OR Function — Data Points", fontsize=12)

for (x1, x2), label in points.items():
    if label == 1:
        ax1.scatter(x1, x2, color="steelblue", s=200, zorder=5)
        ax1.annotate(f"({x1},{x2})\nOR=1", (x1, x2),
                     textcoords="offset points", xytext=(10, 8), fontsize=9, color="steelblue")
    else:
        ax1.scatter(x1, x2, color="tomato", s=200, marker="X", zorder=5)
        ax1.annotate(f"({x1},{x2})\nOR=0", (x1, x2),
                     textcoords="offset points", xytext=(10, -18), fontsize=9, color="tomato")

ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel("x₁", fontsize=11)
ax1.set_ylabel("x₂", fontsize=11)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.axhline(0, color="black", linewidth=0.5)
ax1.axvline(0, color="black", linewidth=0.5)

blue_patch = mpatches.Patch(color="steelblue", label="OR = 1")
red_patch  = mpatches.Patch(color="tomato",    label="OR = 0")
ax1.legend(handles=[blue_patch, red_patch], loc="upper left", fontsize=9)

# ── Plot 2: Data points + decision boundary ───────────────────────────────────
ax2 = axes[1]
ax2.set_title("OR Function — Learned Decision Boundary", fontsize=12)

for (x1, x2), label in points.items():
    if label == 1:
        ax2.scatter(x1, x2, color="steelblue", s=200, zorder=5)
        ax2.annotate(f"({x1},{x2})\nOR=1", (x1, x2),
                     textcoords="offset points", xytext=(10, 8), fontsize=9, color="steelblue")
    else:
        ax2.scatter(x1, x2, color="tomato", s=200, marker="X", zorder=5)
        ax2.annotate(f"({x1},{x2})\nOR=0", (x1, x2),
                     textcoords="offset points", xytext=(10, -18), fontsize=9, color="tomato")

# Decision boundary: w1*x1 + w2*x2 + b = threshold
# => x2 = (threshold - b - w1*x1) / w2
x1_vals = np.linspace(-0.5, 1.5, 200)
x2_vals = (threshold - b - w1 * x1_vals) / w2

ax2.plot(x1_vals, x2_vals, color="green", linewidth=2,
         label=f"Decision boundary\n$x_1 + x_2 = 1$")

# Shade the two regions
ax2.fill_between(x1_vals, x2_vals, 1.5,  alpha=0.08, color="steelblue", label="Predicted OR=1")
ax2.fill_between(x1_vals, x2_vals, -0.5, alpha=0.08, color="tomato",    label="Predicted OR=0")

ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel("x₁", fontsize=11)
ax2.set_ylabel("x₂", fontsize=11)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.axvline(0, color="black", linewidth=0.5)
ax2.legend(loc="upper right", fontsize=9)

# Annotate the formula on the plot
ax2.text(0.02, 0.02,
         f"w₁={w1}, w₂={w2}, b={b}\nz = w₁x₁ + w₂x₂ + b\nfire if z ≥ {threshold}",
         transform=ax2.transAxes, fontsize=8.5,
         verticalalignment="bottom",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "..", "images", "or_function.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {os.path.abspath(output_path)}")
plt.show()
