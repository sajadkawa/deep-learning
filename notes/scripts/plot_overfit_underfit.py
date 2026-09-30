import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# True underlying pattern: a sine curve
x_true = np.linspace(0, 3, 200)
y_true = np.sin(x_true * np.pi / 1.5)

# Training points: 10 samples from the true curve + small noise
x_pts = np.linspace(0.1, 2.9, 10)
y_pts = np.sin(x_pts * np.pi / 1.5) + np.random.normal(0, 0.08, 10)

# Underfitting: degree-1 polynomial (straight line)
c1 = np.polyfit(x_pts, y_pts, 1)
y_under = np.polyval(c1, x_true)

# Overfitting: degree-9 polynomial
c9 = np.polyfit(x_pts, y_pts, 9)
y_over = np.polyval(c9, x_true)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, y_model, label, color, title in zip(
    axes,
    [y_under, y_over],
    ["Underfitted model (degree 1)", "Overfitted model (degree 9)"],
    ["#F44336", "#9C27B0"],
    ["Underfitting", "Overfitting"],
):
    ax.plot(x_true, y_true, color="#2196F3", linewidth=2, label="True pattern", zorder=1)
    ax.scatter(x_pts, y_pts, color="black", s=40, zorder=3, label="Training points")
    ax.plot(x_true, y_model, color=color, linewidth=2, linestyle="--", label=label, zorder=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "images", "overfit_underfit.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", os.path.abspath(out))
