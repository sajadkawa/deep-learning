import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

points = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
w1, w2, b, threshold = 0.2, 0.2, 0.3, 0.5

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('white')
fig.suptitle("Perceptron — OR Function", fontsize=14, fontweight="bold", color='#111111')

for ax, show_boundary in [(axes[0], False), (axes[1], True)]:
    ax.set_facecolor('white')

    for (x1, x2), label in points.items():
        if label == 1:
            ax.scatter(x1, x2, color='#1d4ed8', s=200, zorder=5)
            ax.annotate(f"({x1},{x2})\nOR=1", (x1, x2),
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=9, color='#1d4ed8', fontweight='bold')
        else:
            ax.scatter(x1, x2, color='#b91c1c', s=200, marker="X", zorder=5)
            ax.annotate(f"({x1},{x2})\nOR=0", (x1, x2),
                        textcoords="offset points", xytext=(10, -18),
                        fontsize=9, color='#b91c1c', fontweight='bold')

    if show_boundary:
        x1_vals = np.linspace(-0.5, 1.5, 200)
        x2_vals = (threshold - b - w1 * x1_vals) / w2
        ax.plot(x1_vals, x2_vals, color='#15803d', linewidth=2,
                label="Decision boundary\n$x_1 + x_2 = 1$")
        ax.fill_between(x1_vals, x2_vals, 1.5,  alpha=0.08, color='#1d4ed8', label="Predicted OR=1")
        ax.fill_between(x1_vals, x2_vals, -0.5, alpha=0.08, color='#b91c1c', label="Predicted OR=0")
        ax.text(0.02, 0.02,
                f"w₁={w1}, w₂={w2}, b={b}\nz = w₁x₁ + w₂x₂ + b\nfire if z ≥ {threshold}",
                transform=ax.transAxes, fontsize=8.5, verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor='#fef9c3',
                          edgecolor='#ca8a04', linewidth=1))
        ax.set_title("OR Function — Learned Decision Boundary",
                     fontsize=11, fontweight='bold', color='#111111')
        ax.legend(loc="upper right", fontsize=9, facecolor='white', edgecolor='#d1d5db')
    else:
        ax.set_title("OR Function — Data Points",
                     fontsize=11, fontweight='bold', color='#111111')
        blue_patch = mpatches.Patch(color='#1d4ed8', label='OR = 1')
        red_patch  = mpatches.Patch(color='#b91c1c', label='OR = 0')
        ax.legend(handles=[blue_patch, red_patch], loc="upper left", fontsize=9,
                  facecolor='white', edgecolor='#d1d5db')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁", fontsize=11, color='#374151')
    ax.set_ylabel("x₂", fontsize=11, color='#374151')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(colors='#374151')
    ax.grid(True, linestyle="--", alpha=0.3, color='#9ca3af')
    ax.axhline(0, color='#e5e7eb', linewidth=0.8)
    ax.axvline(0, color='#e5e7eb', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "..", "images", "or_function.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white', edgecolor='none')
plt.close()
print("Saved or_function.png")
