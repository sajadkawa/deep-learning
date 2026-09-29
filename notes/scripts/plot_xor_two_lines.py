import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("XOR — Why One Line Fails, Why Two Lines Work", fontsize=13, fontweight='bold')

points = {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0}
colors = {0: 'tomato', 1: 'steelblue'}
markers = {0: 'X', 1: 'o'}

def plot_points(ax):
    for (x1, x2), label in points.items():
        ax.scatter(x1, x2, color=colors[label], marker=markers[label],
                   s=200, zorder=5, linewidths=2,
                   edgecolors='black' if label == 1 else colors[label])
        offset = (10, 8) if x2 == 1 else (10, -18)
        ax.annotate(f"({x1},{x2})\nXOR={label}", (x1, x2),
                    textcoords="offset points", xytext=offset,
                    fontsize=9, color=colors[label])

def style_ax(ax, title):
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁", fontsize=11)
    ax.set_ylabel("x₂", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=11)

# ── Left: single line attempts ────────────────────────────────────────────────
ax = axes[0]
plot_points(ax)
style_ax(ax, "Single Line — Always Fails")

x_vals = np.linspace(-0.5, 1.5, 200)
ax.plot(x_vals, -x_vals + 1.5, color='gray', linewidth=1.5, linestyle='--', alpha=0.6, label='Attempt 1')
ax.plot(x_vals, x_vals - 0.2,  color='purple', linewidth=1.5, linestyle='--', alpha=0.6, label='Attempt 2')
ax.plot(x_vals, [0.5]*len(x_vals), color='olive', linewidth=1.5, linestyle='--', alpha=0.6, label='Attempt 3')

ax.text(0.5, -0.42, "No single line separates ● from ✕", ha='center',
        fontsize=9, color='gray', style='italic')
ax.legend(loc='upper right', fontsize=8)

# ── Right: two lines solve it ─────────────────────────────────────────────────
ax = axes[1]
plot_points(ax)
style_ax(ax, "Two Lines — XOR Solved")

x_vals = np.linspace(-0.5, 1.5, 200)

# Line 1: OR boundary  x1 + x2 = 0.5  (h1, separates (0,0) from rest)
line1 = 0.5 - x_vals
ax.plot(x_vals, line1, color='steelblue', linewidth=2, label='h₁: OR boundary\n$x_1+x_2=0.5$')

# Line 2: NAND boundary  x1 + x2 = 1.5  (h2, separates (1,1) from rest)
line2 = 1.5 - x_vals
ax.plot(x_vals, line2, color='tomato', linewidth=2, label='h₂: NAND boundary\n$x_1+x_2=1.5$')

# Shade the XOR=1 region (between the two lines)
ax.fill_between(x_vals,
                np.maximum(line1, -0.5),
                np.minimum(line2, 1.5),
                where=(line2 > line1),
                alpha=0.12, color='green', label='XOR=1 region')

ax.text(0.5, -0.42, "Two boundaries isolate the XOR=1 points", ha='center',
        fontsize=9, color='gray', style='italic')
ax.legend(loc='upper right', fontsize=8)

blue_patch  = mpatches.Patch(color='steelblue', label='XOR = 1')
red_patch   = mpatches.Patch(color='tomato',    label='XOR = 0')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'xor_two_lines.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {os.path.abspath(output_path)}")

