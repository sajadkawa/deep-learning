import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('white')
fig.suptitle("XOR — Why One Line Fails, Why Two Lines Work",
             fontsize=13, fontweight='bold', color='#111111')

points = {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0}
c1, c0 = '#1d4ed8', '#b91c1c'   # XOR=1 blue, XOR=0 red — matches all XOR diagrams

def plot_points(ax):
    for (x1, x2), label in points.items():
        color  = c1 if label == 1 else c0
        marker = 'o' if label == 1 else 'X'
        ax.scatter(x1, x2, color=color, marker=marker, s=180, zorder=5,
                   edgecolors='#374151', linewidths=0.8)
        offset = (-48, 8) if x2 == 1 else (-48, -18)
        ax.annotate(f"({x1},{x2})\nXOR={label}", (x1, x2),
                    textcoords="offset points", xytext=offset,
                    fontsize=8, color=color)

def style_ax(ax, title):
    ax.set_facecolor('white')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁", fontsize=11, color='#374151')
    ax.set_ylabel("x₂", fontsize=11, color='#374151')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(colors='#374151')
    ax.grid(True, linestyle='--', alpha=0.3, color='#9ca3af')
    ax.axhline(0, color='#e5e7eb', linewidth=0.8)
    ax.axvline(0, color='#e5e7eb', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d5db')
    ax.set_title(title, fontsize=11, color='#111111')

# ── Left: single line attempts ────────────────────────────────────────────────
ax = axes[0]
plot_points(ax)
style_ax(ax, "Single Line — Always Fails")

x_vals = np.linspace(-0.5, 1.5, 200)
ax.plot(x_vals, -x_vals + 1.5, color='#6b7280', lw=1.5, linestyle='--', alpha=0.7, label='Attempt 1')
ax.plot(x_vals,  x_vals - 0.2, color='#9333ea', lw=1.5, linestyle='--', alpha=0.7, label='Attempt 2')
ax.plot(x_vals, [0.5]*len(x_vals), color='#ca8a04', lw=1.5, linestyle='--', alpha=0.7, label='Attempt 3')

ax.text(0.5, -0.42, "No single line separates ● from ✕", ha='center',
        fontsize=9, color='#6b7280', style='italic')
ax.legend(loc='upper left', fontsize=8, facecolor='white', edgecolor='#d1d5db')

# ── Right: two lines solve it ─────────────────────────────────────────────────
ax = axes[1]
plot_points(ax)
style_ax(ax, "Two Lines — XOR Solved")

line1 = 0.5 - x_vals   # h₁: OR boundary
line2 = 1.5 - x_vals   # h₂: NAND boundary

ax.plot(x_vals, line1, color='#16a34a', lw=2, label='h₁: OR boundary  $x_1+x_2=0.5$')
ax.plot(x_vals, line2, color='#ea580c', lw=2, label='h₂: NAND boundary  $x_1+x_2=1.5$')
ax.fill_between(x_vals,
                np.maximum(line1, -0.5),
                np.minimum(line2,  1.5),
                where=(line2 > line1),
                alpha=0.10, color='#16a34a', label='XOR=1 region')

ax.text(0.5, -0.42, "Two boundaries isolate the XOR=1 points", ha='center',
        fontsize=9, color='#6b7280', style='italic')
ax.legend(loc='upper right', fontsize=7.5, facecolor='white', edgecolor='#d1d5db')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'xor_two_lines.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved to {os.path.abspath(output_path)}")
