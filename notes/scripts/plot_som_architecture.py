import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#1e1e1e')

# --- Grid ---
grid_rows, grid_cols = 4, 5
cell = 1.0
grid_ox, grid_oy = 3.5, 0.5

bmu = (2, 1)   # (row, col) — 0-indexed

def grid_pos(r, c):
    return grid_ox + c * cell, grid_oy + r * cell

node_colors = {}
for r in range(grid_rows):
    for c in range(grid_cols):
        d = abs(r - bmu[0]) + abs(c - bmu[1])   # Manhattan distance for display
        if (r, c) == bmu:
            node_colors[(r, c)] = '#ef5350'
        elif d == 1:
            node_colors[(r, c)] = '#ce93d8'
        elif d == 2:
            node_colors[(r, c)] = '#7e57c2'
        else:
            node_colors[(r, c)] = '#4a4a6a'

# Draw grid edges
for r in range(grid_rows):
    for c in range(grid_cols):
        x, y = grid_pos(r, c)
        if c < grid_cols - 1:
            xn, yn = grid_pos(r, c + 1)
            ax.plot([x, xn], [y, yn], color='#555555', linewidth=0.8, zorder=1)
        if r < grid_rows - 1:
            xn, yn = grid_pos(r + 1, c)
            ax.plot([x, xn], [y, yn], color='#555555', linewidth=0.8, zorder=1)

# Draw nodes
for r in range(grid_rows):
    for c in range(grid_cols):
        x, y = grid_pos(r, c)
        circle = plt.Circle((x, y), 0.28, color=node_colors[(r, c)], zorder=3)
        ax.add_patch(circle)

# BMU label
bx, by = grid_pos(*bmu)
ax.text(bx, by, 'BMU', ha='center', va='center', fontsize=7,
        fontweight='bold', color='white', zorder=4)

# --- Input vector ---
inp_x, inp_y = 1.2, 2.0
ax.add_patch(mpatches.FancyBboxPatch((inp_x - 0.35, inp_y - 0.6), 0.7, 1.2,
             boxstyle='round,pad=0.05', facecolor='#4fc3f7', edgecolor='none', zorder=3))
ax.text(inp_x, inp_y + 0.2, 'x₁', ha='center', va='center', fontsize=10,
        color='#1e1e1e', fontweight='bold', zorder=4)
ax.text(inp_x, inp_y - 0.2, 'x₂', ha='center', va='center', fontsize=10,
        color='#1e1e1e', fontweight='bold', zorder=4)
ax.text(inp_x, inp_y - 0.7, 'Input x', ha='center', va='top', fontsize=9,
        color='#4fc3f7')

# Arrow from input to BMU
ax.annotate('', xy=(bx - 0.3, by), xytext=(inp_x + 0.4, inp_y),
            arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.5))

# Weight vector label on BMU
ax.text(bx + 0.05, by - 0.55, 'w*', ha='center', va='top', fontsize=10,
        color='#ef5350', style='italic')

# Legend
legend_items = [
    mpatches.Patch(color='#ef5350', label='BMU (best matching unit)'),
    mpatches.Patch(color='#ce93d8', label='Neighbourhood d=1'),
    mpatches.Patch(color='#7e57c2', label='Neighbourhood d=2'),
    mpatches.Patch(color='#4a4a6a', label='Outside neighbourhood'),
]
ax.legend(handles=legend_items, loc='upper right', fontsize=8,
          facecolor='#2a2a2a', edgecolor='#555555', labelcolor='#eeeeee')

# Labels
ax.text(grid_ox + (grid_cols - 1) * cell / 2, grid_oy + grid_rows * cell + 0.2,
        '2D map grid', ha='center', fontsize=10, color='#eeeeee')

ax.set_xlim(0, 9)
ax.set_ylim(0, 5.2)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'som_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved som_architecture.png")
