import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

grid_rows, grid_cols = 4, 5
cell = 1.0
grid_ox, grid_oy = 3.5, 0.5

bmu = (2, 1)

def grid_pos(r, c):
    return grid_ox + c * cell, grid_oy + r * cell

node_colors = {}
node_borders = {}
for r in range(grid_rows):
    for c in range(grid_cols):
        d = abs(r - bmu[0]) + abs(c - bmu[1])
        if (r, c) == bmu:
            node_colors[(r, c)] = '#fee2e2'
            node_borders[(r, c)] = '#dc2626'
        elif d == 1:
            node_colors[(r, c)] = '#fef3c7'
            node_borders[(r, c)] = '#b45309'
        elif d == 2:
            node_colors[(r, c)] = '#dbeafe'
            node_borders[(r, c)] = '#2563eb'
        else:
            node_colors[(r, c)] = '#f3f4f6'
            node_borders[(r, c)] = '#d1d5db'

for r in range(grid_rows):
    for c in range(grid_cols):
        x, y = grid_pos(r, c)
        if c < grid_cols - 1:
            xn, yn = grid_pos(r, c + 1)
            ax.plot([x, xn], [y, yn], color='#d1d5db', linewidth=0.8, zorder=1)
        if r < grid_rows - 1:
            xn, yn = grid_pos(r + 1, c)
            ax.plot([x, xn], [y, yn], color='#d1d5db', linewidth=0.8, zorder=1)

for r in range(grid_rows):
    for c in range(grid_cols):
        x, y = grid_pos(r, c)
        circle = plt.Circle((x, y), 0.28, facecolor=node_colors[(r, c)],
                             edgecolor=node_borders[(r, c)], linewidth=1.5, zorder=3)
        ax.add_patch(circle)

bx, by = grid_pos(*bmu)
ax.text(bx, by, 'BMU', ha='center', va='center', fontsize=7,
        fontweight='bold', color='#111111', zorder=4)

inp_x, inp_y = 1.2, 2.0
ax.add_patch(mpatches.FancyBboxPatch((inp_x - 0.35, inp_y - 0.6), 0.7, 1.2,
             boxstyle='round,pad=0.05', facecolor='#dbeafe',
             edgecolor='#2563eb', linewidth=1.5, zorder=3))
ax.text(inp_x, inp_y + 0.2, 'x₁', ha='center', va='center', fontsize=10,
        color='#111111', fontweight='bold', zorder=4)
ax.text(inp_x, inp_y - 0.2, 'x₂', ha='center', va='center', fontsize=10,
        color='#111111', fontweight='bold', zorder=4)
ax.text(inp_x, inp_y - 0.7, 'Input x', ha='center', va='top', fontsize=9,
        color='#2563eb')

ax.annotate('', xy=(bx - 0.3, by), xytext=(inp_x + 0.4, inp_y),
            arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.5))

ax.text(bx + 0.05, by - 0.55, 'w*', ha='center', va='top', fontsize=10,
        color='#dc2626', style='italic')

legend_items = [
    mpatches.Patch(facecolor='#fee2e2', edgecolor='#dc2626', label='BMU (best matching unit)'),
    mpatches.Patch(facecolor='#fef3c7', edgecolor='#b45309', label='Neighbourhood d=1'),
    mpatches.Patch(facecolor='#dbeafe', edgecolor='#2563eb', label='Neighbourhood d=2'),
    mpatches.Patch(facecolor='#f3f4f6', edgecolor='#d1d5db', label='Outside neighbourhood'),
]
ax.legend(handles=legend_items, loc='upper right', fontsize=8,
          facecolor='white', edgecolor='#d1d5db', labelcolor='#111111')

ax.text(grid_ox + (grid_cols - 1) * cell / 2, grid_oy + grid_rows * cell + 0.2,
        '2D map grid', ha='center', fontsize=10, color='#111111')

ax.set_xlim(0, 9)
ax.set_ylim(0, 5.2)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'som_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved som_architecture.png")
