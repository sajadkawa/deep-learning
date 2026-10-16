import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(9, 3.2))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 11.4)
ax.set_ylim(1.0, 3.9)
ax.axis('off')

node_r = 0.38

node_color_val  = '#dbeafe'
node_color_op   = '#fef9c3'
edge_color      = '#9ca3af'

def draw_node(ax, x, y, label, color, fontsize=9):
    circle = plt.Circle((x, y), node_r, facecolor=color, zorder=3,
                         linewidth=1.2, edgecolor='#9ca3af')
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='#111111', zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=edge_color, lw=1.4))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.22
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=7.5,
                color='#374151', style='italic')

# ── Node x positions ──────────────────────────────────────────────
# Row 1 (top): layer 1 path
# Row 2 (bottom): layer 2 path

# Shared input node
x_a0 = 1.0

# Layer 1
x_op1  = 2.6
x_z1   = 4.0
x_f1   = 5.2
x_a1   = 6.4

# Layer 2
x_op2  = 7.6
x_z2   = 8.8
x_f2   = 9.8
x_a2   = 10.8

y_top = 3.2
y_bot = 1.8
y_mid = (y_top + y_bot) / 2

# Input node
draw_node(ax, x_a0, y_mid, 'a⁽⁰⁾\n= x', node_color_val, fontsize=8)

# Layer 1 operation node
draw_node(ax, x_op1, y_top, 'W⁽¹⁾ᵀ·(·)\n+b⁽¹⁾', node_color_op, fontsize=7)

# Layer 1 z node
draw_node(ax, x_z1, y_top, 'z⁽¹⁾', node_color_val)

# Layer 1 activation node
draw_node(ax, x_f1, y_top, 'f₁(·)', node_color_op)

# Layer 1 output
draw_node(ax, x_a1, y_top, 'a⁽¹⁾', node_color_val)

# Layer 2 operation node
draw_node(ax, x_op2, y_bot, 'W⁽²⁾ᵀ·(·)\n+b⁽²⁾', node_color_op, fontsize=7)

# Layer 2 z node
draw_node(ax, x_z2, y_bot, 'z⁽²⁾', node_color_val)

# Layer 2 activation node
draw_node(ax, x_f2, y_bot, 'f₂(·)', node_color_op)

# Layer 2 output
draw_node(ax, x_a2, y_bot, 'a⁽²⁾\n= ŷ', node_color_val, fontsize=8)

# ── Arrows ────────────────────────────────────────────────────────
# a0 → op1
draw_arrow(ax, x_a0 + node_r, y_mid, x_op1 - node_r, y_top)
# op1 → z1
draw_arrow(ax, x_op1 + node_r, y_top, x_z1 - node_r, y_top)
# z1 → f1
draw_arrow(ax, x_z1 + node_r, y_top, x_f1 - node_r, y_top)
# f1 → a1
draw_arrow(ax, x_f1 + node_r, y_top, x_a1 - node_r, y_top)
# a1 → op2
draw_arrow(ax, x_a1 + node_r, y_top, x_op2 - node_r, y_bot)
# op2 → z2
draw_arrow(ax, x_op2 + node_r, y_bot, x_z2 - node_r, y_bot)
# z2 → f2
draw_arrow(ax, x_z2 + node_r, y_bot, x_f2 - node_r, y_bot)
# f2 → a2
draw_arrow(ax, x_f2 + node_r, y_bot, x_a2 - node_r, y_bot)

# ── Legend ────────────────────────────────────────────────────────
val_patch = mpatches.Patch(facecolor=node_color_val, label='Value node', linewidth=1,
                            edgecolor='#9ca3af')
op_patch  = mpatches.Patch(facecolor=node_color_op,  label='Operation node', linewidth=1,
                            edgecolor='#9ca3af')
ax.legend(handles=[val_patch, op_patch], loc='lower left', fontsize=8.5,
          facecolor='white', edgecolor='#d1d5db', labelcolor='#111111', framealpha=1.0)

# ── Row labels ────────────────────────────────────────────────────
ax.text(0.2, y_top, 'Layer 1', va='center', fontsize=8, color='#374151', style='italic')
ax.text(0.2, y_bot, 'Layer 2', va='center', fontsize=8, color='#374151', style='italic')

ax.text(5.5, 3.75, 'Computational graph — 2-layer feedforward network',
        ha='center', fontsize=9, fontweight='bold', color='#111111')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'computational_graph.png')
plt.savefig(out_path, dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved computational_graph.png")
