import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')

# ── Colours ───────────────────────────────────────────────────────────────────
cell_fill   = '#fef3c7'
cell_border = '#b45309'
input_fill  = '#dbeafe'
input_border= '#2563eb'
output_fill = '#dcfce7'
output_border='#15803d'
arrow_col   = '#9ca3af'
label_col   = '#111111'
sub_col     = '#374151'
wlabel_col  = '#6b7280'

cell_w, cell_h = 1.4, 0.9
cell_r = 0.18   # box corner radius
node_r = 0.28

steps = [
    {'t': 1, 'x': 2.0,  'xt': 'x₁', 'ht_prev': 'h₀', 'ht': 'h₁', 'yt': 'ŷ₁'},
    {'t': 2, 'x': 5.5,  'xt': 'x₂', 'ht_prev': 'h₁', 'ht': 'h₂', 'yt': 'ŷ₂'},
    {'t': 3, 'x': 9.0,  'xt': 'x₃', 'ht_prev': 'h₂', 'ht': 'h₃', 'yt': 'ŷ₃'},
]

cell_y = 2.8   # vertical centre of RNN cell
input_y = 1.0  # centre of input node
output_y = 4.6 # centre of output node

def draw_box(ax, cx, cy, w, h, fill, border, label, sublabel=''):
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad=0.05',
        facecolor=fill, edgecolor=border, linewidth=1.4, zorder=3
    )
    ax.add_patch(rect)
    ax.text(cx, cy + (0.1 if sublabel else 0), label,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=label_col, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel,
                ha='center', va='center', fontsize=8, color=sub_col, zorder=4)

def draw_circle(ax, cx, cy, r, fill, border, label):
    c = plt.Circle((cx, cy), r, facecolor=fill, edgecolor=border,
                   linewidth=1.2, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=9,
            fontweight='bold', color=label_col, zorder=4)

def arrow(ax, x1, y1, x2, y2, col=arrow_col, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw))

# ── Draw each time step ───────────────────────────────────────────────────────
for s in steps:
    cx = s['x']

    # RNN cell box
    draw_box(ax, cx, cell_y, cell_w, cell_h,
             cell_fill, cell_border, 'RNN', f"t = {s['t']}")

    # Input node below
    draw_circle(ax, cx, input_y, node_r, input_fill, input_border, s['xt'])

    # Output node above
    draw_circle(ax, cx, output_y, node_r, output_fill, output_border, s['yt'])

    # Arrow: input → cell
    arrow(ax, cx, input_y + node_r, cx, cell_y - cell_h/2)
    ax.text(cx + 0.18, (input_y + node_r + cell_y - cell_h/2)/2,
            'Wₓ', fontsize=8, color=wlabel_col, va='center')

    # Arrow: cell → output
    arrow(ax, cx, cell_y + cell_h/2, cx, output_y - node_r)
    ax.text(cx + 0.18, (cell_y + cell_h/2 + output_y - node_r)/2,
            'W_y', fontsize=8, color=wlabel_col, va='center')

# ── Hidden state arrows between cells ────────────────────────────────────────
for i in range(len(steps) - 1):
    x1 = steps[i]['x'] + cell_w/2
    x2 = steps[i+1]['x'] - cell_w/2
    arrow(ax, x1, cell_y, x2, cell_y, lw=1.6)
    mx = (x1 + x2) / 2
    ax.text(mx, cell_y + 0.22, steps[i+1]['ht_prev'],
            ha='center', fontsize=9, color=sub_col, fontweight='bold')
    ax.text(mx, cell_y - 0.22, 'W_h',
            ha='center', fontsize=8, color=wlabel_col)

# ── h₀ arrow entering from left ──────────────────────────────────────────────
x0_end = steps[0]['x'] - cell_w/2
arrow(ax, x0_end - 0.9, cell_y, x0_end, cell_y)
ax.text(x0_end - 0.95, cell_y + 0.22, 'h₀ = 0',
        ha='center', fontsize=8.5, color=sub_col)

# ── h₃ arrow exiting right ───────────────────────────────────────────────────
x3_start = steps[-1]['x'] + cell_w/2
arrow(ax, x3_start, cell_y, x3_start + 0.9, cell_y)
ax.text(x3_start + 0.95, cell_y + 0.22, 'h₃',
        ha='center', fontsize=8.5, color=sub_col)

# ── "..." ellipsis hint ───────────────────────────────────────────────────────
ax.text(10.6, cell_y, '…', ha='center', va='center', fontsize=14, color='#9ca3af')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=input_fill,  edgecolor=input_border,  label='Input xₜ'),
    mpatches.Patch(facecolor=cell_fill,   edgecolor=cell_border,   label='RNN cell'),
    mpatches.Patch(facecolor=output_fill, edgecolor=output_border, label='Output ŷₜ'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=8.5,
          framealpha=0.9, edgecolor='#d1d5db')

ax.text(5.5, 5.55, 'RNN Unrolled Across Time — same weights Wₓ, W_h, W_y at every step',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'rnn_unrolled.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved rnn_unrolled.png")
