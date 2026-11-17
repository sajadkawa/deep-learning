import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.5)
ax.axis('off')

# ── Colours ───────────────────────────────────────────────────────────────────
cell_fill    = '#fef3c7'
cell_border  = '#b45309'
input_fill   = '#dbeafe'
input_border = '#2563eb'
loss_fill    = '#fee2e2'
loss_border  = '#dc2626'
fwd_col      = '#9ca3af'
bwd_col      = '#dc2626'
label_col    = '#111111'
sub_col      = '#374151'
wlabel_col   = '#6b7280'

cell_w, cell_h = 1.4, 0.9
node_r = 0.28

steps = [
    {'t': 1, 'x': 2.0,  'xt': 'x₁', 'ht': 'h₁', 'lt': 'L₁'},
    {'t': 2, 'x': 5.5,  'xt': 'x₂', 'ht': 'h₂', 'lt': 'L₂'},
    {'t': 3, 'x': 9.0,  'xt': 'x₃', 'ht': 'h₃', 'lt': 'L₃'},
]

cell_y   = 3.2
input_y  = 1.2
loss_y   = 5.0

def draw_box(ax, cx, cy, w, h, fill, border, label, sublabel=''):
    rect = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.05',
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

def fwd_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=fwd_col, lw=1.4))

def bwd_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=bwd_col, lw=1.6,
                                connectionstyle='arc3,rad=-0.35'))

# ── Draw each time step ───────────────────────────────────────────────────────
for s in steps:
    cx = s['x']

    draw_box(ax, cx, cell_y, cell_w, cell_h,
             cell_fill, cell_border, 'RNN', f"t = {s['t']}")

    draw_circle(ax, cx, input_y, node_r, input_fill, input_border, s['xt'])

    # Loss box above
    draw_box(ax, cx, loss_y, 0.9, 0.55, loss_fill, loss_border, s['lt'])

    # Forward: input → cell
    fwd_arrow(ax, cx, input_y + node_r, cx, cell_y - cell_h/2)

    # Forward: cell → loss
    fwd_arrow(ax, cx, cell_y + cell_h/2, cx, loss_y - 0.55/2)

# ── Forward hidden state arrows ───────────────────────────────────────────────
for i in range(len(steps) - 1):
    x1 = steps[i]['x'] + cell_w/2
    x2 = steps[i+1]['x'] - cell_w/2
    fwd_arrow(ax, x1, cell_y, x2, cell_y)
    mx = (x1 + x2) / 2
    ax.text(mx, cell_y + 0.22, steps[i+1]['ht'],
            ha='center', fontsize=9, color=sub_col, fontweight='bold')

# ── h₀ entering ───────────────────────────────────────────────────────────────
x0_end = steps[0]['x'] - cell_w/2
fwd_arrow(ax, x0_end - 0.9, cell_y, x0_end, cell_y)
ax.text(x0_end - 0.95, cell_y + 0.22, 'h₀=0', ha='center', fontsize=8, color=sub_col)

# ── Backward gradient arrows (curved, below cell level) ──────────────────────
# L₃ → h₃ (straight down into cell, then back)
for i in range(len(steps) - 1, -1, -1):
    cx = steps[i]['x']
    # gradient from loss into cell top
    ax.annotate('', xy=(cx + 0.15, cell_y + cell_h/2),
                xytext=(cx + 0.15, loss_y - 0.55/2),
                arrowprops=dict(arrowstyle='->', color=bwd_col, lw=1.4,
                                linestyle='dashed'))

# Backward arrows: h₃→h₂, h₂→h₁ (curved below)
for i in range(len(steps) - 1, 0, -1):
    x1 = steps[i]['x'] - cell_w/2 - 0.05
    x2 = steps[i-1]['x'] + cell_w/2 + 0.05
    bwd_arrow(ax, x1, cell_y - 0.1, x2, cell_y - 0.1)
    mx = (x1 + x2) / 2
    ax.text(mx, cell_y - 0.75, '∂hₜ/∂hₜ₋₁',
            ha='center', fontsize=7.5, color=bwd_col)

# ── Gradient accumulation label ───────────────────────────────────────────────
ax.text(5.5, 0.45,
        '∂L/∂W_h = Σₜ ∂Lₜ/∂W_h   (accumulated over all time steps)',
        ha='center', fontsize=9, color=bwd_col,
        bbox=dict(facecolor='#fee2e2', edgecolor='#dc2626',
                  boxstyle='round,pad=0.3', linewidth=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=fwd_col,  edgecolor=fwd_col,  label='Forward pass'),
    mpatches.Patch(facecolor=bwd_col,  edgecolor=bwd_col,  label='Backward pass (BPTT)'),
    mpatches.Patch(facecolor=loss_fill, edgecolor=loss_border, label='Loss Lₜ'),
]
ax.legend(handles=legend_items, loc='upper left', fontsize=8.5,
          framealpha=0.9, edgecolor='#d1d5db')

ax.text(5.5, 6.2,
        'Backpropagation Through Time — gradients flow back through every step',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'rnn_bptt.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved rnn_bptt.png")
