import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)
ax.axis('off')

# ── Colours ───────────────────────────────────────────────────────────────────
gate_fills   = {'f': '#fce7f3', 'i': '#dbeafe', 'c': '#fef3c7', 'o': '#dcfce7'}
gate_borders = {'f': '#9d174d', 'i': '#1d4ed8', 'c': '#b45309', 'o': '#15803d'}
cell_fill    = '#f9fafb'
cell_border  = '#6b7280'
arrow_col    = '#9ca3af'
highway_col  = '#374151'
label_col    = '#111111'
sub_col      = '#374151'
dim_col      = '#6b7280'

def box(ax, cx, cy, w, h, fill, border, label, sublabel='', fontsize=9):
    r = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.06',
        facecolor=fill, edgecolor=border, linewidth=1.5, zorder=4
    )
    ax.add_patch(r)
    ax.text(cx, cy + (0.12 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=label_col, zorder=5)
    if sublabel:
        ax.text(cx, cy - 0.18, sublabel,
                ha='center', va='center', fontsize=7.5, color=sub_col, zorder=5)

def arr(ax, x1, y1, x2, y2, col=arrow_col, lw=1.4, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col, lw=lw))

def circle(ax, cx, cy, r, fill, border, label, fontsize=9):
    c = plt.Circle((cx, cy), r, facecolor=fill, edgecolor=border,
                   linewidth=1.3, zorder=4)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=label_col, zorder=5)

# ── Layout constants ──────────────────────────────────────────────────────────
highway_y = 5.5      # cell state highway
hidden_y  = 1.4      # hidden state row
gate_y    = 3.4      # gate row
cand_y    = 3.4      # candidate tanh same row

# Gate x positions
xf = 2.2   # forget gate
xi = 4.2   # input gate
xc = 6.2   # candidate tanh
xo = 8.2   # output gate

cell_w, cell_h = 1.1, 0.75
node_r = 0.22
mul_r  = 0.22

# ── Cell state highway (top) ──────────────────────────────────────────────────
# c_{t-1} entering from left
arr(ax, 0.3, highway_y, 1.5, highway_y, col=highway_col, lw=2.0)
ax.text(0.15, highway_y + 0.25, 'c_{t-1}', fontsize=9, color=highway_col, fontweight='bold')

# c_t exiting right
arr(ax, 9.5, highway_y, 10.7, highway_y, col=highway_col, lw=2.0)
ax.text(10.75, highway_y + 0.25, 'c_t', fontsize=9, color=highway_col, fontweight='bold')

# Highway line segments between operations
ax.plot([1.5, xf - mul_r], [highway_y, highway_y], color=highway_col, lw=2.0, zorder=2)
ax.plot([xf + mul_r, xi + mul_r + 0.3], [highway_y, highway_y], color=highway_col, lw=2.0, zorder=2)
ax.plot([xi + mul_r + 0.3, 9.5], [highway_y, highway_y], color=highway_col, lw=2.0, zorder=2)

# ── Multiply circle: c_{t-1} × f_t (forget) ──────────────────────────────────
circle(ax, xf, highway_y, mul_r, '#fce7f3', '#9d174d', '×', fontsize=10)

# ── Add circle: + i_t * c̃_t ──────────────────────────────────────────────────
add_x = xi + mul_r + 0.3
circle(ax, add_x, highway_y, mul_r, '#fef9c3', '#ca8a04', '+', fontsize=11)

# ── Gate boxes ────────────────────────────────────────────────────────────────
box(ax, xf, gate_y, cell_w, cell_h, gate_fills['f'], gate_borders['f'], 'σ', 'forget\ngate f_t')
box(ax, xi, gate_y, cell_w, cell_h, gate_fills['i'], gate_borders['i'], 'σ', 'input\ngate i_t')
box(ax, xc, gate_y, cell_w, cell_h, gate_fills['c'], gate_borders['c'], 'tanh', 'candidate\nc̃_t')
box(ax, xo, gate_y, cell_w, cell_h, gate_fills['o'], gate_borders['o'], 'σ', 'output\ngate o_t')

# ── Multiply circle: i_t × c̃_t → add circle ─────────────────────────────────
mul_ic_x = (xi + xc) / 2 + 0.1
mul_ic_y = gate_y + 1.1
circle(ax, mul_ic_x, mul_ic_y, mul_r, '#fef9c3', '#ca8a04', '×', fontsize=10)

arr(ax, xi, gate_y + cell_h/2, mul_ic_x - 0.1, mul_ic_y - mul_r, col=gate_borders['i'], lw=1.2)
arr(ax, xc, gate_y + cell_h/2, mul_ic_x + 0.1, mul_ic_y - mul_r, col=gate_borders['c'], lw=1.2)
arr(ax, mul_ic_x, mul_ic_y + mul_r, add_x - mul_r, highway_y, col=arrow_col, lw=1.2)

# ── Forget gate → multiply on highway ────────────────────────────────────────
arr(ax, xf, gate_y + cell_h/2, xf, highway_y - mul_r, col=gate_borders['f'], lw=1.2)

# ── tanh(c_t) → multiply with o_t → h_t ─────────────────────────────────────
tanh_ct_x = xo - 0.9
tanh_ct_y = highway_y - 1.1
box(ax, tanh_ct_x, tanh_ct_y, 0.85, 0.6, '#f3f4f6', '#9ca3af', 'tanh', fontsize=8)

# c_t down to tanh box
arr(ax, tanh_ct_x + 0.1, highway_y, tanh_ct_x + 0.1, tanh_ct_y + 0.3,
    col=highway_col, lw=1.4)

# multiply circle: o_t × tanh(c_t)
mul_o_x = xo - 0.1
mul_o_y = tanh_ct_y
circle(ax, mul_o_x, mul_o_y, mul_r, '#dcfce7', '#15803d', '×', fontsize=10)

arr(ax, tanh_ct_x + 0.43, tanh_ct_y, mul_o_x - mul_r, mul_o_y, col=arrow_col, lw=1.2)
arr(ax, xo, gate_y + cell_h/2, xo, mul_o_y + mul_r, col=gate_borders['o'], lw=1.2)

# h_t output downward
arr(ax, mul_o_x, mul_o_y - mul_r, mul_o_x, hidden_y + 0.22,
    col='#15803d', lw=1.8)
ax.text(mul_o_x + 0.2, (mul_o_y - mul_r + hidden_y + 0.22) / 2,
        'h_t', fontsize=9, color='#15803d', fontweight='bold')

# h_t circle
circle(ax, mul_o_x, hidden_y, 0.22, '#dcfce7', '#15803d', 'h_t', fontsize=8)

# ── Input row: h_{t-1} and x_t feed all gates ────────────────────────────────
# h_{t-1} from left
arr(ax, 0.3, hidden_y, xf - cell_w/2 - 0.1, hidden_y, col='#374151', lw=1.6)
ax.text(0.15, hidden_y + 0.25, 'h_{t-1}', fontsize=9, color='#374151', fontweight='bold')

# horizontal bus at hidden_y feeding all gates
ax.plot([xf - cell_w/2 - 0.1, xo + cell_w/2], [hidden_y, hidden_y],
        color='#374151', lw=1.4, zorder=2)

for gx in [xf, xi, xc, xo]:
    arr(ax, gx, hidden_y, gx, gate_y - cell_h/2, col='#374151', lw=1.2)

# x_t from below
input_y_pos = 0.5
for gx in [xf, xi, xc, xo]:
    ax.plot([gx, gx], [input_y_pos + 0.15, hidden_y - 0.01],
            color='#2563eb', lw=1.0, linestyle='dotted', zorder=1)

ax.annotate('', xy=(5.3, input_y_pos + 0.15), xytext=(0.5, input_y_pos + 0.15),
            arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.3))
ax.text(0.15, input_y_pos + 0.15, 'x_t', fontsize=9, color='#2563eb', fontweight='bold',
        va='center')
ax.text(5.5, input_y_pos - 0.1,
        '(dotted lines: x_t feeds all four computations)',
        ha='center', fontsize=7.5, color=dim_col)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=gate_fills['f'], edgecolor=gate_borders['f'], label='Forget gate  fₜ = σ(Wf hₜ₋₁ + Uf xₜ + bf)'),
    mpatches.Patch(facecolor=gate_fills['i'], edgecolor=gate_borders['i'], label='Input gate   iₜ = σ(Wi hₜ₋₁ + Ui xₜ + bi)'),
    mpatches.Patch(facecolor=gate_fills['c'], edgecolor=gate_borders['c'], label='Candidate    c̃ₜ = tanh(Wc hₜ₋₁ + Uc xₜ + bc)'),
    mpatches.Patch(facecolor=gate_fills['o'], edgecolor=gate_borders['o'], label='Output gate  oₜ = σ(Wo hₜ₋₁ + Uo xₜ + bo)'),
    mpatches.Patch(facecolor='white', edgecolor=highway_col, label='Cell state   cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=7.8,
          framealpha=0.95, edgecolor='#d1d5db', bbox_to_anchor=(0.0, 0.0))

ax.text(5.5, 6.7, 'LSTM Cell — cell state highway (top) and three gates',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'lstm_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved lstm_architecture.png")
