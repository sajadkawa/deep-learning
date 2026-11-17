import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 9)
ax.set_ylim(0, 6)
ax.axis('off')

# ── Colours ───────────────────────────────────────────────────────────────────
reset_fill   = '#fce7f3';  reset_border  = '#9d174d'
update_fill  = '#dbeafe';  update_border = '#1d4ed8'
cand_fill    = '#fef3c7';  cand_border   = '#b45309'
arrow_col    = '#9ca3af'
hidden_col   = '#374151'
input_col    = '#2563eb'
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

def arr(ax, x1, y1, x2, y2, col=arrow_col, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw))

def circle(ax, cx, cy, r, fill, border, label, fontsize=10):
    c = plt.Circle((cx, cy), r, facecolor=fill, edgecolor=border,
                   linewidth=1.3, zorder=4)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=label_col, zorder=5)

# ── Layout ────────────────────────────────────────────────────────────────────
hidden_y = 1.2    # h_{t-1} / h_t row
gate_y   = 3.0    # reset and update gate row
cand_y   = 3.0    # candidate tanh same row
interp_y = 4.8    # interpolation row

xr = 2.0   # reset gate
xc = 4.5   # candidate tanh
xz = 7.0   # update gate

cell_w, cell_h = 1.1, 0.75
mul_r = 0.22

# ── h_{t-1} input bus ─────────────────────────────────────────────────────────
arr(ax, 0.2, hidden_y, xr - cell_w/2 - 0.1, hidden_y, col=hidden_col, lw=1.8)
ax.text(0.05, hidden_y + 0.25, 'h_{t-1}', fontsize=9, color=hidden_col, fontweight='bold')
ax.plot([xr - cell_w/2 - 0.1, xz + cell_w/2 + 0.1], [hidden_y, hidden_y],
        color=hidden_col, lw=1.6, zorder=2)

# h_{t-1} feeds reset gate and update gate
for gx in [xr, xz]:
    arr(ax, gx, hidden_y, gx, gate_y - cell_h/2, col=hidden_col, lw=1.2)

# h_{t-1} also feeds interpolation directly (bypass)
ax.plot([xz + cell_w/2 + 0.1, xz + cell_w/2 + 0.1], [hidden_y, interp_y],
        color=hidden_col, lw=1.4, zorder=2)
arr(ax, xz + cell_w/2 + 0.1, interp_y, xz + 0.3, interp_y,
    col=hidden_col, lw=1.4)

# ── x_t input (dotted, feeds reset, update, candidate) ───────────────────────
for gx in [xr, xc, xz]:
    ax.plot([gx, gx], [0.35, hidden_y - 0.01],
            color=input_col, lw=1.0, linestyle='dotted', zorder=1)
arr(ax, 0.2, 0.35, 4.8, 0.35, col=input_col, lw=1.3)
ax.text(0.05, 0.35, 'x_t', fontsize=9, color=input_col, fontweight='bold', va='center')
ax.text(4.5, 0.05, '(dotted: x_t feeds reset gate, candidate, update gate)',
        ha='center', fontsize=7.5, color=dim_col)

# ── Gate boxes ────────────────────────────────────────────────────────────────
box(ax, xr, gate_y, cell_w, cell_h, reset_fill,  reset_border,  'σ', 'reset\ngate rₜ')
box(ax, xz, gate_y, cell_w, cell_h, update_fill, update_border, 'σ', 'update\ngate zₜ')

# ── Multiply circle: r_t ⊙ h_{t-1} ──────────────────────────────────────────
mul_r_x = (xr + xc) / 2 - 0.2
mul_r_y = gate_y
circle(ax, mul_r_x, mul_r_y, mul_r, '#fce7f3', '#9d174d', '×')

arr(ax, xr + cell_w/2, gate_y, mul_r_x - mul_r, mul_r_y, col=reset_border, lw=1.2)
# h_{t-1} branch down to multiply
ax.plot([xr, xr], [hidden_y, gate_y - cell_h/2 - 0.01], color=hidden_col, lw=0.0)  # already drawn
ax.annotate('', xy=(mul_r_x - 0.05, mul_r_y - mul_r),
            xytext=(mul_r_x - 0.05, hidden_y + 0.01),
            arrowprops=dict(arrowstyle='->', color=hidden_col, lw=1.0,
                            connectionstyle='arc3,rad=0.0'))

# ── Candidate tanh ────────────────────────────────────────────────────────────
box(ax, xc, cand_y, cell_w, cell_h, cand_fill, cand_border, 'tanh', 'candidate\nĥₜ')
arr(ax, mul_r_x + mul_r, mul_r_y, xc - cell_w/2, cand_y, col=cand_border, lw=1.2)

# ── Interpolation row ─────────────────────────────────────────────────────────
# (1 - z_t) ⊙ h_{t-1}  +  z_t ⊙ h̃_t
# Two multiply circles and one add circle

mul_1_x = xr + 0.3   # (1-z) × h_{t-1}
mul_2_x = xc + 0.3   # z × h̃_t
add_x   = (mul_1_x + mul_2_x) / 2 + 0.5

circle(ax, mul_1_x, interp_y, mul_r, '#dbeafe', '#1d4ed8', '×')
circle(ax, mul_2_x, interp_y, mul_r, '#dbeafe', '#1d4ed8', '×')
circle(ax, add_x,   interp_y, mul_r, '#fef9c3', '#ca8a04', '+', fontsize=11)

# (1-z_t) label
ax.text(mul_1_x, interp_y + 0.38, '(1−zₜ)⊙',
        ha='center', fontsize=7.5, color=update_border)

# z_t label
ax.text(mul_2_x, interp_y + 0.38, 'zₜ⊙',
        ha='center', fontsize=7.5, color=update_border)

# update gate → both multiply circles
arr(ax, xz, gate_y + cell_h/2, mul_1_x + 0.05, interp_y - mul_r,
    col=update_border, lw=1.2)
arr(ax, xz, gate_y + cell_h/2, mul_2_x - 0.05, interp_y - mul_r,
    col=update_border, lw=1.2)

# candidate → mul_2
arr(ax, xc, cand_y + cell_h/2, mul_2_x, interp_y - mul_r, col=cand_border, lw=1.2)

# h_{t-1} bypass → mul_1
# already drawn the bypass line above; connect it
arr(ax, mul_1_x - 0.3, interp_y, mul_1_x - mul_r, interp_y,
    col=hidden_col, lw=1.2)
ax.plot([xz + cell_w/2 + 0.1, mul_1_x - 0.3], [interp_y, interp_y],
        color=hidden_col, lw=1.4, zorder=2)

# mul_1 → add
arr(ax, mul_1_x + mul_r, interp_y, add_x - mul_r, interp_y, col=arrow_col, lw=1.2)
# mul_2 → add
arr(ax, mul_2_x + mul_r, interp_y, add_x - mul_r, interp_y, col=arrow_col, lw=1.2)

# add → h_t output
arr(ax, add_x + mul_r, interp_y, add_x + 1.0, interp_y, col='#15803d', lw=1.8)
ax.text(add_x + 1.1, interp_y + 0.25, 'h_t', fontsize=9,
        color='#15803d', fontweight='bold')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=reset_fill,  edgecolor=reset_border,  label='Reset gate   rₜ = σ(Wr hₜ₋₁ + Ur xₜ + br)'),
    mpatches.Patch(facecolor=update_fill, edgecolor=update_border, label='Update gate  zₜ = σ(Wz hₜ₋₁ + Uz xₜ + bz)'),
    mpatches.Patch(facecolor=cand_fill,   edgecolor=cand_border,   label='Candidate    ĥₜ = tanh(Wh (rₜ⊙hₜ₋₁) + Uh xₜ + bh)'),
    mpatches.Patch(facecolor='white',     edgecolor='#15803d',     label='Output       hₜ = (1−zₜ)⊙hₜ₋₁ + zₜ⊙ĥₜ'),
]
ax.legend(handles=legend_items, loc='upper left', fontsize=7.8,
          framealpha=0.95, edgecolor='#d1d5db', bbox_to_anchor=(0.0, 1.0))

ax.text(4.5, 5.75, 'GRU Cell — two gates, one state vector',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'gru_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved gru_architecture.png")
