import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 11)
ax.set_ylim(0, 7)
ax.axis('off')

# ── Colours ───────────────────────────────────────────────────────────────────
ctrl_fill   = '#fef3c7';  ctrl_border  = '#b45309'
mem_fill    = '#dbeafe';  mem_border   = '#2563eb'
read_fill   = '#dcfce7';  read_border  = '#15803d'
write_fill  = '#fce7f3';  write_border = '#9d174d'
io_fill     = '#f3f4f6';  io_border    = '#9ca3af'
arrow_col   = '#9ca3af'
attn_col    = '#6b7280'
label_col   = '#111111'
sub_col     = '#374151'
dim_col     = '#6b7280'

def box(ax, cx, cy, w, h, fill, border, label, sublabel='', fontsize=9):
    r = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.07',
        facecolor=fill, edgecolor=border, linewidth=1.5, zorder=3
    )
    ax.add_patch(r)
    ax.text(cx, cy + (0.12 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=label_col, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.2, sublabel,
                ha='center', va='center', fontsize=7.5, color=sub_col, zorder=4)

def arr(ax, x1, y1, x2, y2, col=arrow_col, lw=1.4, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=col, lw=lw))

# ── Memory matrix ─────────────────────────────────────────────────────────────
mem_cx, mem_cy = 5.5, 5.0
mem_w, mem_h   = 3.2, 1.6

# Draw memory as a grid of slots
n_slots, slot_w, slot_h = 5, 0.56, 0.55
slot_y = mem_cy
slot_xs = [mem_cx - (n_slots-1)*slot_w/2 + i*slot_w for i in range(n_slots)]

for i, sx in enumerate(slot_xs):
    alpha = 0.3 + 0.5 * (1 - abs(i - 1) / n_slots)   # highlight slot 1
    rect = mpatches.FancyBboxPatch(
        (sx - slot_w/2 + 0.03, slot_y - slot_h/2), slot_w - 0.06, slot_h,
        boxstyle='round,pad=0.03',
        facecolor=mem_fill, edgecolor=mem_border,
        linewidth=1.2, alpha=0.5 + 0.5*(i==1), zorder=3
    )
    ax.add_patch(rect)
    ax.text(sx, slot_y, f'M{i+1}', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#1d4ed8', zorder=4)

ax.text(mem_cx, mem_cy + slot_h/2 + 0.3, 'External Memory  M ∈ ℝᴺˣᵈ',
        ha='center', fontsize=9.5, fontweight='bold', color='#1d4ed8')
ax.text(mem_cx, mem_cy - slot_h/2 - 0.22,
        'N memory slots, each d-dimensional',
        ha='center', fontsize=7.5, color=dim_col)

# ── Controller ────────────────────────────────────────────────────────────────
ctrl_cx, ctrl_cy = 5.5, 2.5
box(ax, ctrl_cx, ctrl_cy, 2.6, 0.9, ctrl_fill, ctrl_border,
    'Controller', '(LSTM or feedforward)', fontsize=10)

# ── Input / Output ────────────────────────────────────────────────────────────
box(ax, 1.2, 2.5, 1.4, 0.65, io_fill, io_border, 'Input xₜ', fontsize=9)
box(ax, 9.8, 2.5, 1.4, 0.65, io_fill, io_border, 'Output yₜ', fontsize=9)

arr(ax, 1.9, 2.5, ctrl_cx - 1.3, 2.5, col=arrow_col, lw=1.4)
arr(ax, ctrl_cx + 1.3, 2.5, 9.1, 2.5, col=arrow_col, lw=1.4)

# ── Read head ─────────────────────────────────────────────────────────────────
read_cx, read_cy = 3.8, 3.8
box(ax, read_cx, read_cy, 1.5, 0.65, read_fill, read_border,
    'Read Head', fontsize=8.5)

# Controller → read head (query)
arr(ax, ctrl_cx - 0.6, ctrl_cy + 0.45, read_cx + 0.2, read_cy - 0.33,
    col=ctrl_border, lw=1.2)
ax.text(3.6, 3.1, 'query qₜ', ha='center', fontsize=7.5,
        color=ctrl_border, style='italic')

# Read head → memory (attention)
arr(ax, read_cx + 0.3, read_cy + 0.33, slot_xs[1] - 0.1, slot_y - slot_h/2,
    col=read_border, lw=1.3)
ax.text(3.5, 4.45, 'α = softmax(qₜKᵀ/√d)',
        ha='center', fontsize=7.5, color=read_border, style='italic')

# Memory → controller (read vector)
arr(ax, slot_xs[0] - 0.1, slot_y - slot_h/2,
    ctrl_cx - 1.0, ctrl_cy + 0.45, col=read_border, lw=1.3)
ax.text(3.2, 4.0, 'rₜ = ΣᵢαᵢMᵢ',
        ha='left', fontsize=7.5, color=read_border, style='italic')

# ── Write head ────────────────────────────────────────────────────────────────
write_cx, write_cy = 7.2, 3.8
box(ax, write_cx, write_cy, 1.5, 0.65, write_fill, write_border,
    'Write Head', fontsize=8.5)

# Controller → write head
arr(ax, ctrl_cx + 0.6, ctrl_cy + 0.45, write_cx - 0.2, write_cy - 0.33,
    col=ctrl_border, lw=1.2)
ax.text(7.4, 3.1, 'erase eₜ\nadd aₜ', ha='center', fontsize=7.5,
        color=ctrl_border, style='italic')

# Write head → memory
arr(ax, write_cx - 0.3, write_cy + 0.33, slot_xs[3] + 0.1, slot_y - slot_h/2,
    col=write_border, lw=1.3)
ax.text(7.5, 4.45, 'Mᵢ ← Mᵢ(1−αᵢeₜ) + αᵢaₜ',
        ha='center', fontsize=7.5, color=write_border, style='italic')

# ── Previous hidden state loop ────────────────────────────────────────────────
ax.annotate('', xy=(ctrl_cx - 1.3, ctrl_cy - 0.2),
            xytext=(ctrl_cx + 1.3, ctrl_cy - 0.2),
            arrowprops=dict(arrowstyle='->', color=ctrl_border, lw=1.0,
                            connectionstyle='arc3,rad=0.4'))
ax.text(ctrl_cx, ctrl_cy - 0.85, 'hₜ₋₁ (hidden state)',
        ha='center', fontsize=7.5, color=ctrl_border, style='italic')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=ctrl_fill,  edgecolor=ctrl_border,  label='Controller — produces queries, erase/add vectors'),
    mpatches.Patch(facecolor=mem_fill,   edgecolor=mem_border,   label='External memory M — N slots of size d'),
    mpatches.Patch(facecolor=read_fill,  edgecolor=read_border,  label='Read head — soft attention lookup → rₜ'),
    mpatches.Patch(facecolor=write_fill, edgecolor=write_border, label='Write head — erase + add via attention weights'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=2, fontsize=7.8,
          framealpha=0.95, edgecolor='#d1d5db', bbox_to_anchor=(0.5, 0.0))

ax.text(5.5, 6.65,
        'Memory-Augmented Network — controller with external memory and attention-based read/write',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'memory_augmented_network.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved memory_augmented_network.png")
