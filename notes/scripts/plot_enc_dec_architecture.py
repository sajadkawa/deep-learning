import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.axis('off')

# ── Colours ───────────────────────────────────────────────────────────────────
enc_fill    = '#dbeafe';  enc_border  = '#2563eb'
dec_fill    = '#dcfce7';  dec_border  = '#15803d'
ctx_fill    = '#fef9c3';  ctx_border  = '#ca8a04'
input_fill  = '#f3f4f6';  input_border= '#9ca3af'
output_fill = '#f3f4f6';  output_border='#9ca3af'
arrow_col   = '#9ca3af'
hidden_col  = '#374151'
label_col   = '#111111'
sub_col     = '#374151'
dim_col     = '#6b7280'

cell_w, cell_h = 1.1, 0.8
node_r = 0.24

def box(ax, cx, cy, w, h, fill, border, label, sublabel='', fontsize=9):
    r = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.06',
        facecolor=fill, edgecolor=border, linewidth=1.5, zorder=3
    )
    ax.add_patch(r)
    ax.text(cx, cy + (0.1 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=label_col, zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.18, sublabel,
                ha='center', va='center', fontsize=7.5, color=sub_col, zorder=4)

def circle(ax, cx, cy, r, fill, border, label, fontsize=8.5):
    c = plt.Circle((cx, cy), r, facecolor=fill, edgecolor=border,
                   linewidth=1.2, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=label_col, zorder=4)

def arr(ax, x1, y1, x2, y2, col=arrow_col, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw))

# ── Layout ────────────────────────────────────────────────────────────────────
cell_y   = 3.2
input_y  = 1.5
output_y = 4.9

# Encoder: 3 steps
enc_steps = [
    {'x': 1.3, 'xt': 'x₁', 'ht': 'h₁⁽ᵉ⁾'},
    {'x': 2.8, 'xt': 'x₂', 'ht': 'h₂⁽ᵉ⁾'},
    {'x': 4.3, 'xt': 'x₃', 'ht': 'h₃⁽ᵉ⁾'},
]

# Context vector
ctx_x = 6.1

# Decoder: 2 steps
dec_steps = [
    {'x': 8.2,  'yt_prev': 'y₀\n⟨BOS⟩', 'yt': 'ŷ₁', 'ht': 'h₁⁽ᵈ⁾'},
    {'x': 10.2, 'yt_prev': 'ŷ₁',         'yt': 'ŷ₂', 'ht': 'h₂⁽ᵈ⁾'},
]

# ── Section labels ────────────────────────────────────────────────────────────
ax.text(2.8, 5.65, 'ENCODER', ha='center', fontsize=10.5,
        fontweight='bold', color='#2563eb')
ax.text(9.2, 5.65, 'DECODER', ha='center', fontsize=10.5,
        fontweight='bold', color='#15803d')
ax.axvline(5.35, ymin=0.05, ymax=0.92, color='#d1d5db', lw=1.2, linestyle='--')

# ── Encoder cells ─────────────────────────────────────────────────────────────
for s in enc_steps:
    cx = s['x']
    box(ax, cx, cell_y, cell_w, cell_h, enc_fill, enc_border, 'RNN', 'enc')
    circle(ax, cx, input_y, node_r, input_fill, input_border, s['xt'])
    arr(ax, cx, input_y + node_r, cx, cell_y - cell_h/2, col='#2563eb', lw=1.2)

# Encoder hidden state arrows
arr(ax, 0.4, cell_y, enc_steps[0]['x'] - cell_w/2, cell_y, col=hidden_col, lw=1.4)
ax.text(0.25, cell_y + 0.22, 'h₀=0', ha='center', fontsize=7.5, color=dim_col)

for i in range(len(enc_steps) - 1):
    x1 = enc_steps[i]['x'] + cell_w/2
    x2 = enc_steps[i+1]['x'] - cell_w/2
    arr(ax, x1, cell_y, x2, cell_y, col=hidden_col, lw=1.4)
    mx = (x1 + x2) / 2
    ax.text(mx, cell_y + 0.22, enc_steps[i+1]['ht'],
            ha='center', fontsize=7.5, color=sub_col)

# Arrow from last encoder cell to context vector
arr(ax, enc_steps[-1]['x'] + cell_w/2, cell_y,
    ctx_x - 0.45, cell_y, col=hidden_col, lw=1.6)
ax.text((enc_steps[-1]['x'] + cell_w/2 + ctx_x - 0.45) / 2,
        cell_y + 0.22, 'h₃⁽ᵉ⁾', ha='center', fontsize=7.5, color=sub_col)

# ── Context vector ────────────────────────────────────────────────────────────
box(ax, ctx_x, cell_y, 0.9, 0.75, ctx_fill, ctx_border, 'c', fontsize=11)
ax.text(ctx_x, cell_y - 0.62, 'context\nvector', ha='center',
        fontsize=7.5, color='#92400e')

# Arrow from context vector to decoder init
arr(ax, ctx_x + 0.45, cell_y, dec_steps[0]['x'] - cell_w/2 - 0.05,
    cell_y, col='#ca8a04', lw=1.8)
ax.text((ctx_x + 0.45 + dec_steps[0]['x'] - cell_w/2) / 2,
        cell_y + 0.25, 'h₀⁽ᵈ⁾= c',
        ha='center', fontsize=8, color='#92400e', fontweight='bold')

# ── Decoder cells ─────────────────────────────────────────────────────────────
for s in dec_steps:
    cx = s['x']
    box(ax, cx, cell_y, cell_w, cell_h, dec_fill, dec_border, 'RNN', 'dec')

    # Input from below (previous output)
    circle(ax, cx, input_y, node_r, input_fill, input_border, '')
    ax.text(cx, input_y, s['yt_prev'], ha='center', va='center',
            fontsize=6.5, color=label_col, zorder=4)
    arr(ax, cx, input_y + node_r, cx, cell_y - cell_h/2, col='#15803d', lw=1.2)

    # Output above
    circle(ax, cx, output_y, node_r, output_fill, output_border, s['yt'])
    arr(ax, cx, cell_y + cell_h/2, cx, output_y - node_r, col='#15803d', lw=1.2)

# Decoder hidden state arrows
for i in range(len(dec_steps) - 1):
    x1 = dec_steps[i]['x'] + cell_w/2
    x2 = dec_steps[i+1]['x'] - cell_w/2
    arr(ax, x1, cell_y, x2, cell_y, col=hidden_col, lw=1.4)
    mx = (x1 + x2) / 2
    ax.text(mx, cell_y + 0.22, dec_steps[i+1]['ht'],
            ha='center', fontsize=7.5, color=sub_col)

# Autoregressive feedback arrow: ŷ₁ → input of step 2
ax.annotate('', xy=(dec_steps[1]['x'], input_y + node_r),
            xytext=(dec_steps[0]['x'] + node_r + 0.05, output_y),
            arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.0,
                            connectionstyle='arc3,rad=-0.35'))
ax.text(9.55, 3.85, 'autoregressive\nfeedback', ha='center',
        fontsize=6.5, color=dim_col)

# EOS hint
ax.text(dec_steps[-1]['x'] + cell_w/2 + 0.15, cell_y, '…',
        ha='left', va='center', fontsize=13, color='#9ca3af')
ax.text(dec_steps[-1]['x'] + cell_w/2 + 0.7, cell_y - 0.3,
        'until\n⟨EOS⟩', ha='center', fontsize=7, color=dim_col)

# ── Row labels ────────────────────────────────────────────────────────────────
ax.text(0.15, input_y,  'input\nxₜ',  ha='center', fontsize=7.5, color=dim_col, va='center')
ax.text(0.15, output_y, 'output\nŷₜ', ha='center', fontsize=7.5, color=dim_col, va='center')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=enc_fill, edgecolor=enc_border, label='Encoder RNN'),
    mpatches.Patch(facecolor=ctx_fill, edgecolor=ctx_border, label='Context vector  c = h_T^(e)'),
    mpatches.Patch(facecolor=dec_fill, edgecolor=dec_border, label='Decoder RNN'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=8.5,
          framealpha=0.95, edgecolor='#d1d5db')

ax.text(6.5, 5.65,
        'Encoder-Decoder Architecture — variable input length T, variable output length T\'',
        ha='center', va='center', fontsize=10.5, fontweight='bold', color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'enc_dec_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved enc_dec_architecture.png")
