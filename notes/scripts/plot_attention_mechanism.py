import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')

# ── Colours ───────────────────────────────────────────────────────────────────
q_fill  = '#dbeafe';  q_border  = '#2563eb'
k_fill  = '#fce7f3';  k_border  = '#9d174d'
v_fill  = '#dcfce7';  v_border  = '#15803d'
s_fill  = '#fef9c3';  s_border  = '#ca8a04'
o_fill  = '#f3f4f6';  o_border  = '#6b7280'
arrow_col = '#9ca3af'
label_col = '#111111'
sub_col   = '#374151'
dim_col   = '#6b7280'

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

def arr(ax, x1, y1, x2, y2, col=arrow_col, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col, lw=lw))

# ── Input vectors ─────────────────────────────────────────────────────────────
# Three key/value positions on the left, one query on the right
positions = [
    {'y': 4.5, 'k': '[1.0, 0.0]', 'v': '[1.0, 0.0]'},
    {'y': 3.0, 'k': '[0.5, 0.5]', 'v': '[0.5, 0.5]'},
    {'y': 1.5, 'k': '[0.0, 1.0]', 'v': '[0.0, 1.0]'},
]
alpha = [0.430, 0.325, 0.245]

kv_x = 1.5
score_x = 3.8
softmax_x = 5.8
weight_x = 7.5
out_x = 9.5

query_y = 3.0
query_x = 3.8

# ── Keys ──────────────────────────────────────────────────────────────────────
for p in positions:
    box(ax, kv_x, p['y'], 1.6, 0.55, k_fill, k_border,
        f"k = {p['k']}", fontsize=8)

ax.text(kv_x, 5.5, 'Keys  K', ha='center', fontsize=9,
        fontweight='bold', color='#9d174d')

# ── Query ─────────────────────────────────────────────────────────────────────
box(ax, query_x, query_y, 1.6, 0.55, q_fill, q_border,
    'q = [0.9, 0.1]', fontsize=8)
ax.text(query_x, query_y - 0.55, 'Query  Q', ha='center', fontsize=9,
        fontweight='bold', color='#2563eb')

# ── Score computation ─────────────────────────────────────────────────────────
scores_raw = [0.636, 0.354, 0.071]
for i, p in enumerate(positions):
    # Arrow: key → score
    arr(ax, kv_x + 0.8, p['y'], score_x - 0.45, p['y'], col='#9d174d', lw=1.0)
    # Arrow: query → score (diagonal)
    arr(ax, query_x + 0.0, query_y + 0.28 * (1 if p['y'] > query_y else (-1 if p['y'] < query_y else 0)),
        score_x - 0.45, p['y'], col='#2563eb', lw=0.8)
    box(ax, score_x, p['y'], 0.9, 0.5, s_fill, s_border,
        f'{scores_raw[i]:.3f}', fontsize=8.5)

ax.text(score_x, 5.5, 'Scores\nqKᵀ/√dₖ', ha='center', fontsize=8.5,
        fontweight='bold', color='#92400e')

# ── Softmax ───────────────────────────────────────────────────────────────────
# Bracket connecting all scores to softmax box
ax.plot([score_x + 0.45, softmax_x - 0.55, softmax_x - 0.55],
        [4.5, 4.5, 1.5], color='#ca8a04', lw=1.0, zorder=2)
arr(ax, softmax_x - 0.55, 3.0, softmax_x - 0.45, 3.0, col='#ca8a04', lw=1.2)

box(ax, softmax_x, 3.0, 1.0, 0.6, s_fill, s_border, 'softmax', fontsize=8.5)
ax.text(softmax_x, 5.5, 'Attention\nWeights α', ha='center', fontsize=8.5,
        fontweight='bold', color='#92400e')

# ── Weights ───────────────────────────────────────────────────────────────────
for i, p in enumerate(positions):
    arr(ax, softmax_x + 0.5, 3.0 + 0.3*(i-1)*0,
        weight_x - 0.55, p['y'], col='#ca8a04', lw=1.0)
    box(ax, weight_x, p['y'], 0.9, 0.5, s_fill, s_border,
        f'α={alpha[i]:.3f}', fontsize=8)

# ── Values ────────────────────────────────────────────────────────────────────
val_x = 8.6
for p in positions:
    box(ax, val_x, p['y'], 1.4, 0.5, v_fill, v_border,
        f"v = {p['v']}", fontsize=8)
    # multiply arrow
    arr(ax, weight_x + 0.45, p['y'], val_x - 0.7, p['y'],
        col='#15803d', lw=1.0)

ax.text(val_x, 5.5, 'Values  V', ha='center', fontsize=9,
        fontweight='bold', color='#15803d')

# ── Output ────────────────────────────────────────────────────────────────────
ax.plot([val_x + 0.7, out_x - 0.05, out_x - 0.05],
        [4.5, 4.5, 1.5], color='#15803d', lw=1.0, zorder=2)
arr(ax, out_x - 0.05, 3.0, out_x + 0.05, 3.0, col='#15803d', lw=1.2)

box(ax, out_x + 0.5, 3.0, 1.0, 0.6, o_fill, o_border,
    'out\n[0.593\n 0.408]', fontsize=7.5)
ax.text(out_x + 0.5, 5.5, 'Output\nΣ αᵢvᵢ', ha='center', fontsize=8.5,
        fontweight='bold', color='#374151')

# ── Formula box ───────────────────────────────────────────────────────────────
ax.text(5.5, 0.45,
        'Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V',
        ha='center', fontsize=9.5, color='#111111',
        bbox=dict(facecolor='#fef9c3', edgecolor='#ca8a04',
                  boxstyle='round,pad=0.3', linewidth=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=q_fill, edgecolor=q_border, label='Query Q — what am I looking for?'),
    mpatches.Patch(facecolor=k_fill, edgecolor=k_border, label='Keys K — what does each position offer?'),
    mpatches.Patch(facecolor=v_fill, edgecolor=v_border, label='Values V — what does each position contain?'),
    mpatches.Patch(facecolor=s_fill, edgecolor=s_border, label='Attention weights α (softmax scores)'),
]
ax.legend(handles=legend_items, loc='upper left', fontsize=7.8,
          framealpha=0.95, edgecolor='#d1d5db', bbox_to_anchor=(0.0, 1.0))

ax.text(5.5, 5.75,
        'Scaled Dot-Product Attention',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'attention_mechanism.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved attention_mechanism.png")
