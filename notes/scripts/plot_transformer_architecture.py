import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(12, 7))
fig.patch.set_facecolor('white')

# ── Colours ───────────────────────────────────────────────────────────────────
attn_fill   = '#dbeafe';  attn_border  = '#2563eb'
mattn_fill  = '#fce7f3';  mattn_border = '#9d174d'
cattn_fill  = '#dcfce7';  cattn_border = '#15803d'
ffn_fill    = '#fef3c7';  ffn_border   = '#b45309'
norm_fill   = '#f3f4f6';  norm_border  = '#9ca3af'
io_fill     = '#f9fafb';  io_border    = '#6b7280'
arrow_col   = '#9ca3af'
res_col     = '#374151'
label_col   = '#111111'
sub_col     = '#374151'
dim_col     = '#6b7280'

titles = ['Encoder Layer', 'Decoder Layer']
configs = [
    # Encoder: bottom to top
    [
        {'label': 'Input + Positional\nEncoding', 'fill': io_fill,   'border': io_border},
        {'label': 'Multi-Head\nSelf-Attention',   'fill': attn_fill,  'border': attn_border},
        {'label': 'Add & Norm',                   'fill': norm_fill,  'border': norm_border},
        {'label': 'Feed-Forward\nNetwork (FFN)',  'fill': ffn_fill,   'border': ffn_border},
        {'label': 'Add & Norm',                   'fill': norm_fill,  'border': norm_border},
        {'label': 'Encoder Output',               'fill': io_fill,    'border': io_border},
    ],
    # Decoder: bottom to top
    [
        {'label': 'Output (shifted) +\nPositional Encoding', 'fill': io_fill,    'border': io_border},
        {'label': 'Masked Multi-Head\nSelf-Attention',        'fill': mattn_fill, 'border': mattn_border},
        {'label': 'Add & Norm',                               'fill': norm_fill,  'border': norm_border},
        {'label': 'Multi-Head\nCross-Attention',              'fill': cattn_fill, 'border': cattn_border},
        {'label': 'Add & Norm',                               'fill': norm_fill,  'border': norm_border},
        {'label': 'Feed-Forward\nNetwork (FFN)',              'fill': ffn_fill,   'border': ffn_border},
        {'label': 'Add & Norm',                               'fill': norm_fill,  'border': norm_border},
        {'label': 'Linear + Softmax\n→ Output Logits',        'fill': io_fill,    'border': io_border},
    ],
]

notes = [
    # Encoder annotations
    {1: 'Q=K=V from\nsame sequence', 2: 'residual +\nlayer norm'},
    # Decoder annotations
    {1: 'masked: cannot\nattend to future', 3: 'Q from decoder\nK,V from encoder', 4: 'residual +\nlayer norm'},
]

for ax, title, config, note in zip(axes, titles, configs, notes):
    ax.set_facecolor('white')
    n = len(config)
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.3, n * 1.05 + 0.3)
    ax.axis('off')

    box_w, box_h = 2.8, 0.72
    cx = 2.0
    spacing = 1.05

    for i, layer in enumerate(config):
        cy = i * spacing + box_h / 2
        rect = mpatches.FancyBboxPatch(
            (cx - box_w/2, cy - box_h/2), box_w, box_h,
            boxstyle='round,pad=0.06',
            facecolor=layer['fill'], edgecolor=layer['border'],
            linewidth=1.4, zorder=3
        )
        ax.add_patch(rect)
        ax.text(cx, cy, layer['label'], ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=label_col, zorder=4)

        # Annotation note
        if i in note:
            ax.text(cx + box_w/2 + 0.1, cy, note[i],
                    ha='left', va='center', fontsize=6.8, color=dim_col,
                    style='italic')

        # Arrow to next layer
        if i < n - 1:
            ax.annotate('', xy=(cx, (i+1)*spacing),
                        xytext=(cx, cy + box_h/2),
                        arrowprops=dict(arrowstyle='->', color=arrow_col, lw=1.3))

    # Residual connection bracket (encoder: around attn+norm, ffn+norm)
    # Decoder: around each sub-layer pair
    if title == 'Encoder Layer':
        for start, end in [(0, 2), (2, 4)]:
            y1 = start * spacing + box_h/2
            y2 = (end) * spacing + box_h/2
            bx = cx - box_w/2 - 0.25
            ax.plot([bx, bx], [y1, y2], color=res_col, lw=1.0, zorder=2)
            ax.annotate('', xy=(cx - box_w/2, y2),
                        xytext=(bx, y2),
                        arrowprops=dict(arrowstyle='->', color=res_col, lw=1.0))
            ax.text(bx - 0.08, (y1 + y2)/2, 'residual', ha='right',
                    va='center', fontsize=6.5, color=res_col, rotation=90)

    ax.set_title(title, fontsize=11, fontweight='bold', color=label_col, pad=8)

# ── Cross-attention arrow between encoder and decoder ─────────────────────────
fig.text(0.5, 0.5, '← encoder output\n   (K, V for\n   cross-attention)',
         ha='center', va='center', fontsize=8, color='#15803d',
         bbox=dict(facecolor='#dcfce7', edgecolor='#15803d',
                   boxstyle='round,pad=0.3', linewidth=0.8))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=attn_fill,  edgecolor=attn_border,  label='Self-attention (Q=K=V from same sequence)'),
    mpatches.Patch(facecolor=mattn_fill, edgecolor=mattn_border, label='Masked self-attention (future tokens hidden)'),
    mpatches.Patch(facecolor=cattn_fill, edgecolor=cattn_border, label='Cross-attention (Q from decoder, K/V from encoder)'),
    mpatches.Patch(facecolor=ffn_fill,   edgecolor=ffn_border,   label='Feed-forward network (per position)'),
    mpatches.Patch(facecolor=norm_fill,  edgecolor=norm_border,  label='Add & Norm (residual + layer normalisation)'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=2, fontsize=7.8,
           framealpha=0.95, edgecolor='#d1d5db', bbox_to_anchor=(0.5, 0.0))

fig.suptitle('Transformer Architecture — Encoder Layer and Decoder Layer',
             fontsize=11.5, fontweight='bold', color='#111111', y=1.01)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'transformer_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved transformer_architecture.png")
