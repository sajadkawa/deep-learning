import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Layout: 5 task types in a single row ─────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(14, 4.2))
fig.patch.set_facecolor('white')

input_fill   = '#dbeafe'
input_border = '#2563eb'
cell_fill    = '#fef3c7'
cell_border  = '#b45309'
output_fill  = '#dcfce7'
output_border= '#15803d'
arrow_col    = '#9ca3af'
label_col    = '#111111'
sub_col      = '#374151'

node_r = 0.28

tasks = [
    {
        'title': 'One-to-One',
        'subtitle': '(feedforward)',
        'inputs':  [1],
        'cells':   [1],
        'outputs': [1],
    },
    {
        'title': 'One-to-Many',
        'subtitle': '(image captioning)',
        'inputs':  [1, 0, 0],
        'cells':   [1, 1, 1],
        'outputs': [0, 1, 1],
    },
    {
        'title': 'Many-to-One',
        'subtitle': '(sentiment analysis)',
        'inputs':  [1, 1, 1],
        'cells':   [1, 1, 1],
        'outputs': [0, 0, 1],
    },
    {
        'title': 'Many-to-Many',
        'subtitle': '(POS tagging)',
        'inputs':  [1, 1, 1],
        'cells':   [1, 1, 1],
        'outputs': [1, 1, 1],
    },
    {
        'title': 'Enc–Dec',
        'subtitle': '(translation)',
        'inputs':  [1, 1, 0, 0],
        'cells':   [1, 1, 1, 1],
        'outputs': [0, 0, 1, 1],
        'enc_dec_split': 2,   # index where decoder starts
    },
]

def draw_node(ax, cx, cy, fill, border, label='', r=node_r):
    c = plt.Circle((cx, cy), r, facecolor=fill, edgecolor=border,
                   linewidth=1.2, zorder=3)
    ax.add_patch(c)
    if label:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color=label_col, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=arrow_col, lw=1.2))

for ax, task in zip(axes, tasks):
    ax.set_facecolor('white')
    n = len(task['cells'])
    xs = [i * 1.1 for i in range(n)]
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(-0.3, 3.2)
    ax.axis('off')

    cell_y   = 1.6
    input_y  = 0.5
    output_y = 2.7

    enc_dec_split = task.get('enc_dec_split', None)

    for i, x in enumerate(xs):
        # RNN cell — always present
        draw_node(ax, x, cell_y, cell_fill, cell_border, r=node_r + 0.04)

        # Hidden state arrow to next cell
        if i < n - 1:
            draw_arrow(ax, x + node_r + 0.04, cell_y,
                       xs[i+1] - node_r - 0.04, cell_y)

        # Input node
        if task['inputs'][i]:
            draw_node(ax, x, input_y, input_fill, input_border)
            draw_arrow(ax, x, input_y + node_r, x, cell_y - node_r - 0.04)

        # Output node
        if task['outputs'][i]:
            draw_node(ax, x, output_y, output_fill, output_border)
            draw_arrow(ax, x, cell_y + node_r + 0.04, x, output_y - node_r)

    # Encoder/decoder divider
    if enc_dec_split is not None:
        split_x = (xs[enc_dec_split - 1] + xs[enc_dec_split]) / 2
        ax.axvline(split_x, ymin=0.05, ymax=0.95,
                   color='#d1d5db', linewidth=1.2, linestyle='--', zorder=1)
        ax.text(xs[enc_dec_split - 1] / 2 + xs[0] / 2 - 0.1, -0.15,
                'Enc', ha='center', fontsize=7, color=sub_col)
        ax.text((xs[enc_dec_split] + xs[-1]) / 2, -0.15,
                'Dec', ha='center', fontsize=7, color=sub_col)

    ax.set_title(task['title'], fontsize=9.5, fontweight='bold',
                 color=label_col, pad=4)
    ax.text((xs[0] + xs[-1]) / 2, -0.25, task['subtitle'],
            ha='center', fontsize=7.5, color=sub_col)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=input_fill,  edgecolor=input_border,  label='Input'),
    mpatches.Patch(facecolor=cell_fill,   edgecolor=cell_border,   label='RNN cell'),
    mpatches.Patch(facecolor=output_fill, edgecolor=output_border, label='Output'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=3,
           fontsize=8.5, framealpha=0.9, edgecolor='#d1d5db',
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('RNN Sequence Task Types', fontsize=11.5,
             fontweight='bold', color='#111111', y=1.01)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'rnn_task_types.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved rnn_task_types.png")
