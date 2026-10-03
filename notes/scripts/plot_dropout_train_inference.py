import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0d1117')

LAYER_XS   = [0.15, 0.42, 0.69, 0.96]
NEURON_YS  = [0.2, 0.4, 0.6, 0.8]
DROPPED    = {1: [1, 3], 2: [0, 2]}   # layer index → neuron indices dropped

layer_labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output']

active_color  = '#388bfd'
dropped_color = '#21262d'
edge_active   = '#388bfd'
edge_dropped  = '#30363d'
conn_active   = '#30363d'
conn_dropped  = '#21262d'
text_color    = '#e6edf3'
sub_color     = '#8b949e'

def draw_network(ax, title, subtitle, show_dropped):
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    r = 0.032

    # Draw connections first (behind neurons)
    for li in range(len(LAYER_XS) - 1):
        for yi in NEURON_YS:
            for yj in NEURON_YS:
                src_dropped = show_dropped and li in DROPPED and \
                              NEURON_YS.index(yi) in DROPPED[li]
                dst_dropped = show_dropped and (li + 1) in DROPPED and \
                              NEURON_YS.index(yj) in DROPPED[li + 1]
                color = conn_dropped if (src_dropped or dst_dropped) else conn_active
                ax.plot([LAYER_XS[li], LAYER_XS[li + 1]], [yi, yj],
                        color=color, lw=0.8, zorder=1)

    # Draw neurons
    for li, lx in enumerate(LAYER_XS):
        for ni, ny in enumerate(NEURON_YS):
            is_dropped = show_dropped and li in DROPPED and ni in DROPPED[li]
            fc = dropped_color if is_dropped else active_color
            ec = edge_dropped  if is_dropped else edge_active
            circle = mpatches.Circle((lx, ny), r, facecolor=fc,
                                     edgecolor=ec, lw=1.5, zorder=2)
            ax.add_patch(circle)
            if is_dropped:
                ax.text(lx, ny, '×', ha='center', va='center',
                        color='#f85149', fontsize=11, fontweight='bold', zorder=3)

        ax.text(lx, 0.07, layer_labels[li], ha='center', color=sub_color, fontsize=8.5)

    ax.set_title(title, color=text_color, fontsize=11, pad=8)
    ax.text(0.55, 0.95, subtitle, ha='center', va='top', color=sub_color,
            fontsize=9, transform=ax.transAxes, style='italic')

draw_network(axes[0], 'Training  (dropout active)',
             'dropped neurons marked ×, connections greyed', show_dropped=True)
draw_network(axes[1], 'Inference  (dropout disabled)',
             'all neurons active, no scaling needed', show_dropped=False)

fig.suptitle('Dropout — Training vs Inference', color=text_color, fontsize=12, y=1.01)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'dropout_train_inference.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
