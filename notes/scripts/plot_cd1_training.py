import matplotlib.pyplot as plt
import matplotlib.patches as FancyArrowPatch
import os

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.patch.set_facecolor('#1e1e1e')

v_color  = '#4fc3f7'
h_color  = '#ce93d8'
bg_color = '#1e1e1e'
text_color = '#eeeeee'
arrow_color = '#aaaaaa'
edge_color  = '#666666'
active_edge = '#80cbc4'

titles = ['Positive Phase', 'Negative Phase', 'Weight Update']
subtitles = [
    'Clamp v to data\nInfer h ~ P(h|v)',
    'Reconstruct v\u0302 ~ P(v|h)\nInfer h\u0302 ~ P(h|v\u0302)',
    '\u0394w = \u03b7(\u27e8vh\u1d40\u27e9\u2090\u2091\u2090\u2090 \u2212 \u27e8v\u0302h\u0302\u1d40\u27e9\u2098\u2092\u2091\u2091\u2097)',
]

n_v, n_h = 3, 2
v_ys = [0.8, 1.6, 2.4]
h_ys = [1.2, 2.0]
v_x, h_x = 0.6, 2.0

for col, ax in enumerate(axes):
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0.2, 3.2)
    ax.axis('off')

    # Edges
    for vy in v_ys:
        for hy in h_ys:
            ec = active_edge if col < 2 else edge_color
            ax.plot([v_x, h_x], [vy, hy], color=ec, linewidth=1.0, zorder=1)

    # Visible nodes
    for i, vy in enumerate(v_ys):
        fc = v_color if col == 0 else ('#80cbc4' if col == 1 else v_color)
        c = plt.Circle((v_x, vy), 0.22, color=fc, zorder=3)
        ax.add_patch(c)
        label = 'v' if col < 2 else 'v'
        marker = '' if col == 0 else '\u0302' if col == 1 else ''
        ax.text(v_x, vy, f'v{i+1}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='#1e1e1e', zorder=4)

    # Hidden nodes
    for j, hy in enumerate(h_ys):
        fc = h_color if col == 0 else ('#f48fb1' if col == 1 else h_color)
        c = plt.Circle((h_x, hy), 0.22, color=fc, zorder=3)
        ax.add_patch(c)
        ax.text(h_x, hy, f'h{j+1}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='#1e1e1e', zorder=4)

    # Layer labels
    ax.text(v_x, 0.35, 'v', ha='center', fontsize=9, color=v_color)
    ax.text(h_x, 0.35, 'h', ha='center', fontsize=9, color=h_color)

    # Title
    ax.text(1.3, 3.05, titles[col], ha='center', va='center',
            fontsize=10, fontweight='bold', color=text_color)
    ax.text(1.3, 2.72, subtitles[col], ha='center', va='center',
            fontsize=7.5, color='#aaaaaa', linespacing=1.5)

# Arrows between panels
for spine_x in [0.355, 0.688]:
    fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
        (spine_x, 0.5), (spine_x + 0.005, 0.5),
        transform=fig.transFigure,
        arrowstyle='->', color=arrow_color,
        mutation_scale=15, linewidth=1.5
    ))

plt.tight_layout(pad=1.2)
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'cd1_training.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved cd1_training.png")
