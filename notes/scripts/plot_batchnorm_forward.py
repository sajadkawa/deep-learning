import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(13, 4))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 13)
ax.set_ylim(0, 4)
ax.axis('off')

BOX_W   = 1.8
BOX_H   = 1.4
BOX_Y   = 1.3
ARROW_Y = BOX_Y + BOX_H / 2

box_color   = '#161b22'
edge_color  = '#30363d'
title_color = '#e6edf3'
sub_color   = '#8b949e'
arrow_color = '#8b949e'

steps = [
    (0.7,  'Input',          'z₁, z₂, ..., zₘ',          '#388bfd'),
    (3.1,  'Step 1 & 2',     'Compute μ_B and σ²_B',      '#f0883e'),
    (5.5,  'Step 3',         'Normalize\nẑᵢ = (zᵢ − μ_B) / √(σ²_B + ε)', '#f0883e'),
    (7.9,  'Step 4',         'Scale & Shift\nyᵢ = γ · ẑᵢ + β',            '#3fb950'),
    (10.3, 'Output',         'y₁, y₂, ..., yₘ\nmean=β, std=γ',            '#388bfd'),
]

for x, title, subtitle, color in steps:
    rect = mpatches.FancyBboxPatch(
        (x, BOX_Y), BOX_W, BOX_H,
        boxstyle='round,pad=0.08',
        linewidth=1.5,
        edgecolor=color,
        facecolor=box_color
    )
    ax.add_patch(rect)
    ax.text(x + BOX_W / 2, BOX_Y + BOX_H * 0.68, title,
            ha='center', va='center', color=title_color, fontsize=10, fontweight='bold')
    ax.text(x + BOX_W / 2, BOX_Y + BOX_H * 0.28, subtitle,
            ha='center', va='center', color=sub_color, fontsize=8.5,
            linespacing=1.4)

# Arrows between boxes
arrow_xs = [(steps[i][0] + BOX_W, steps[i+1][0]) for i in range(len(steps) - 1)]
for x_start, x_end in arrow_xs:
    ax.annotate('', xy=(x_end, ARROW_Y), xytext=(x_start, ARROW_Y),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

# Learnable params annotation
ax.text(steps[3][0] + BOX_W / 2, BOX_Y - 0.35,
        'γ, β — learnable parameters', ha='center', color='#8b949e', fontsize=8.5,
        style='italic')

ax.set_title('Batch Normalization — Forward Pass',
             color='#e6edf3', fontsize=12, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'batchnorm_forward.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
