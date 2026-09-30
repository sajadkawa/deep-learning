import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(10, 3.5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

total_w = 8.0
x0 = 1.0
y0 = 1.5
h = 1.0

splits = [
    ('Training Set\n(60–80%)', 0.70, '#1a3a1a', '#3fb950'),
    ('Validation Set\n(10–20%)', 0.15, '#3a2a1a', '#f0883e'),
    ('Test Set\n(10–20%)', 0.15, '#1a1a3a', '#388bfd'),
]

x = x0
for label, frac, fc, ec in splits:
    w = total_w * frac
    rect = mpatches.FancyBboxPatch((x, y0), w, h,
                                    boxstyle='square,pad=0',
                                    facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y0 + h / 2, label, ha='center', va='center',
            color='#e6edf3', fontsize=10, fontweight='bold', zorder=4)
    x += w

# Outer border for full dataset
outer = mpatches.FancyBboxPatch((x0, y0), total_w, h,
                                 boxstyle='square,pad=0',
                                 facecolor='none', edgecolor='#8b949e',
                                 linewidth=2, zorder=5)
ax.add_patch(outer)
ax.text(x0 + total_w / 2, y0 + h + 0.3, 'Full Dataset',
        ha='center', va='bottom', color='#8b949e', fontsize=11, fontweight='bold')

# Annotations below
annotations = [
    (x0 + total_w * 0.35, y0 - 0.25, 'Update weights\n(gradient descent)', '#3fb950'),
    (x0 + total_w * 0.775, y0 - 0.25, 'Guide decisions\n(hyperparameters, early stop)', '#f0883e'),
    (x0 + total_w * 0.925, y0 - 0.25, 'Final evaluation\n(used once)', '#388bfd'),
]
for x, y, note, color in annotations:
    ax.text(x, y, note, ha='center', va='top', color=color, fontsize=8)

ax.set_xlim(0.5, 9.5)
ax.set_ylim(0.3, 3.2)
ax.axis('off')
ax.set_title('Dataset Split: Training / Validation / Test', color='#e6edf3', fontsize=12, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'dataset_split.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
