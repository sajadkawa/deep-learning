import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(9, 3))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

total_w = 8.0
x0 = 1.0
y0 = 1.5
h  = 1.0

splits = [
    ('Training Set\n(60–80%)',      0.70, '#dcfce7', '#15803d'),
    ('Validation Set\n(10–20%)',    0.15, '#ffedd5', '#ea580c'),
    ('Test Set\n(10–20%)',          0.15, '#dbeafe', '#2563eb'),
]

x = x0
for label, frac, fc, ec in splits:
    w = total_w * frac
    rect = mpatches.FancyBboxPatch((x, y0), w, h,
                                    boxstyle='square,pad=0',
                                    facecolor=fc, edgecolor=ec, linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y0 + h / 2, label, ha='center', va='center',
            color='#111111', fontsize=10, fontweight='bold', zorder=4)
    x += w

outer = mpatches.FancyBboxPatch((x0, y0), total_w, h,
                                 boxstyle='square,pad=0',
                                 facecolor='none', edgecolor='#9ca3af',
                                 linewidth=2, zorder=5)
ax.add_patch(outer)
ax.text(x0 + total_w / 2, y0 + h + 0.3, 'Full Dataset',
        ha='center', va='bottom', color='#374151', fontsize=11, fontweight='bold')

annotations = [
    (x0 + total_w * 0.35,  y0 - 0.25, 'Update weights\n(gradient descent)',              '#15803d'),
    (x0 + total_w * 0.775, y0 - 0.25, 'Guide decisions\n(hyperparameters, early stop)',  '#ea580c'),
    (x0 + total_w * 0.925, y0 - 0.25, 'Final evaluation\n(used once)',                   '#2563eb'),
]
for x, y, note, color in annotations:
    ax.text(x, y, note, ha='center', va='top', color=color, fontsize=8)

ax.set_xlim(0.5, 9.5)
ax.set_ylim(0.3, 3.2)
ax.axis('off')
ax.set_title('Dataset Split: Training / Validation / Test', color='#111111', fontsize=12, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'dataset_split.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
