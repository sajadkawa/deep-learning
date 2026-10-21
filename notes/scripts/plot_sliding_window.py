import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.patch.set_facecolor('white')
titles = ['Scale 1 — large window', 'Scale 2 — medium window', 'Scale 3 — small window']
window_sizes = [2.8, 2.0, 1.3]
colors  = ['#2563eb', '#15803d', '#dc2626']
fills   = ['#dbeafe', '#dcfce7', '#fee2e2']

for ax, title, ws, col, fill in zip(axes, titles, window_sizes, colors, fills):
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=9.5, fontweight='bold', color='#111111')

    img = mpatches.FancyBboxPatch((0.2, 0.2), 5.6, 5.6,
        boxstyle="square,pad=0", linewidth=1.5,
        edgecolor='#9ca3af', facecolor='#f3f4f6')
    ax.add_patch(img)

    step = ws * 0.6
    x = 0.2
    while x + ws <= 5.8:
        y = 0.2
        while y + ws <= 5.8:
            rect = mpatches.Rectangle((x, y), ws, ws,
                linewidth=0.8, edgecolor=col, facecolor=fill, alpha=0.15)
            ax.add_patch(rect)
            y += step
        x += step

    active = mpatches.Rectangle((1.8, 2.2), ws, ws,
        linewidth=2.5, edgecolor=col, facecolor=fill, alpha=0.5)
    ax.add_patch(active)
    ax.text(1.8 + ws/2, 2.2 + ws/2, 'CNN', ha='center', va='center',
            fontsize=9, fontweight='bold', color=col)

axes[1].text(3.0, -0.2, 'Each window passed through CNN independently',
             ha='center', fontsize=8.5, color='#374151', style='italic',
             transform=axes[1].transData)

plt.suptitle('Sliding Window Detection — Exhaustive Search at Multiple Scales',
             fontsize=12, fontweight='bold', color='#111111', y=1.02)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'sliding_window.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved sliding_window.png")
