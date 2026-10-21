import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_facecolor('white')

def box(ax, x, y, w, h, color, border, label, sublabel='', fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.05', facecolor=color, edgecolor=border, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2+(0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold', color='#111111')
    if sublabel:
        ax.text(x+w/2, y+h/2-0.22, sublabel,
                ha='center', va='center', fontsize=7.5, color='#374151')

def arrow(ax, x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.8))

ax.text(3.5, 5.55, 'Phase 1 — Pre-training on ImageNet', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#2563eb')

box(ax, 0.2, 4.1, 1.6, 0.9, '#dbeafe', '#2563eb', 'ImageNet', '1.2M images\n1,000 classes', fontsize=8)
arrow(ax, 1.8, 2.2, 4.55)

layer_labels = ['Conv\nBlock 1', 'Conv\nBlock 2', 'Conv\nBlock 3', 'Conv\nBlock 4', 'Conv\nBlock 5']
for i, lbl in enumerate(layer_labels):
    box(ax, 2.2+i*0.85, 4.1, 0.75, 0.9, '#dbeafe', '#2563eb', lbl, fontsize=7.5)
    if i < 4:
        arrow(ax, 2.95+i*0.85, 3.2+i*0.85, 4.55)

arrow(ax, 6.45, 6.85, 4.55)
box(ax, 6.85, 4.1, 0.9, 0.9, '#dbeafe', '#2563eb', 'FC\nHead', '1000\nclasses', fontsize=7.5)

ax.plot([0.1, 13.9], [3.7, 3.7], color='#d1d5db', lw=1, linestyle='--')
ax.text(7.0, 3.45, 'Adapt to target task', ha='center', va='center',
        fontsize=9, color='#6b7280', style='italic')

ax.text(3.5, 3.1, 'Feature Extraction', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#374151')
for i, lbl in enumerate(layer_labels):
    box(ax, 0.3+i*0.85, 1.9, 0.75, 0.9, '#f3f4f6', '#9ca3af', lbl, fontsize=7.5)
    if i < 4:
        arrow(ax, 1.05+i*0.85, 1.3+i*0.85, 2.35)
arrow(ax, 4.55, 4.95, 2.35)
box(ax, 4.95, 1.9, 1.1, 0.9, '#fee2e2', '#dc2626', 'New\nHead', 'K classes', fontsize=8)
ax.text(0.3, 1.65, 'Frozen (no gradient)', fontsize=7.5, color='#6b7280', style='italic')
ax.text(4.95, 1.65, 'Trained', fontsize=7.5, color='#dc2626', style='italic')

ax.text(10.2, 3.1, 'Fine-tuning', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#374151')
ft_colors  = ['#f3f4f6', '#f3f4f6', '#f3f4f6', '#fef3c7', '#fef3c7']
ft_borders = ['#9ca3af', '#9ca3af', '#9ca3af', '#b45309', '#b45309']
for i, (lbl, col, brd) in enumerate(zip(layer_labels, ft_colors, ft_borders)):
    box(ax, 7.5+i*0.85, 1.9, 0.75, 0.9, col, brd, lbl, fontsize=7.5)
    if i < 4:
        arrow(ax, 8.25+i*0.85, 8.5+i*0.85, 2.35)
arrow(ax, 11.75, 12.15, 2.35)
box(ax, 12.15, 1.9, 1.1, 0.9, '#fee2e2', '#dc2626', 'New\nHead', 'K classes', fontsize=8)
ax.text(7.5, 1.65, 'Frozen', fontsize=7.5, color='#6b7280', style='italic')
ax.text(10.7, 1.65, 'Fine-tuned (low lr)', fontsize=7.5, color='#b45309', style='italic')
ax.text(12.15, 1.65, 'Trained', fontsize=7.5, color='#dc2626', style='italic')

ax.plot([7.0, 7.0], [1.5, 3.6], color='#d1d5db', lw=1, linestyle='--')

legend_items = [
    mpatches.Patch(facecolor='#dbeafe', edgecolor='#2563eb', label='Pre-trained (ImageNet)'),
    mpatches.Patch(facecolor='#f3f4f6', edgecolor='#9ca3af', label='Frozen'),
    mpatches.Patch(facecolor='#fef3c7', edgecolor='#b45309', label='Fine-tuned'),
    mpatches.Patch(facecolor='#fee2e2', edgecolor='#dc2626', label='New head (trained)'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=4,
          fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'transfer_learning_pipeline.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved transfer_learning_pipeline.png')
