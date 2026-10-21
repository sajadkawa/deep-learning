import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 9)
ax.set_ylim(-0.8, 8)
ax.axis('off')
ax.set_facecolor('white')

def box(ax, x, y, w, h, color, border, lines, fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.06', facecolor=color, edgecolor=border, linewidth=1.3)
    ax.add_patch(rect)
    n = len(lines)
    for i, line in enumerate(lines):
        offset = (n-1-i)*0.2 - (n-1)*0.1
        ax.text(x+w/2, y+h/2+offset, line, ha='center', va='center', fontsize=fontsize,
                fontweight='bold' if i == 0 else 'normal', color='#111111')

cx, bw, bh = 3.5, 2.0, 0.65

ax.text(cx+bw/2, 7.6, 'Input  x', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#111111')
ax.annotate('', xy=(cx+bw/2, 6.85), xytext=(cx+bw/2, 7.35),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.6))

box(ax, cx, 6.2, bw, bh, '#dbeafe', '#2563eb', ['Conv 3x3', 'C filters, stride=1, pad=1'])
ax.annotate('', xy=(cx+bw/2, 5.7), xytext=(cx+bw/2, 6.2),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.6))

box(ax, cx, 5.05, bw, bh, '#fef3c7', '#b45309', ['Batch Norm'])
ax.annotate('', xy=(cx+bw/2, 4.55), xytext=(cx+bw/2, 5.05),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.6))

box(ax, cx, 3.9, bw, bh, '#dcfce7', '#15803d', ['ReLU'])
ax.annotate('', xy=(cx+bw/2, 3.4), xytext=(cx+bw/2, 3.9),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.6))

box(ax, cx, 2.75, bw, bh, '#dbeafe', '#2563eb', ['Conv 3x3', 'C filters, stride=1, pad=1'])
ax.annotate('', xy=(cx+bw/2, 2.25), xytext=(cx+bw/2, 2.75),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.6))

box(ax, cx, 1.6, bw, bh, '#fef3c7', '#b45309', ['Batch Norm'])
ax.annotate('', xy=(cx+bw/2, 1.1), xytext=(cx+bw/2, 1.6),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.6))

add_x, add_y = cx+bw/2, 0.75
circle = plt.Circle((add_x, add_y), 0.28, facecolor='#fee2e2', edgecolor='#dc2626', linewidth=1.3, zorder=3)
ax.add_patch(circle)
ax.text(add_x, add_y, '+', ha='center', va='center', fontsize=14, fontweight='bold', color='#111111', zorder=4)

ax.annotate('', xy=(add_x, 0.1), xytext=(add_x, add_y-0.28),
    arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.6))

box(ax, cx, -0.55, bw, bh, '#dcfce7', '#15803d', ['ReLU'])
ax.text(cx+bw/2, -0.75, 'Output  F(x) + x', ha='center', va='top',
        fontsize=10, fontweight='bold', color='#111111')

sx = cx+bw+0.5
ax.plot([cx+bw/2, sx], [7.35, 7.35], color='#374151', lw=1.8)
ax.plot([sx, sx], [7.35, add_y], color='#374151', lw=1.8)
ax.annotate('', xy=(add_x+0.28, add_y), xytext=(sx, add_y),
    arrowprops=dict(arrowstyle='->', color='#374151', lw=1.8))
ax.text(sx+0.15, (7.35+add_y)/2, 'Shortcut\n(identity)', ha='left', va='center',
        fontsize=8.5, color='#374151', style='italic')

ax.text(0.2, 0.3, 'output = F(x) + x', ha='left', va='center', fontsize=10, color='#111111',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9c3', edgecolor='#ca8a04', linewidth=1.2))

legend_items = [
    mpatches.Patch(facecolor='#dbeafe', edgecolor='#2563eb', label='Conv 3x3 (no bias — BN absorbs it)'),
    mpatches.Patch(facecolor='#fef3c7', edgecolor='#b45309', label='Batch Normalisation'),
    mpatches.Patch(facecolor='#dcfce7', edgecolor='#15803d', label='ReLU'),
    mpatches.Patch(facecolor='#fee2e2', edgecolor='#dc2626', label='Element-wise addition'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=8, framealpha=0.9)
ax.set_title('ResNet Basic Residual Block', fontsize=13, fontweight='bold', pad=8, color='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'resnet_residual_block.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved resnet_residual_block.png')
