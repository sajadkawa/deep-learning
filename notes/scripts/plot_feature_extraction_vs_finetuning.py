import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(12, 7))

C_FROZEN = '#B0BEC5'
C_ACTIVE = '#64B5F6'
C_NEW    = '#EF9A9A'
C_TEXT   = '#212121'

layer_configs = [
    ('Conv Block 1\n(edges)',      64),
    ('Conv Block 2\n(textures)',  128),
    ('Conv Block 3\n(shapes)',    256),
    ('Conv Block 4\n(parts)',     512),
    ('Conv Block 5\n(semantics)', 512),
    ('Global\nAvg Pool',          512),
]

def draw_network(ax, title, frozen_count, new_head_label):
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 9.5)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color=C_TEXT)
    n = len(layer_configs)
    h, gap = 1.0, 0.25
    for i, (label, filters) in enumerate(layer_configs):
        y = (n-1-i)*(h+gap) + 1.5
        color = C_FROZEN if i < frozen_count else C_ACTIVE
        rect = mpatches.FancyBboxPatch((0.5, y), 3.0, h,
            boxstyle='round,pad=0.05', facecolor=color, edgecolor='#455A64', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(2.0, y+h/2+0.1, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=C_TEXT)
        ax.text(2.0, y+h/2-0.2, f'{filters} channels', ha='center', va='center',
                fontsize=7.5, color='#546E7A')
        status = 'frozen' if i < frozen_count else 'trainable'
        ax.text(3.65, y+h/2, status, ha='left', va='center', fontsize=7,
                color='#546E7A' if i < frozen_count else '#1565C0', style='italic')
        if i < n-1:
            ax.annotate('', xy=(2.0, y), xytext=(2.0, y-gap),
                arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.2))
    y_head = 0.0
    ax.annotate('', xy=(2.0, y_head+h), xytext=(2.0, 1.5),
        arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.2))
    rect = mpatches.FancyBboxPatch((0.5, y_head), 3.0, h,
        boxstyle='round,pad=0.05', facecolor=C_NEW, edgecolor='#C62828', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2.0, y_head+h/2, new_head_label, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=C_TEXT)
    ax.text(3.65, y_head+h/2, 'trainable', ha='left', va='center',
            fontsize=7, color='#C62828', style='italic')

draw_network(axes[0], 'Feature Extraction\n(all backbone frozen)', 6, 'New Head\n(FC layers, K classes)')
draw_network(axes[1], 'Fine-tuning\n(last 2 blocks unfrozen)', 4, 'New Head\n(FC layers, K classes)')

legend_items = [
    mpatches.Patch(facecolor=C_FROZEN, edgecolor='#455A64', label='Frozen — no gradient'),
    mpatches.Patch(facecolor=C_ACTIVE, edgecolor='#455A64', label='Fine-tuned — low lr'),
    mpatches.Patch(facecolor=C_NEW,    edgecolor='#C62828', label='New head — trained from scratch'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=3,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.06, 1, 1])
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'feature_extraction_vs_finetuning.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved feature_extraction_vs_finetuning.png')
