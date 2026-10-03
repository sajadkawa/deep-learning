import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

C_AUG   = '#90CAF9'
C_DET   = '#A5D6A7'
C_MODEL = '#FFCC80'
C_LOSS  = '#EF9A9A'
C_TEXT  = '#212121'

def draw_pipeline(ax, title, steps, title_color):
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, len(steps) * 1.4 + 0.5)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', color=title_color, pad=10)

    for i, (label, color, sublabel) in enumerate(reversed(steps)):
        y = i * 1.4
        rect = mpatches.FancyBboxPatch((0.5, y), 4.0, 0.9,
            boxstyle='round,pad=0.06', facecolor=color, edgecolor='#455A64', linewidth=1.3)
        ax.add_patch(rect)
        ax.text(2.5, y + 0.9/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontsize=9.5, fontweight='bold', color=C_TEXT)
        if sublabel:
            ax.text(2.5, y + 0.9/2 - 0.18, sublabel,
                    ha='center', va='center', fontsize=8, color='#546E7A')
        if i < len(steps) - 1:
            ax.annotate('', xy=(2.5, y + 0.9), xytext=(2.5, y + 1.4),
                arrowprops=dict(arrowstyle='<-', color='#37474F', lw=1.5))

train_steps = [
    ('Loss + Backprop',    C_LOSS,  'cross-entropy, update weights'),
    ('Model',              C_MODEL, 'CNN forward pass'),
    ('Normalise',          C_DET,   'mean/std from training set'),
    ('Colour Jitter',      C_AUG,   'random brightness/contrast/saturation'),
    ('Random Crop',        C_AUG,   'pad 4px, crop to original size'),
    ('Random H-Flip',      C_AUG,   'p=0.5'),
    ('Training Image',     '#E0E0E0', ''),
]

test_steps = [
    ('Prediction',         C_LOSS,  'argmax of softmax output'),
    ('Model',              C_MODEL, 'CNN forward pass'),
    ('Normalise',          C_DET,   'same mean/std as training'),
    ('Centre Crop / Resize', C_DET, 'deterministic — fixed size'),
    ('Test Image',         '#E0E0E0', ''),
]

draw_pipeline(axes[0], 'Training Pipeline', train_steps, '#1565C0')
draw_pipeline(axes[1], 'Test Pipeline', test_steps, '#2E7D32')

legend_items = [
    mpatches.Patch(facecolor=C_AUG,   edgecolor='#455A64', label='Random augmentation (training only)'),
    mpatches.Patch(facecolor=C_DET,   edgecolor='#455A64', label='Deterministic preprocessing'),
    mpatches.Patch(facecolor=C_MODEL, edgecolor='#455A64', label='Model'),
    mpatches.Patch(facecolor=C_LOSS,  edgecolor='#455A64', label='Output / Loss'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.07, 1, 1])
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'augmentation_train_vs_test.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved augmentation_train_vs_test.png')
