import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('white')

def draw_pipeline(ax, title, steps, title_color):
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, len(steps) * 1.4 + 0.5)
    ax.axis('off')
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=12, fontweight='bold', color=title_color, pad=10)

    for i, (label, color, border, sublabel) in enumerate(reversed(steps)):
        y = i * 1.4
        rect = mpatches.FancyBboxPatch((0.5, y), 4.0, 0.9,
            boxstyle='round,pad=0.06', facecolor=color, edgecolor=border, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(2.5, y + 0.9/2 + (0.1 if sublabel else 0), label,
                ha='center', va='center', fontsize=9.5, fontweight='bold', color='#111111')
        if sublabel:
            ax.text(2.5, y + 0.9/2 - 0.18, sublabel,
                    ha='center', va='center', fontsize=8, color='#374151')
        if i < len(steps) - 1:
            ax.annotate('', xy=(2.5, y + 0.9), xytext=(2.5, y + 1.4),
                arrowprops=dict(arrowstyle='<-', color='#9ca3af', lw=1.5))

train_steps = [
    ('Loss + Backprop',    '#fee2e2', '#dc2626', 'cross-entropy, update weights'),
    ('Model',              '#fef3c7', '#b45309', 'CNN forward pass'),
    ('Normalise',          '#dcfce7', '#15803d', 'mean/std from training set'),
    ('Colour Jitter',      '#dbeafe', '#2563eb', 'random brightness/contrast/saturation'),
    ('Random Crop',        '#dbeafe', '#2563eb', 'pad 4px, crop to original size'),
    ('Random H-Flip',      '#dbeafe', '#2563eb', 'p=0.5'),
    ('Training Image',     '#f3f4f6', '#9ca3af', ''),
]

test_steps = [
    ('Prediction',           '#fee2e2', '#dc2626', 'argmax of softmax output'),
    ('Model',                '#fef3c7', '#b45309', 'CNN forward pass'),
    ('Normalise',            '#dcfce7', '#15803d', 'same mean/std as training'),
    ('Centre Crop / Resize', '#dcfce7', '#15803d', 'deterministic — fixed size'),
    ('Test Image',           '#f3f4f6', '#9ca3af', ''),
]

draw_pipeline(axes[0], 'Training Pipeline', train_steps, '#2563eb')
draw_pipeline(axes[1], 'Test Pipeline',     test_steps,  '#15803d')

legend_items = [
    mpatches.Patch(facecolor='#dbeafe', edgecolor='#2563eb', label='Random augmentation (training only)'),
    mpatches.Patch(facecolor='#dcfce7', edgecolor='#15803d', label='Deterministic preprocessing'),
    mpatches.Patch(facecolor='#fef3c7', edgecolor='#b45309', label='Model'),
    mpatches.Patch(facecolor='#fee2e2', edgecolor='#dc2626', label='Output / Loss'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=4,
           fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(rect=[0, 0.07, 1, 1])
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'augmentation_train_vs_test.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved augmentation_train_vs_test.png')
