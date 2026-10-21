import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(14, 3.5))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_facecolor('white')

blocks = [
    (0.3,  "Input\nImage\n224×224×3",   '#f3f4f6', '#9ca3af'),
    (2.3,  "Conv\nBlocks\n(×5)",         '#dbeafe', '#2563eb'),
    (4.3,  "Feature\nMaps\nH×W×C",       '#dbeafe', '#2563eb'),
    (6.3,  "Flatten\n+\nFC Layers",      '#fee2e2', '#dc2626'),
    (8.3,  "Softmax\nOutput\n(K scores)", '#fef3c7', '#b45309'),
    (10.3, "Predicted\nClass\nLabel",     '#dcfce7', '#15803d'),
]

for x, label, color, border in blocks:
    rect = mpatches.FancyBboxPatch((x, 1.0), 1.7, 2.0,
        boxstyle="round,pad=0.1", linewidth=1.5,
        edgecolor=border, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + 0.85, 2.0, label, ha='center', va='center',
            fontsize=9, color='#111111', fontweight='bold')

arrows_x = [2.0, 4.0, 6.0, 8.0, 10.0]
for x in arrows_x:
    ax.annotate('', xy=(x + 0.3, 2.0), xytext=(x, 2.0),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.8))

ax.text(7.0, 0.35, 'Feature extraction', ha='center', fontsize=8.5,
        color='#2563eb', style='italic')
ax.text(10.15, 0.35, 'Classification', ha='center', fontsize=8.5,
        color='#dc2626', style='italic')

ax.axvline(x=9.1, ymin=0.12, ymax=0.88, color='#d1d5db', lw=1, linestyle='--')

ax.set_title('Image Classification Pipeline', fontsize=12, fontweight='bold',
             color='#111111', pad=10)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'classification_pipeline.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved classification_pipeline.png")
