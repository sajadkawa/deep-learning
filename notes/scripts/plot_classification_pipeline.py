import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(14, 3.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')

blocks = [
    (0.3,  "Input\nImage\n224×224×3",  "#4C72B0"),
    (2.3,  "Conv\nBlocks\n(×5)",        "#55A868"),
    (4.3,  "Feature\nMaps\nH×W×C",      "#55A868"),
    (6.3,  "Flatten\n+\nFC Layers",     "#C44E52"),
    (8.3,  "Softmax\nOutput\n(K scores)","#8172B2"),
    (10.3, "Predicted\nClass\nLabel",    "#CCB974"),
]

for x, label, color in blocks:
    rect = mpatches.FancyBboxPatch((x, 1.0), 1.7, 2.0,
        boxstyle="round,pad=0.1", linewidth=1.5,
        edgecolor='#333333', facecolor=color, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + 0.85, 2.0, label, ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

arrows_x = [2.0, 4.0, 6.0, 8.0, 10.0]
for x in arrows_x:
    ax.annotate('', xy=(x + 0.3, 2.0), xytext=(x, 2.0),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.8))

ax.text(7.0, 0.35, 'Feature extraction', ha='center', fontsize=8.5,
        color='#55A868', style='italic')
ax.text(10.15, 0.35, 'Classification', ha='center', fontsize=8.5,
        color='#C44E52', style='italic')

ax.axvline(x=9.1, ymin=0.12, ymax=0.88, color='#aaaaaa', lw=1, linestyle='--')

ax.set_title('Image Classification Pipeline', fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'classification_pipeline.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print("Saved classification_pipeline.png")
