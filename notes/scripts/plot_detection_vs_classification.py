import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax in axes:
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    img_rect = mpatches.FancyBboxPatch((0.3, 0.8), 5.4, 4.4,
        boxstyle="round,pad=0.05", linewidth=1.5,
        edgecolor='#888888', facecolor='#e8e8e8')
    ax.add_patch(img_rect)
    # simple cat silhouette suggestion — filled ellipse body
    body = mpatches.Ellipse((3.0, 2.8), 2.2, 1.4, angle=0,
        facecolor='#aaaaaa', edgecolor='#666666', lw=1.2)
    ax.add_patch(body)
    head = mpatches.Circle((4.0, 3.6), 0.55,
        facecolor='#aaaaaa', edgecolor='#666666', lw=1.2)
    ax.add_patch(head)
    # ears
    ax.plot([3.65, 3.5, 3.8], [4.05, 4.45, 4.45], color='#666666', lw=1.2)
    ax.plot([4.2, 4.1, 4.4], [4.05, 4.45, 4.45], color='#666666', lw=1.2)

# Left — classification
ax = axes[0]
ax.set_title('Classification', fontsize=12, fontweight='bold', pad=8)
label_box = mpatches.FancyBboxPatch((1.5, 0.05), 3.0, 0.6,
    boxstyle="round,pad=0.08", linewidth=1.5,
    edgecolor='#4C72B0', facecolor='#4C72B0', alpha=0.85)
ax.add_patch(label_box)
ax.text(3.0, 0.35, '"cat"   p = 0.94', ha='center', va='center',
        fontsize=10, color='white', fontweight='bold')
ax.text(3.0, 5.45, 'One label per image', ha='center', fontsize=9,
        color='#444444', style='italic')

# Right — detection
ax = axes[1]
ax.set_title('Detection', fontsize=12, fontweight='bold', pad=8)
bbox = mpatches.FancyBboxPatch((1.6, 1.9), 2.9, 2.4,
    boxstyle="square,pad=0.0", linewidth=2.5,
    edgecolor='#C44E52', facecolor='none')
ax.add_patch(bbox)
ax.text(1.65, 4.45, '"cat"  0.91', fontsize=9, color='white',
        fontweight='bold',
        bbox=dict(facecolor='#C44E52', edgecolor='none', pad=2))
ax.text(3.0, 5.45, 'Label + bounding box', ha='center', fontsize=9,
        color='#444444', style='italic')
ax.text(3.0, 0.35, '(x=1.6, y=1.9, w=2.9, h=2.4)', ha='center', fontsize=8.5,
        color='#C44E52')

plt.suptitle('Classification vs Object Detection', fontsize=13,
             fontweight='bold', y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'detection_vs_classification.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print("Saved detection_vs_classification.png")
