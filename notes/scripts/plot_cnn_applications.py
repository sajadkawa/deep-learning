import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor('#f8f8f8')

def draw_image_placeholder(ax, x0, y0, w, h, label, color='#cccccc'):
    rect = mpatches.FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.05',
                                    facecolor=color, edgecolor='#888', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x0 + w/2, y0 + h/2, label, ha='center', va='center',
            fontsize=9, color='#444', fontweight='bold')

# ---- Panel 1: Image Classification ----
ax = axes[0]
ax.set_xlim(0, 6)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('#f8f8f8')
ax.set_title('Image Classification\nInput image → single label', fontsize=10, fontweight='bold', color='#222')

# Draw a simple cat-like image
img_data = np.random.RandomState(42).rand(8, 8) * 0.3 + 0.4
img_data[2:5, 2:6] = 0.8  # body
img_data[1:3, 2:4] = 0.9  # head
img_data[0:2, 1:3] = 0.7  # ear left
img_data[0:2, 3:5] = 0.7  # ear right
ax.imshow(img_data, extent=[0.5, 4.5, 2.5, 6.5], cmap='gray', vmin=0, vmax=1, aspect='auto', zorder=2)
ax.add_patch(plt.Rectangle((0.5, 2.5), 4.0, 4.0, facecolor='none', edgecolor='#888', linewidth=1.5))
ax.text(2.5, 6.8, 'Input image', ha='center', fontsize=8.5, color='#555')

# Arrow down
ax.annotate('', xy=(2.5, 1.8), xytext=(2.5, 2.4),
            arrowprops=dict(arrowstyle='->', color='#555', lw=2.0))
ax.text(2.8, 2.1, 'CNN', ha='left', fontsize=8, color='#555')

# Label output
label_box = mpatches.FancyBboxPatch((1.0, 0.3), 3.0, 1.2, boxstyle='round,pad=0.1',
                                     facecolor='#d5f5e3', edgecolor='#27ae60', linewidth=2)
ax.add_patch(label_box)
ax.text(2.5, 0.9, '"Cat"  (confidence: 94%)', ha='center', va='center', fontsize=9,
        color='#1a7a1a', fontweight='bold')

# ---- Panel 2: Object Detection ----
ax2 = axes[1]
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 7)
ax2.axis('off')
ax2.set_facecolor('#f8f8f8')
ax2.set_title('Object Detection\nInput image → bounding boxes + labels', fontsize=10, fontweight='bold', color='#222')

# Image with two objects
img2 = np.ones((8, 8)) * 0.5
img2[1:4, 0:3] = 0.85  # object 1
img2[3:7, 4:8] = 0.25  # object 2
ax2.imshow(img2, extent=[0.3, 5.3, 1.5, 6.5], cmap='gray', vmin=0, vmax=1, aspect='auto', zorder=2)
ax2.add_patch(plt.Rectangle((0.3, 1.5), 5.0, 5.0, facecolor='none', edgecolor='#888', linewidth=1.5))

# Bounding boxes
# Object 1 (top-left region)
bb1 = plt.Rectangle((0.3 + 0*5/8, 1.5 + (8-4)*5/8), 3*5/8, 3*5/8,
                      facecolor='none', edgecolor='#e74c3c', linewidth=2.5, zorder=4)
ax2.add_patch(bb1)
ax2.text(0.3 + 1.5*5/8, 1.5 + (8-1)*5/8 + 0.1, 'Cat', ha='center', fontsize=8,
         color='white', fontweight='bold',
         bbox=dict(facecolor='#e74c3c', edgecolor='none', pad=1))

# Object 2 (bottom-right region)
bb2 = plt.Rectangle((0.3 + 4*5/8, 1.5 + (8-7)*5/8), 4*5/8, 4*5/8,
                      facecolor='none', edgecolor='#3498db', linewidth=2.5, zorder=4)
ax2.add_patch(bb2)
ax2.text(0.3 + 6*5/8, 1.5 + (8-3)*5/8 + 0.1, 'Dog', ha='center', fontsize=8,
         color='white', fontweight='bold',
         bbox=dict(facecolor='#3498db', edgecolor='none', pad=1))

ax2.text(2.8, 0.8, 'Localise + classify\nmultiple objects', ha='center', fontsize=9, color='#555',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#eaf4ff', edgecolor='#3498db'))

# ---- Panel 3: Segmentation ----
ax3 = axes[2]
ax3.set_xlim(0, 6)
ax3.set_ylim(0, 7)
ax3.axis('off')
ax3.set_facecolor('#f8f8f8')
ax3.set_title('Semantic Segmentation\nInput image → pixel-level labels', fontsize=10, fontweight='bold', color='#222')

# Segmentation mask
seg = np.zeros((8, 8))
seg[0:4, :] = 1   # sky
seg[4:6, :] = 2   # road
seg[4:8, 0:4] = 3  # car
seg[4:8, 4:8] = 2  # road

colors_seg = {0: [0.5, 0.5, 0.5], 1: [0.4, 0.7, 1.0], 2: [0.6, 0.6, 0.6], 3: [0.9, 0.3, 0.3]}
seg_rgb = np.array([[colors_seg[seg[r, c]] for c in range(8)] for r in range(8)])
ax3.imshow(seg_rgb, extent=[0.3, 5.3, 1.5, 6.5], aspect='auto', zorder=2)
ax3.add_patch(plt.Rectangle((0.3, 1.5), 5.0, 5.0, facecolor='none', edgecolor='#888', linewidth=1.5))

# Legend
legend_items = [
    mpatches.Patch(facecolor=[0.4, 0.7, 1.0], label='Sky'),
    mpatches.Patch(facecolor=[0.6, 0.6, 0.6], label='Road'),
    mpatches.Patch(facecolor=[0.9, 0.3, 0.3], label='Car'),
]
ax3.legend(handles=legend_items, loc='lower center', fontsize=8, ncol=3,
           bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

ax3.text(2.8, 7.0, 'Every pixel gets a class label', ha='center', fontsize=8.5, color='#555')

plt.tight_layout(pad=1.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_applications.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cnn_applications.png")
