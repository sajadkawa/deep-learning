import matplotlib.pyplot as plt
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
fig.patch.set_facecolor('white')

# ── Panel 1: Input X-ray ──────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('white')
rng = np.random.default_rng(7)
xray = rng.uniform(0.15, 0.55, (60, 60))
for cx, cy, r in [(20, 30, 12), (40, 30, 12)]:
    for i in range(60):
        for j in range(60):
            if (i - cx)**2 + (j - cy)**2 < r**2:
                xray[i, j] = rng.uniform(0.55, 0.85)
ax.imshow(xray, cmap='gray', vmin=0, vmax=1)
ax.set_title('Input X-ray', color='#111111', fontsize=11, fontweight='bold', pad=6)
ax.axis('off')

# ── Panel 2: CNN prediction ───────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('white')
ax.text(0.5, 0.6, 'CNN predicts:', ha='center', va='center',
        color='#374151', fontsize=12, fontweight='bold')
ax.text(0.5, 0.4, 'Pneumonia', ha='center', va='center',
        color='#b91c1c', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fee2e2',
                  edgecolor='#b91c1c', linewidth=1.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Prediction', color='#111111', fontsize=11, fontweight='bold', pad=6)

# ── Panel 3: Grad-CAM heatmap ─────────────────────────────────────────────────
ax = axes[2]
ax.set_facecolor('white')

heatmap = np.zeros((60, 60))
cx, cy, r = 40, 30, 14
for i in range(60):
    for j in range(60):
        d = np.sqrt((i - cx)**2 + (j - cy)**2)
        if d < r:
            heatmap[i, j] = np.clip(1.0 - d / r, 0, 1) ** 1.5

ax.imshow(xray, cmap='gray', vmin=0, vmax=1, alpha=0.6)
ax.imshow(heatmap, cmap='hot', alpha=0.55, vmin=0, vmax=1)
ax.annotate('high activation\n(network focused here)', xy=(40, 30),
            xytext=(10, 8), color='#111111', fontsize=8, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#374151', lw=1.2))
ax.set_title('Grad-CAM Heatmap', color='#111111', fontsize=11, fontweight='bold', pad=6)
ax.axis('off')

fig.suptitle('Grad-CAM — Where the CNN Looked to Predict Pneumonia',
             color='#111111', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'gradcam_example.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved gradcam_example.png")
