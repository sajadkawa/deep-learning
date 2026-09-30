import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
fig.patch.set_facecolor('#0d1117')

# ── Panel 1: Input X-ray (simulated grayscale) ────────────────────────────────
ax = axes[0]
ax.set_facecolor('#0d1117')
rng = np.random.default_rng(7)
xray = rng.uniform(0.15, 0.55, (60, 60))
# Add some lung-like structure
for cx, cy, r in [(20, 30, 12), (40, 30, 12)]:
    for i in range(60):
        for j in range(60):
            if (i - cx)**2 + (j - cy)**2 < r**2:
                xray[i, j] = rng.uniform(0.55, 0.85)
ax.imshow(xray, cmap='gray', vmin=0, vmax=1)
ax.set_title('Input X-ray', color='#e6edf3', fontsize=11, pad=6)
ax.axis('off')

# ── Panel 2: CNN prediction label ─────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('#0d1117')
ax.text(0.5, 0.6, 'CNN predicts:', ha='center', va='center',
        color='#8b949e', fontsize=12)
ax.text(0.5, 0.4, 'Pneumonia', ha='center', va='center',
        color='#f85149', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#3a1a1a',
                  edgecolor='#f85149', linewidth=1.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Prediction', color='#e6edf3', fontsize=11, pad=6)

# ── Panel 3: Grad-CAM heatmap ─────────────────────────────────────────────────
ax = axes[2]
ax.set_facecolor('#0d1117')

# Build a heatmap: high activation in the right lung region
heatmap = np.zeros((60, 60))
cx, cy, r = 40, 30, 14
for i in range(60):
    for j in range(60):
        d = np.sqrt((i - cx)**2 + (j - cy)**2)
        if d < r:
            heatmap[i, j] = np.clip(1.0 - d / r, 0, 1) ** 1.5

# Overlay on grayscale
ax.imshow(xray, cmap='gray', vmin=0, vmax=1, alpha=0.6)
ax.imshow(heatmap, cmap='hot', alpha=0.55, vmin=0, vmax=1)
ax.annotate('high activation\n(network focused here)', xy=(40, 30),
            xytext=(10, 8), color='white', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
ax.set_title('Grad-CAM Heatmap', color='#e6edf3', fontsize=11, pad=6)
ax.axis('off')

fig.suptitle('Grad-CAM — Where the CNN Looked to Predict Pneumonia',
             color='#e6edf3', fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'gradcam_example.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
