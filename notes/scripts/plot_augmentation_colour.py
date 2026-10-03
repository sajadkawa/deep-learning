import matplotlib.pyplot as plt
import numpy as np
import os

rng = np.random.default_rng(42)

def make_image():
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.85
    for r in range(32):
        img[r, :, 2] = 0.6 + 0.3 * (1 - r / 32)
    img[22:, :] = [0.3, 0.6, 0.2]
    img[14:22, 14:18] = [0.4, 0.25, 0.1]
    for r in range(32):
        for c in range(32):
            if (r - 10)**2 + (c - 16)**2 < 64:
                img[r, c] = [0.1, 0.55, 0.15]
    return np.clip(img, 0, 1)

def brightness(img, factor=1.6):
    return np.clip(img * factor, 0, 1)

def contrast(img, factor=2.0):
    mean = img.mean()
    return np.clip(mean + factor * (img - mean), 0, 1)

def saturation(img, factor=0.2):
    grey = img.mean(axis=2, keepdims=True)
    return np.clip(grey + factor * (img - grey), 0, 1)

def hue_jitter(img, shift=0.15):
    # Simple hue shift via RGB rotation approximation
    out = img.copy()
    out[:,:,0] = np.clip(img[:,:,0] + shift * img[:,:,1], 0, 1)
    out[:,:,1] = np.clip(img[:,:,1] - shift * img[:,:,0], 0, 1)
    return np.clip(out, 0, 1)

src = make_image()

fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
fig.suptitle('Colour Augmentation', fontsize=13, fontweight='bold', y=1.02)

titles = ['Original', 'Brightness\n(factor=1.6)', 'Contrast\n(factor=2.0)', 'Saturation\n(factor=0.2)', 'Hue\nJitter']
imgs = [src, brightness(src), contrast(src), saturation(src), hue_jitter(src)]

for ax, img, title in zip(axes, imgs, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#455A64')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'augmentation_colour.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print('Saved augmentation_colour.png')
