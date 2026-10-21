import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

rng = np.random.default_rng(42)

# Synthetic "image": a simple coloured shape on a background
def make_image():
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.85
    # sky background gradient
    for r in range(32):
        img[r, :, 2] = 0.6 + 0.3 * (1 - r / 32)
    # green ground
    img[22:, :] = [0.3, 0.6, 0.2]
    # brown tree trunk
    img[14:22, 14:18] = [0.4, 0.25, 0.1]
    # green canopy
    for r in range(32):
        for c in range(32):
            if (r - 10)**2 + (c - 16)**2 < 64:
                img[r, c] = [0.1, 0.55, 0.15]
    return np.clip(img, 0, 1)

def h_flip(img):
    return img[:, ::-1, :].copy()

def rotate(img, deg):
    from scipy.ndimage import rotate as nd_rotate
    out = np.stack([nd_rotate(img[:,:,c], deg, reshape=False, mode='nearest') for c in range(3)], axis=2)
    return np.clip(out, 0, 1)

def random_crop(img, pad=4):
    h, w = img.shape[:2]
    padded = np.pad(img, ((pad,pad),(pad,pad),(0,0)), mode='edge')
    r = rng.integers(0, 2*pad)
    c = rng.integers(0, 2*pad)
    return padded[r:r+h, c:c+w]

def scale_crop(img, scale=0.75):
    h, w = img.shape[:2]
    sh, sw = int(h*scale), int(w*scale)
    r = (h - sh) // 2
    c = (w - sw) // 2
    crop = img[r:r+sh, c:c+sw]
    # resize back using repeat
    zoom = h // sh
    return np.repeat(np.repeat(crop, zoom, axis=0), zoom, axis=1)[:h, :w]

try:
    from scipy.ndimage import rotate as nd_rotate
    has_scipy = True
except ImportError:
    has_scipy = False

src = make_image()

fig, axes = plt.subplots(1, 5, figsize=(14, 3.5))
fig.patch.set_facecolor('white')
fig.suptitle('Geometric Augmentation', fontsize=13, fontweight='bold', color='#111111', y=1.02)

titles = ['Original', 'Horizontal\nFlip', 'Rotation\n(+15 deg)', 'Random\nCrop', 'Scale\nCrop']
imgs = [
    src,
    h_flip(src),
    rotate(src, 15) if has_scipy else h_flip(h_flip(src)),
    random_crop(src),
    scale_crop(src, 0.75),
]

for ax, img, title in zip(axes, imgs, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight='bold', color='#111111')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('#9ca3af')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'augmentation_geometric.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved augmentation_geometric.png')
