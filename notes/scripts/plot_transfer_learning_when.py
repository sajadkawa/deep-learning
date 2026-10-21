import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('white')

colors = {
    (0, 1): '#dbeafe',   # small + similar
    (1, 1): '#fee2e2',   # small + different
    (0, 0): '#dcfce7',   # large + similar
    (1, 0): '#fef3c7',   # large + different
}
borders = {
    (0, 1): '#2563eb',
    (1, 1): '#dc2626',
    (0, 0): '#15803d',
    (1, 0): '#b45309',
}

quads = [
    (0, 1, 0.5, 3.5, 4.5, 3.0,
     'Feature Extraction',
     'Freeze all backbone layers\nTrain new head only\n\nExample:\nCIFAR-10 (50k images)\nfrom ImageNet ResNet18'),
    (1, 1, 5.0, 3.5, 4.5, 3.0,
     'Feature Extraction\n(may underperform)',
     'Freeze all backbone layers\nTrain new head only\nFeatures may not transfer well\n\nExample:\nSatellite imagery (small)\nfrom ImageNet ResNet18'),
    (0, 0, 0.5, 0.5, 4.5, 3.0,
     'Fine-tune\n(last blocks)',
     'Freeze early layers\nUnfreeze last 1-2 blocks\nLow learning rate\n\nExample:\nCats vs Dogs (25k images)\nfrom ImageNet ResNet18'),
    (1, 0, 5.0, 0.5, 4.5, 3.0,
     'Fine-tune all\nor Train from Scratch',
     'Unfreeze all layers\nor random initialisation\n\nExample:\nMedical X-rays (large)\nfrom ImageNet or scratch'),
]

for col, row, x0, y0, w, h, strategy, detail in quads:
    rect = mpatches.FancyBboxPatch((x0, y0), w, h,
        boxstyle='round,pad=0.1', facecolor=colors[(col, row)], edgecolor=borders[(col, row)], linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x0+w/2, y0+h-0.45, strategy, ha='center', va='center',
            fontsize=10, fontweight='bold', color='#111111')
    ax.text(x0+w/2, y0+h/2-0.3, detail, ha='center', va='center',
            fontsize=8, color='#374151', linespacing=1.5)

ax.text(5.0, 6.75, 'Domain Similarity to ImageNet', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#111111')
ax.text(2.75, 6.45, 'Similar domain', ha='center', va='center', fontsize=9.5, color='#374151')
ax.text(7.25, 6.45, 'Different domain', ha='center', va='center', fontsize=9.5, color='#374151')
ax.text(0.2, 3.5, 'Dataset Size', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#111111', rotation=90)
ax.text(0.45, 5.0, 'Large', ha='center', va='center', fontsize=9.5, color='#374151', rotation=90)
ax.text(0.45, 2.0, 'Small', ha='center', va='center', fontsize=9.5, color='#374151', rotation=90)

ax.plot([5.0, 5.0], [0.5, 6.3], color='#d1d5db', lw=1.5, linestyle='--')
ax.plot([0.5, 9.5], [3.5, 3.5], color='#d1d5db', lw=1.5, linestyle='--')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'transfer_learning_when.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved transfer_learning_when.png')
