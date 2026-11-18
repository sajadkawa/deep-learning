import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.5)
ax.axis('off')
fig.patch.set_facecolor('white')

TEXT_MAIN = '#111111'
TEXT_SUB  = '#374151'
ARROW     = '#9ca3af'

C = ['#dbeafe', '#fce7f3', '#fef3c7', '#dcfce7', '#f3e8ff', '#fef9c3']
B = ['#93c5fd', '#f9a8d4', '#fcd34d', '#86efac', '#c084fc', '#fcd34d']

BOX_W = 1.8
BOX_H = 0.75
Y     = 2.25

labels = [
    ('Medical\nImage', 'X-ray / CT\nMRI / Histo'),
    ('Pre-\nprocessing', 'Resize, normalise\naugment'),
    ('CNN\nBackbone', 'ResNet / VGG\n(pre-trained)'),
    ('Task\nHead', 'Classify / Detect\nSegment'),
    ('Prediction', 'Disease label\nbounding box'),
    ('Explanation', 'Grad-CAM\nheatmap'),
]

xs = [1.2, 3.3, 5.4, 7.5, 9.6, 11.7]

for i, (x, (lbl, sub)) in enumerate(zip(xs, labels)):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - BOX_W/2, Y - BOX_H/2), BOX_W, BOX_H,
        boxstyle='round,pad=0.07',
        facecolor=C[i], edgecolor=B[i], linewidth=1.5, zorder=2))
    ax.text(x, Y + 0.14, lbl, ha='center', va='center',
            fontsize=8.5, color=TEXT_MAIN, fontweight='bold', zorder=3)
    ax.text(x, Y - 0.22, sub, ha='center', va='center',
            fontsize=7, color=TEXT_SUB, zorder=3)

# Arrows between boxes
for i in range(len(xs) - 1):
    ax.annotate('', xy=(xs[i+1] - BOX_W/2, Y),
                xytext=(xs[i] + BOX_W/2, Y),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.5))

# ── Class imbalance annotation ────────────────────────────────────────────────
ax.annotate('', xy=(xs[0], Y - BOX_H/2 - 0.05),
            xytext=(xs[0], Y - BOX_H/2 - 0.65),
            arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.2))
ax.text(xs[0], Y - BOX_H/2 - 0.82,
        'Class imbalance\n→ cost-sensitive loss',
        ha='center', va='top', fontsize=7, color='#b45309')

# ── Transfer learning annotation ─────────────────────────────────────────────
ax.annotate('', xy=(xs[2], Y - BOX_H/2 - 0.05),
            xytext=(xs[2], Y - BOX_H/2 - 0.65),
            arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.2))
ax.text(xs[2], Y - BOX_H/2 - 0.82,
        'Transfer learning\nfrom ImageNet',
        ha='center', va='top', fontsize=7, color='#2563eb')

# ── Grad-CAM annotation ───────────────────────────────────────────────────────
ax.annotate('', xy=(xs[5], Y - BOX_H/2 - 0.05),
            xytext=(xs[5], Y - BOX_H/2 - 0.65),
            arrowprops=dict(arrowstyle='->', color='#15803d', lw=1.2))
ax.text(xs[5], Y - BOX_H/2 - 0.82,
        'Interpretability\nGrad-CAM',
        ha='center', va='top', fontsize=7, color='#15803d')

ax.set_title('Medical Image Analysis Pipeline', fontsize=12,
             color=TEXT_MAIN, fontweight='bold', pad=8)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'medical_imaging_pipeline.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved medical_imaging_pipeline.png')
