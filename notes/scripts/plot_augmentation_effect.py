import matplotlib.pyplot as plt
import numpy as np
import os

epochs = np.arange(1, 51)

# Without augmentation: training loss drops fast, val loss rises (overfitting)
rng = np.random.default_rng(0)
tr_no_aug  = 0.8 * np.exp(-0.12 * epochs) + 0.05 + rng.normal(0, 0.008, 50)
val_no_aug = 0.55 * np.exp(-0.06 * epochs) + 0.38 + 0.003 * epochs + rng.normal(0, 0.012, 50)

# With augmentation: both losses decrease, smaller gap
tr_aug  = 0.75 * np.exp(-0.09 * epochs) + 0.12 + rng.normal(0, 0.008, 50)
val_aug = 0.65 * np.exp(-0.09 * epochs) + 0.18 + rng.normal(0, 0.010, 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fig.patch.set_facecolor('white')

for ax, tr, val, title, color_tr, color_val in [
    (axes[0], tr_no_aug, val_no_aug, 'Without Augmentation', '#dc2626', '#2563eb'),
    (axes[1], tr_aug,    val_aug,    'With Augmentation',    '#dc2626', '#2563eb'),
]:
    ax.set_facecolor('white')
    ax.plot(epochs, tr,  color=color_tr,  lw=2,   label='Training loss')
    ax.plot(epochs, val, color=color_val, lw=2,   label='Validation loss')
    ax.set_title(title, fontsize=11, fontweight='bold', color='#111111')
    ax.set_xlabel('Epoch', fontsize=10, color='#374151')
    ax.set_ylabel('Loss', fontsize=10, color='#374151')
    ax.legend(fontsize=9)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, color='#d1d5db')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9ca3af')
    ax.spines['bottom'].set_color('#9ca3af')

# Annotate overfitting gap on left plot
axes[0].annotate('', xy=(49, float(val_no_aug[-1])), xytext=(49, float(tr_no_aug[-1])),
    arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.5))
axes[0].text(47, (float(val_no_aug[-1]) + float(tr_no_aug[-1])) / 2,
             'overfit\ngap', ha='right', va='center', fontsize=8.5, color='#374151')

axes[1].annotate('', xy=(49, float(val_aug[-1])), xytext=(49, float(tr_aug[-1])),
    arrowprops=dict(arrowstyle='<->', color='#374151', lw=1.5))
axes[1].text(47, (float(val_aug[-1]) + float(tr_aug[-1])) / 2,
             'smaller\ngap', ha='right', va='center', fontsize=8.5, color='#374151')

fig.suptitle('Effect of Data Augmentation on Learning Curves', fontsize=12, fontweight='bold', color='#111111')
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'augmentation_effect.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('Saved augmentation_effect.png')
