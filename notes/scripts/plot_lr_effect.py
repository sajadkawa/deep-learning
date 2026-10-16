import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)
epochs = np.arange(1, 81)

def simulated_loss(eta_case, noise=0.02):
    if eta_case == 'too_small':
        loss = 2.0 * np.exp(-0.008 * epochs) + 0.5 + rng.normal(0, noise, len(epochs))
    elif eta_case == 'too_large':
        loss = np.where(epochs < 10,
                        2.0 - 0.1 * epochs + rng.normal(0, 0.15, len(epochs)),
                        1.5 + 0.8 * np.sin(0.5 * epochs) + rng.normal(0, 0.2, len(epochs)))
        loss = np.clip(loss, 0.3, 3.5)
    else:  # just right
        loss = 0.15 + 1.85 * np.exp(-0.07 * epochs) + rng.normal(0, noise, len(epochs))
        loss = np.clip(loss, 0.1, 2.5)
    return loss

configs = [
    ('too_large',  '#dc2626', 'η too large  — diverges / oscillates'),
    ('too_small',  '#b45309', 'η too small  — stalls'),
    ('just_right', '#15803d', 'η well-chosen — converges'),
]

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

for case, color, label in configs:
    loss = simulated_loss(case)
    ax.plot(epochs, loss, color=color, lw=2.2, label=label)

ax.set_xlabel('Epoch', color='#374151', fontsize=9)
ax.set_ylabel('Training Loss', color='#374151', fontsize=9)
ax.tick_params(colors='#374151')
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')
ax.legend(facecolor='white', edgecolor='#d1d5db', labelcolor='#111111', fontsize=9)
ax.set_title('Effect of Learning Rate on Training', color='#111111', fontsize=12, pad=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'lr_effect.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
