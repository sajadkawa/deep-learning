import numpy as np
import matplotlib.pyplot as plt
import os

epochs = np.arange(0, 90)

# Step decay: η₀=0.1, γ=0.1, k=30
eta0_step = 0.1
gamma     = 0.1
k         = 30
step_decay = eta0_step * (gamma ** (epochs // k))

# Cosine annealing: η₀=0.1, η_min=0.001, T=90
eta0_cos  = 0.1
eta_min   = 0.001
T         = 90
cosine    = eta_min + 0.5 * (eta0_cos - eta_min) * (1 + np.cos(np.pi * epochs / T))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor('#0d1117')

configs = [
    (axes[0], step_decay,  '#f0883e', 'Step Decay',
     r'$\eta(t) = \eta_0 \times \gamma^{\lfloor t/k \rfloor}$'
     '\n' + r'$\eta_0=0.1,\ \gamma=0.1,\ k=30$'),
    (axes[1], cosine,      '#388bfd', 'Cosine Annealing',
     r'$\eta(t) = \eta_{min} + \frac{1}{2}(\eta_0 - \eta_{min})(1 + \cos(\pi t / T))$'
     '\n' + r'$\eta_0=0.1,\ \eta_{min}=0.001,\ T=90$'),
]

for ax, lr, color, title, formula in configs:
    ax.set_facecolor('#0d1117')
    ax.plot(epochs, lr, color=color, lw=2.5)
    ax.set_xlabel('Epoch', color='#8b949e', fontsize=9)
    ax.set_ylabel('Learning rate η', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(title, color='#e6edf3', fontsize=11, pad=8)
    ax.text(0.97, 0.95, formula, transform=ax.transAxes,
            ha='right', va='top', color='#8b949e', fontsize=8.5,
            linespacing=1.5)
    ax.set_ylim(bottom=0)

fig.suptitle('Learning Rate Schedules', color='#e6edf3', fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'lr_schedules.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
