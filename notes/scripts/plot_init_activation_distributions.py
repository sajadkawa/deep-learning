import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def forward_activations(layer_sizes, init_std, activation, n_samples=1000):
    """Run a forward pass and collect activation std per layer."""
    a = rng.standard_normal((n_samples, layer_sizes[0]))
    stds = [a.std()]
    for i in range(1, len(layer_sizes)):
        n_in  = layer_sizes[i - 1]
        n_out = layer_sizes[i]
        W = rng.normal(0, init_std(n_in, n_out), (n_in, n_out))
        z = a @ W
        a = activation(z)
        stds.append(a.std())
    return stds

layer_sizes = [500] + [500] * 8   # 8 hidden layers

configs = [
    ('Too small  (std=0.01)',  lambda n_in, n_out: 0.01,                    relu,  '#f85149'),
    ('Too large  (std=1.0)',   lambda n_in, n_out: 1.0,                     relu,  '#f0883e'),
    ('He init',               lambda n_in, n_out: np.sqrt(2.0 / n_in),     relu,  '#3fb950'),
    ('Xavier init (tanh)',    lambda n_in, n_out: np.sqrt(1.0 / n_in),     tanh,  '#388bfd'),
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)
fig.patch.set_facecolor('#0d1117')

for ax, (label, init_fn, act_fn, color) in zip(axes, configs):
    ax.set_facecolor('#0d1117')
    stds = forward_activations(layer_sizes, init_fn, act_fn)
    layers = list(range(len(stds)))
    ax.plot(layers, stds, color=color, lw=2.5, marker='o', markersize=5)
    ax.axhline(0, color='#30363d', lw=0.8)
    ax.set_xlabel('Layer', color='#8b949e', fontsize=9)
    ax.set_ylabel('Activation std', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(label, color='#e6edf3', fontsize=10, pad=6)
    ax.set_xticks(layers)

fig.suptitle('Activation Standard Deviation Across Layers — Effect of Initialization',
             color='#e6edf3', fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'init_activation_distributions.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
