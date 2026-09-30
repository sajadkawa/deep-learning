import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def forward_and_backward(layer_sizes, init_std, n_samples=500):
    """Forward pass then one backward pass. Return gradient std per layer."""
    # Forward pass — store z and a at each layer
    a = rng.standard_normal((n_samples, layer_sizes[0]))
    weights = []
    zs = []
    acts = [a]
    for i in range(1, len(layer_sizes)):
        n_in  = layer_sizes[i - 1]
        n_out = layer_sizes[i]
        W = rng.normal(0, init_std(n_in, n_out), (n_in, n_out))
        weights.append(W)
        z = acts[-1] @ W
        zs.append(z)
        acts.append(relu(z))

    # Backward pass — random upstream gradient from output
    L = len(weights)
    delta = rng.standard_normal((n_samples, layer_sizes[-1]))
    grad_stds = []
    for l in range(L - 1, -1, -1):
        dW = acts[l].T @ delta / n_samples
        grad_stds.insert(0, dW.std())
        delta = delta @ weights[l].T * relu_grad(zs[l])

    return grad_stds

layer_sizes = [500] + [500] * 6 + [500]

configs = [
    ('Too small  (std=0.01)', lambda n_in, n_out: 0.01,                '#f85149'),
    ('Too large  (std=1.0)',  lambda n_in, n_out: 1.0,                 '#f0883e'),
    ('He init',              lambda n_in, n_out: np.sqrt(2.0 / n_in), '#3fb950'),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)
fig.patch.set_facecolor('#0d1117')

for ax, (label, init_fn, color) in zip(axes, configs):
    ax.set_facecolor('#0d1117')
    grad_stds = forward_and_backward(layer_sizes, init_fn)
    layers = list(range(1, len(grad_stds) + 1))
    ax.plot(layers, grad_stds, color=color, lw=2.5, marker='o', markersize=5)
    ax.axhline(0, color='#30363d', lw=0.8)
    ax.set_xlabel('Layer (1 = first hidden)', color='#8b949e', fontsize=9)
    ax.set_ylabel('Gradient std', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(label, color='#e6edf3', fontsize=10, pad=6)
    ax.set_xticks(layers)

fig.suptitle('Gradient Standard Deviation Across Layers — Effect of Initialization',
             color='#e6edf3', fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'init_gradient_flow.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
