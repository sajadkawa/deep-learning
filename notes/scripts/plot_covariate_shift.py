import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

def relu(z):
    return np.maximum(0, z)

def batch_norm(z):
    mu = z.mean(axis=0)
    var = z.var(axis=0)
    return (z - mu) / np.sqrt(var + 1e-8)

def run_forward(n_layers, n_units, n_samples, use_bn):
    a = rng.standard_normal((n_samples, n_units))
    distributions = [a[:, 0].copy()]
    for _ in range(n_layers):
        W = rng.normal(0, np.sqrt(2.0 / n_units), (n_units, n_units))
        z = a @ W
        if use_bn:
            z = batch_norm(z)
        a = relu(z)
        distributions.append(a[:, 0].copy())
    return distributions

n_layers  = 5
n_units   = 256
n_samples = 2000

dists_no_bn = run_forward(n_layers, n_units, n_samples, use_bn=False)
dists_bn    = run_forward(n_layers, n_units, n_samples, use_bn=True)

fig, axes = plt.subplots(2, n_layers + 1, figsize=(14, 5))
fig.patch.set_facecolor('white')

titles     = ['Input'] + [f'Layer {i}' for i in range(1, n_layers + 1)]
row_labels = ['Without Batch Norm', 'With Batch Norm']
colors     = ['#dc2626', '#15803d']

for row, (dists, color, row_label) in enumerate(zip(
        [dists_no_bn, dists_bn], colors, row_labels)):
    for col, (dist, title) in enumerate(zip(dists, titles)):
        ax = axes[row, col]
        ax.set_facecolor('white')
        ax.hist(dist, bins=40, color=color, alpha=0.85, edgecolor='none')
        ax.set_title(title, color='#111111', fontsize=9, pad=4)
        ax.tick_params(colors='#374151', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#d1d5db')
        if col == 0:
            ax.set_ylabel(row_label, color='#374151', fontsize=9)
        ax.set_yticks([])

fig.suptitle('Internal Covariate Shift — Activation Distributions Across Layers',
             color='#111111', fontsize=12, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'covariate_shift.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
