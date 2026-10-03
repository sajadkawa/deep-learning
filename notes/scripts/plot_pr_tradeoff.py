import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

n_pos, n_neg = 100, 200
scores_pos = rng.beta(5, 2, n_pos)
scores_neg = rng.beta(2, 5, n_neg)
scores = np.concatenate([scores_pos, scores_neg])
labels = np.array([1] * n_pos + [0] * n_neg)

thresholds = np.linspace(0.01, 0.99, 200)
precisions, recalls = [], []

for t in thresholds:
    preds = (scores >= t).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

precisions = np.array(precisions)
recalls    = np.array(recalls)

fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

ax.plot(thresholds, precisions, color='#388bfd', lw=2.5, label='Precision')
ax.plot(thresholds, recalls,    color='#3fb950', lw=2.5, label='Recall')

# Mark crossover point
diff = np.abs(precisions - recalls)
cross_idx = diff.argmin()
ax.axvline(thresholds[cross_idx], color='#f0883e', lw=1.5, linestyle='--',
           label=f'Crossover  (threshold ≈ {thresholds[cross_idx]:.2f})')
ax.scatter([thresholds[cross_idx]], [precisions[cross_idx]],
           color='#f0883e', s=60, zorder=5)

ax.set_xlabel('Classification Threshold', color='#8b949e', fontsize=9)
ax.set_ylabel('Score', color='#8b949e', fontsize=9)
ax.tick_params(colors='#8b949e')
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')
ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3', fontsize=9)
ax.set_title('Precision-Recall Tradeoff vs Classification Threshold',
             color='#e6edf3', fontsize=12, pad=8)

# Annotations
ax.annotate('Lower threshold\n→ higher recall\n→ lower precision',
            xy=(0.2, recalls[int(0.2 * len(thresholds))]),
            xytext=(0.08, 0.35),
            color='#8b949e', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.0))
ax.annotate('Higher threshold\n→ higher precision\n→ lower recall',
            xy=(0.8, precisions[int(0.8 * len(thresholds))]),
            xytext=(0.62, 0.35),
            color='#8b949e', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.0))

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'pr_tradeoff.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
