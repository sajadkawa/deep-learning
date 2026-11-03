import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(42)

# Simulate a binary classifier — scores for positives and negatives
n_pos, n_neg = 100, 200
scores_pos = rng.beta(5, 2, n_pos)    # positives tend to score high
scores_neg = rng.beta(2, 5, n_neg)    # negatives tend to score low
scores  = np.concatenate([scores_pos, scores_neg])
labels  = np.array([1] * n_pos + [0] * n_neg)

# Compute ROC and PR curves by varying threshold
thresholds = np.linspace(0, 1, 300)
tprs, fprs, precisions, recalls = [], [], [], []

for t in thresholds:
    preds = (scores >= t).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0

    tprs.append(tpr)
    fprs.append(fpr)
    precisions.append(prec)
    recalls.append(rec)

tprs = np.array(tprs)
fprs = np.array(fprs)
precisions = np.array(precisions)
recalls    = np.array(recalls)

# AUC (trapezoidal)
roc_auc = -np.trapezoid(tprs, fprs)

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(fprs, tprs, color='#2563eb', lw=2.5, label=f'Classifier  (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='#9ca3af', lw=1.2, linestyle='--', label='Random  (AUC = 0.50)')
ax.fill_between(fprs, tprs, alpha=0.08, color='#2563eb')
ax.set_xlabel('False Positive Rate (FPR)', color='#374151', fontsize=10)
ax.set_ylabel('True Positive Rate (TPR = Recall)', color='#374151', fontsize=10)
ax.set_title('ROC Curve and AUC', color='#111111', fontsize=12, pad=10)
ax.tick_params(colors='#374151')
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')
ax.legend(facecolor='white', edgecolor='#d1d5db', labelcolor='#111111', fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'roc_pr_curves.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
