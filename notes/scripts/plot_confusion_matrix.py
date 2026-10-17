import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.axis('off')

colors = {
    'TP': '#dcfce7',
    'FN': '#fee2e2',
    'FP': '#fee2e2',
    'TN': '#dcfce7',
}
borders = {
    'TP': '#15803d',
    'FN': '#dc2626',
    'FP': '#dc2626',
    'TN': '#15803d',
}
values = {
    'TP': ('28', 'True Positive', 'Predicted +, Actually +'),
    'FN': ('12', 'False Negative', 'Predicted −, Actually +'),
    'FP': ('8',  'False Positive', 'Predicted +, Actually −'),
    'TN': ('52', 'True Negative',  'Predicted −, Actually −'),
}
cells = {
    'TP': (1.0, 2), 'FN': (2.1, 2),
    'FP': (1.0, 1), 'TN': (2.1, 1),
}

for key, (col, row) in cells.items():
    rect = mpatches.FancyBboxPatch(
        (col, row), 0.95, 0.95,
        boxstyle='round,pad=0.04',
        facecolor=colors[key], edgecolor=borders[key], linewidth=1.5
    )
    ax.add_patch(rect)
    count, label, desc = values[key]
    ax.text(col + 0.475, row + 0.65, count,
            ha='center', va='center', color='#111111', fontsize=22, fontweight='bold')
    ax.text(col + 0.475, row + 0.38, label,
            ha='center', va='center', color='#111111', fontsize=9, fontweight='bold')
    ax.text(col + 0.475, row + 0.16, desc,
            ha='center', va='center', color='#374151', fontsize=7.5)

ax.text(1.475, 3.2, 'Predicted Positive', ha='center', color='#2563eb', fontsize=10, fontweight='bold')
ax.text(2.575, 3.2, 'Predicted Negative', ha='center', color='#2563eb', fontsize=10, fontweight='bold')

ax.text(0.85, 2.475, 'Actual\nPositive', ha='right', va='center', color='#2563eb', fontsize=10, fontweight='bold')
ax.text(0.85, 1.475, 'Actual\nNegative', ha='right', va='center', color='#2563eb', fontsize=10, fontweight='bold')

ax.text(2.0, 0.65, 'Accuracy = (TP+TN)/total = 80%     Precision = TP/(TP+FP) = 78%     Recall = TP/(TP+FN) = 70%',
        ha='center', color='#374151', fontsize=8.5)

ax.set_title('Confusion Matrix — 100 test samples, 40 positive, 60 negative',
             color='#111111', fontsize=11, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'confusion_matrix.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
