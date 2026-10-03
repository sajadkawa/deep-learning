import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.axis('off')

# Cell colours
colors = {
    'TP': '#1f4e2e',
    'FN': '#4e1f1f',
    'FP': '#4e1f1f',
    'TN': '#1f4e2e',
}
values = {
    'TP': ('28', 'True Positive', 'Predicted +, Actually +'),
    'FN': ('12', 'False Negative', 'Predicted −, Actually +'),
    'FP': ('8',  'False Positive', 'Predicted +, Actually −'),
    'TN': ('52', 'True Negative',  'Predicted −, Actually −'),
}
cells = {
    'TP': (2, 2), 'FN': (3, 2),
    'FP': (2, 1), 'TN': (3, 1),
}

for key, (col, row) in cells.items():
    rect = mpatches.FancyBboxPatch(
        (col, row), 0.95, 0.95,
        boxstyle='round,pad=0.04',
        facecolor=colors[key], edgecolor='#30363d', linewidth=1.5
    )
    ax.add_patch(rect)
    count, label, desc = values[key]
    ax.text(col + 0.475, row + 0.65, count,
            ha='center', va='center', color='#e6edf3', fontsize=22, fontweight='bold')
    ax.text(col + 0.475, row + 0.38, label,
            ha='center', va='center', color='#e6edf3', fontsize=9, fontweight='bold')
    ax.text(col + 0.475, row + 0.16, desc,
            ha='center', va='center', color='#8b949e', fontsize=7.5)

# Column headers
ax.text(2.475, 3.2, 'Predicted Positive', ha='center', color='#388bfd', fontsize=10, fontweight='bold')
ax.text(3.475, 3.2, 'Predicted Negative', ha='center', color='#388bfd', fontsize=10, fontweight='bold')

# Row headers
ax.text(1.85, 2.475, 'Actual\nPositive', ha='right', va='center', color='#388bfd', fontsize=10, fontweight='bold')
ax.text(1.85, 1.475, 'Actual\nNegative', ha='right', va='center', color='#388bfd', fontsize=10, fontweight='bold')

# Summary stats
ax.text(2.0, 0.7, 'Accuracy = (TP + TN) / total = (28 + 52) / 100 = 80%',
        ha='center', color='#8b949e', fontsize=9)

ax.set_title('Confusion Matrix — 100 test samples, 40 positive, 60 negative',
             color='#e6edf3', fontsize=11, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'confusion_matrix.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
