import os
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.1, 10, 300)

bias2    = 1.2 / x
variance = 0.05 * x ** 1.5
total    = bias2 + variance + 0.1
optimal  = x[np.argmin(total)]

fig, ax = plt.subplots(figsize=(6, 3.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(x, bias2,    color='#2563eb', linewidth=2, label='Bias²')
ax.plot(x, variance, color='#dc2626', linewidth=2, label='Variance')
ax.plot(x, total,    color='#15803d', linewidth=2, linestyle='--', label='Total Error')
ax.set_xlabel('Model Complexity →', color='#374151')
ax.set_ylabel('Error', color='#374151')
ax.set_xlim(0.1, 10)
ax.set_ylim(0, 2.2)
ax.tick_params(colors='#374151')
for spine in ax.spines.values():
    spine.set_edgecolor('#d1d5db')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.axvline(optimal, color='#9ca3af', linewidth=1, linestyle=':')
ax.text(optimal + 0.15, 2.0, 'Sweet spot', fontsize=9, color='#6b7280')
ax.legend(fontsize=9, facecolor='white', edgecolor='#d1d5db', labelcolor='#111111')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'bias_variance_tradeoff.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print('saved:', os.path.abspath(out))
