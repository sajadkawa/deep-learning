import matplotlib.pyplot as plt
import numpy as np
import os

# Parameter count for a fully connected network
# with fixed input n0=64, output nL=10
# comparing deep-narrow vs shallow-wide

n0 = 64
n_out = 10

def param_count(layer_sizes):
    total = 0
    for i in range(1, len(layer_sizes)):
        total += layer_sizes[i-1] * layer_sizes[i] + layer_sizes[i]
    return total

depths = list(range(1, 8))  # 1 to 7 hidden layers

# Deep narrow: width = 32 throughout hidden layers
deep_narrow = []
for d in depths:
    sizes = [n0] + [32] * d + [n_out]
    deep_narrow.append(param_count(sizes))

# Shallow wide: 1 hidden layer, width grows to match total neurons
# total neurons = 32 * d, so width = 32 * d
shallow_wide = []
for d in depths:
    width = 32 * d
    sizes = [n0, width, n_out]
    shallow_wide.append(param_count(sizes))

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(depths, deep_narrow, color='#2196F3', linewidth=2.5,
        marker='o', markersize=7, label='Deep narrow (width=32 per layer)')
ax.plot(depths, shallow_wide, color='#F44336', linewidth=2.5,
        marker='s', markersize=7, label='Shallow wide (1 hidden layer, same total neurons)')

ax.set_xlabel('Number of hidden layers', fontsize=12)
ax.set_ylabel('Total parameters', fontsize=12)
ax.set_title('Parameter count: deep narrow vs shallow wide', fontsize=13)
ax.legend(fontsize=11)
ax.set_xticks(depths)
ax.yaxis.get_major_formatter().set_scientific(False)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'param_count_depth_width.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print("Saved param_count_depth_width.png")
