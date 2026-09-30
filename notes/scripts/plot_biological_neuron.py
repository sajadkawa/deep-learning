import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# --- Cell body (soma) ---
soma = plt.Circle((5.5, 2.5), 0.9, color='#388bfd', zorder=3)
ax.add_patch(soma)
ax.text(5.5, 2.5, 'Soma\n(Cell Body)', ha='center', va='center',
        color='white', fontsize=9, fontweight='bold', zorder=4)

# Nucleus inside soma
nucleus = plt.Circle((5.5, 2.7), 0.3, color='#1f6feb', zorder=4)
ax.add_patch(nucleus)
ax.text(5.5, 2.7, 'N', ha='center', va='center', color='white', fontsize=7, zorder=5)

# --- Dendrites (left side) ---
dendrite_starts = [(1.5, 3.8), (1.5, 2.8), (1.5, 1.5), (2.2, 4.3), (2.2, 0.9)]
dendrite_ends   = [(4.6, 2.9), (4.6, 2.6), (4.6, 2.3), (4.6, 2.8), (4.6, 2.4)]
for (x0, y0), (x1, y1) in zip(dendrite_starts, dendrite_ends):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5))

ax.text(1.0, 2.5, 'Dendrites\n(receive signals)', ha='center', va='center',
        color='#3fb950', fontsize=9)

# --- Axon (right side) ---
ax.annotate('', xy=(9.5, 2.5), xytext=(6.4, 2.5),
            arrowprops=dict(arrowstyle='->', color='#f0883e', lw=2.5))
ax.text(7.9, 2.85, 'Axon', ha='center', va='bottom', color='#f0883e', fontsize=10, fontweight='bold')

# Myelin sheath bumps
for xm in [7.0, 7.6, 8.2, 8.8]:
    ellipse = mpatches.Ellipse((xm, 2.5), 0.45, 0.35, color='#f0883e', alpha=0.25, zorder=2)
    ax.add_patch(ellipse)

# --- Axon terminals ---
terminal_ends = [(10.2, 3.5), (10.2, 2.5), (10.2, 1.5)]
for xe, ye in terminal_ends:
    ax.annotate('', xy=(xe, ye), xytext=(9.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='#d2a8ff', lw=1.5))
    dot = plt.Circle((xe, ye), 0.12, color='#d2a8ff', zorder=4)
    ax.add_patch(dot)

ax.text(10.6, 2.5, 'Axon\nTerminals', ha='left', va='center', color='#d2a8ff', fontsize=9)

# --- Synapse label ---
ax.text(10.2, 0.9, '(synapse → next neuron)', ha='center', va='center',
        color='#8b949e', fontsize=8, style='italic')

ax.set_xlim(0, 11.5)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title('Structure of a Biological Neuron', color='#e6edf3', fontsize=13, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'biological_neuron.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
