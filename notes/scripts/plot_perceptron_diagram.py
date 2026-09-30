import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

def circle(ax, x, y, r, fc, ec, label, fontsize=10):
    c = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=4)
    ax.add_patch(c)
    ax.text(x, y, label, ha='center', va='center', color='white',
            fontsize=fontsize, fontweight='bold', zorder=5)

def arrow(ax, x0, y0, x1, y1, label='', color='#8b949e'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=3)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.18, label, ha='center', va='bottom',
                color='#f0883e', fontsize=9)

# Input nodes
input_ys = [4.0, 2.5, 1.0]
input_labels = ['x₁', 'x₂', 'xₙ']
weight_labels = ['w₁', 'w₂', 'wₙ']
for y, lbl in zip(input_ys, input_labels):
    circle(ax, 1.5, y, 0.35, '#1f6feb', '#388bfd', lbl)

ax.text(1.5, 1.75, '⋮', ha='center', va='center', color='#8b949e', fontsize=14)

# Bias node
circle(ax, 3.5, 4.8, 0.3, '#21262d', '#8b949e', 'b', fontsize=9)
arrow(ax, 3.5, 4.5, 4.3, 3.1, '', '#8b949e')

# Sum node
circle(ax, 4.5, 2.5, 0.5, '#161b22', '#3fb950', 'Σwᵢxᵢ+b', fontsize=8)

# Activation node
circle(ax, 7.0, 2.5, 0.5, '#161b22', '#d2a8ff', 'f(z)', fontsize=10)

# Output node
circle(ax, 9.5, 2.5, 0.35, '#1f6feb', '#388bfd', 'ŷ')

# Arrows: inputs → sum
for y, wlbl in zip(input_ys, weight_labels):
    arrow(ax, 1.85, y, 4.0, 2.5, wlbl)

# Sum → activation
arrow(ax, 5.0, 2.5, 6.5, 2.5, 'z')

# Activation → output
arrow(ax, 7.5, 2.5, 9.15, 2.5)

# Labels
ax.text(1.5, 0.45, 'Real-valued\ninputs', ha='center', color='#8b949e', fontsize=8)
ax.text(4.5, 1.75, 'Weighted sum\n(learnable weights)', ha='center', color='#8b949e', fontsize=8)
ax.text(7.0, 1.75, 'Activation\nfunction', ha='center', color='#8b949e', fontsize=8)
ax.text(9.5, 1.95, 'Output\nŷ = f(z)', ha='center', color='#8b949e', fontsize=8)

ax.set_xlim(0.5, 10.5)
ax.set_ylim(0.2, 5.5)
ax.axis('off')
ax.set_title('Perceptron (Artificial Neuron)', color='#e6edf3', fontsize=13, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'perceptron_diagram.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
