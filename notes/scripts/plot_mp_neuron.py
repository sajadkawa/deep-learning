import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

node_kw = dict(zorder=4, linewidth=1.5)

def circle(ax, x, y, r, fc, ec, label, fontsize=10):
    c = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec, **node_kw)
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

# Dots between x2 and xn
ax.text(1.5, 1.75, '⋮', ha='center', va='center', color='#8b949e', fontsize=14)

# Sum node
circle(ax, 4.5, 2.5, 0.5, '#161b22', '#3fb950', 'Σwᵢxᵢ', fontsize=9)

# Threshold node
circle(ax, 7.0, 2.5, 0.5, '#161b22', '#f0883e', 'z ≥ θ?', fontsize=9)

# Output node
circle(ax, 9.5, 2.5, 0.35, '#1f6feb', '#388bfd', 'ŷ')

# Arrows: inputs → sum
for y, wlbl in zip(input_ys, weight_labels):
    arrow(ax, 1.85, y, 4.0, 2.5, wlbl)

# Sum → threshold
arrow(ax, 5.0, 2.5, 6.5, 2.5, 'z')

# Threshold → output
arrow(ax, 7.5, 2.5, 9.15, 2.5)
ax.text(8.3, 2.75, '{0,1}', ha='center', va='bottom', color='#8b949e', fontsize=9)

# Labels below nodes
ax.text(1.5, 0.45, 'Binary inputs\n(0 or 1)', ha='center', color='#8b949e', fontsize=8)
ax.text(4.5, 1.8, 'Weighted sum\n(fixed weights)', ha='center', color='#8b949e', fontsize=8)
ax.text(7.0, 1.8, 'Threshold θ\n(fixed)', ha='center', color='#8b949e', fontsize=8)
ax.text(9.5, 1.95, 'Output', ha='center', color='#8b949e', fontsize=8)

ax.set_xlim(0.5, 10.5)
ax.set_ylim(0.2, 5.0)
ax.axis('off')
ax.set_title('McCulloch-Pitts Neuron', color='#e6edf3', fontsize=13, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'mp_neuron.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
