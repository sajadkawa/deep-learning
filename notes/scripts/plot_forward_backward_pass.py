import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#0d1117')

def node(ax, x, y, label, fc, ec, fontsize=9):
    c = plt.Circle((x, y), 0.38, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=4)
    ax.add_patch(c)
    ax.text(x, y, label, ha='center', va='center', color='white',
            fontsize=fontsize, fontweight='bold', zorder=5)

def arr(ax, x0, y0, x1, y1, color='#8b949e', label='', above=True):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=3)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        offset = 0.22 if above else -0.22
        ax.text(mx, my + offset, label, ha='center', va='center',
                color=color, fontsize=8)

# ── FORWARD PASS ──────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#0d1117')

xs = [1.0, 2.8, 4.2, 5.6, 7.0, 8.4]
labels = ['x', 'z⁽¹⁾', 'h', 'z⁽²⁾', 'ŷ', 'L']
colors_fc = ['#1f6feb', '#161b22', '#1a3a1a', '#161b22', '#1f6feb', '#3a1a1a']
colors_ec = ['#388bfd', '#3fb950', '#3fb950', '#3fb950', '#388bfd', '#f85149']

for x, lbl, fc, ec in zip(xs, labels, colors_fc, colors_ec):
    node(ax, x, 2.5, lbl, fc, ec)

edge_labels = ['', 'W⁽¹⁾ᵀx+b⁽¹⁾', 'f(z⁽¹⁾)', 'W⁽²⁾ᵀh+b⁽²⁾', 'f(z⁽²⁾)', 'Loss']
for i in range(len(xs) - 1):
    arr(ax, xs[i] + 0.38, 2.5, xs[i+1] - 0.38, 2.5, '#3fb950', edge_labels[i+1])

# Store annotations
store_ys = [1.5, 1.5, 1.5, 1.5]
store_xs = [2.8, 4.2, 5.6, 7.0]
store_labels = ['store z⁽¹⁾', 'store h', 'store z⁽²⁾', 'store ŷ']
for x, lbl in zip(store_xs, store_labels):
    ax.annotate('', xy=(x, 1.88), xytext=(x, 2.12),
                arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.2), zorder=3)
    ax.text(x, 1.65, lbl, ha='center', va='top', color='#f0883e', fontsize=7.5)

ax.set_xlim(0.3, 9.2)
ax.set_ylim(1.0, 3.5)
ax.axis('off')
ax.set_title('Forward Pass — compute and store', color='#e6edf3', fontsize=11, pad=8)

# ── BACKWARD PASS ─────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('#0d1117')

xs2 = [8.4, 7.0, 5.6, 4.2, 2.8, 1.0]
labels2 = ['L', 'ŷ', 'z⁽²⁾', 'h', 'z⁽¹⁾', 'W⁽¹⁾']
colors_fc2 = ['#3a1a1a', '#1f6feb', '#161b22', '#1a3a1a', '#161b22', '#2a1a3a']
colors_ec2 = ['#f85149', '#388bfd', '#f0883e', '#3fb950', '#f0883e', '#d2a8ff']

for x, lbl, fc, ec in zip(xs2, labels2, colors_fc2, colors_ec2):
    node(ax, x, 2.5, lbl, fc, ec)

grad_labels = ['∂L/∂ŷ', '∂L/∂z⁽²⁾', '∂L/∂h', '∂L/∂z⁽¹⁾', '∂L/∂W⁽¹⁾']
for i in range(len(xs2) - 1):
    arr(ax, xs2[i] - 0.38, 2.5, xs2[i+1] + 0.38, 2.5, '#f0883e', grad_labels[i])

# Weight gradient branches
branch_xs = [5.6, 2.8]
branch_labels = ['→ ∂L/∂W⁽²⁾', '→ ∂L/∂b⁽¹⁾']
for x, lbl in zip(branch_xs, branch_labels):
    ax.annotate('', xy=(x, 3.2), xytext=(x, 2.88),
                arrowprops=dict(arrowstyle='->', color='#d2a8ff', lw=1.2), zorder=3)
    ax.text(x, 3.35, lbl, ha='center', va='bottom', color='#d2a8ff', fontsize=7.5)

ax.set_xlim(0.3, 9.2)
ax.set_ylim(1.8, 4.0)
ax.axis('off')
ax.set_title('Backward Pass — reuse stored values', color='#e6edf3', fontsize=11, pad=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'forward_backward_pass.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
