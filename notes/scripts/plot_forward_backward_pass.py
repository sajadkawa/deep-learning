import matplotlib.pyplot as plt
import os

out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')

# ── SHARED HELPERS ─────────────────────────────────────────────────────────────

def node(ax, x, y, label, fc, ec, fontsize=10):
    c = plt.Circle((x, y), 0.42, facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=4)
    ax.add_patch(c)
    ax.text(x, y, label, ha='center', va='center', color='#111111',
            fontsize=fontsize, fontweight='bold', zorder=5)

def arr(ax, x0, y0, x1, y1, color='#9ca3af', label='', above=True, fontsize=8.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0), zorder=3)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        offset = 0.26 if above else -0.26
        ax.text(mx, my + offset, label, ha='center', va='center',
                color=color, fontsize=fontsize)

# ── FORWARD PASS ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 3.2))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

xs     = [1.2, 3.2, 5.2, 7.2, 9.2, 11.2]
labels = ['x', 'z⁽¹⁾', 'h', 'z⁽²⁾', 'ŷ', 'L']
colors_fc = ['#dbeafe', '#f9fafb', '#dcfce7', '#f9fafb', '#dbeafe', '#fee2e2']
colors_ec = ['#2563eb', '#6b7280', '#15803d', '#6b7280', '#2563eb', '#dc2626']

for x, lbl, fc, ec in zip(xs, labels, colors_fc, colors_ec):
    node(ax, x, 2.5, lbl, fc, ec)

edge_labels = ['', 'W⁽¹⁾ᵀx + b⁽¹⁾', 'σ(z⁽¹⁾)', 'W⁽²⁾ᵀh + b⁽²⁾', 'σ(z⁽²⁾)', 'BCE']
for i in range(len(xs) - 1):
    arr(ax, xs[i] + 0.42, 2.5, xs[i+1] - 0.42, 2.5, '#15803d', edge_labels[i+1])

# Store annotations below nodes
store_xs     = [3.2, 5.2, 7.2, 9.2]
store_labels = ['store z⁽¹⁾', 'store h', 'store z⁽²⁾', 'store ŷ']
for x, lbl in zip(store_xs, store_labels):
    ax.annotate('', xy=(x, 1.85), xytext=(x, 2.08),
                arrowprops=dict(arrowstyle='->', color='#b45309', lw=1.4), zorder=3)
    ax.text(x, 1.62, lbl, ha='center', va='top', color='#b45309', fontsize=8)

ax.set_xlim(0.4, 12.0)
ax.set_ylim(1.1, 3.4)
ax.axis('off')
ax.set_title('Forward Pass — compute and store intermediate values', color='#111111', fontsize=12, pad=10)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'forward_pass.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# ── BACKWARD PASS ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4.2))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Main flow right to left: L → ŷ → z⁽²⁾ → h → z⁽¹⁾ → x
xs2     = [11.2, 9.2, 7.2, 5.2, 3.2, 1.2]
labels2 = ['L', 'ŷ', 'z⁽²⁾', 'h', 'z⁽¹⁾', 'x']
colors_fc2 = ['#fee2e2', '#dbeafe', '#fff7ed', '#dcfce7', '#fff7ed', '#dbeafe']
colors_ec2 = ['#dc2626', '#2563eb', '#ea580c', '#15803d', '#ea580c', '#2563eb']

for x, lbl, fc, ec in zip(xs2, labels2, colors_fc2, colors_ec2):
    node(ax, x, 2.5, lbl, fc, ec)

# Arrow labels: show what quantity travels on each segment
# L→ŷ: ∂L/∂ŷ
# ŷ→z⁽²⁾: δ⁽²⁾ = ∂L/∂z⁽²⁾  (this is where delta is formed)
# z⁽²⁾→h: ∂L/∂h  (crosses layer boundary — not delta)
# h→z⁽¹⁾: δ⁽¹⁾ = ∂L/∂z⁽¹⁾  (layer 1 converts ∂L/∂h into its own delta)
# z⁽¹⁾→x: ∂L/∂x
grad_labels = ['∂L/∂ŷ', 'δ⁽²⁾ = ∂L/∂z⁽²⁾', '∂L/∂h', 'δ⁽¹⁾ = ∂L/∂z⁽¹⁾', '∂L/∂x']
for i in range(len(xs2) - 1):
    arr(ax, xs2[i] - 0.42, 2.5, xs2[i+1] + 0.42, 2.5, '#dc2626', grad_labels[i], fontsize=8)

# Job 1 branch at z⁽²⁾ — upward arrow + box
ax.annotate('', xy=(7.2, 3.35), xytext=(7.2, 2.92),
            arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1.5), zorder=3)
ax.text(7.2, 4.05, '∂L/∂W⁽²⁾ = δ⁽²⁾ · hᵀ\n∂L/∂b⁽²⁾ = δ⁽²⁾',
        ha='center', va='center', color='#7c3aed', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f3ff', edgecolor='#7c3aed', linewidth=1.0))

# Job 1 branch at z⁽¹⁾ — upward arrow + box
ax.annotate('', xy=(3.2, 3.35), xytext=(3.2, 2.92),
            arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1.5), zorder=3)
ax.text(3.2, 4.05, '∂L/∂W⁽¹⁾ = δ⁽¹⁾ · xᵀ\n∂L/∂b⁽¹⁾ = δ⁽¹⁾',
        ha='center', va='center', color='#7c3aed', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f3ff', edgecolor='#7c3aed', linewidth=1.0))

# Legend
ax.text(6.2, 1.55, '─── error signal / delta (main flow)', color='#dc2626', fontsize=8, ha='center')
ax.text(6.2, 1.22, '─── weight & bias gradients — Job 1 (stays local)', color='#7c3aed', fontsize=8, ha='center')

ax.set_xlim(0.4, 12.0)
ax.set_ylim(1.0, 4.8)
ax.axis('off')
ax.set_title('Backward Pass — reuse stored values, two jobs per layer', color='#111111', fontsize=12, pad=10)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'backward_pass.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
