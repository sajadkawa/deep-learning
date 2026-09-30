import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(6, 10))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

boxes = [
    (0.5, 9.2, 'Training Data', '#1f6feb', '#388bfd'),
    (0.5, 7.8, '1. Forward Pass\nz = Wᵀx + b,  ŷ = f(z)', '#1a3a1a', '#3fb950'),
    (0.5, 6.4, '2. Compute Loss\nL = Loss(y, ŷ)', '#3a1a1a', '#f85149'),
    (0.5, 5.0, '3. Backward Pass\nCompute ∂L/∂W via Chain Rule', '#3a2a1a', '#f0883e'),
    (0.5, 3.6, '4. Parameter Update\nW ← W − η (∂L/∂W)', '#2a1a3a', '#d2a8ff'),
    (0.5, 2.2, 'Converged?', '#21262d', '#8b949e'),
]

box_h = 0.55
box_w = 0.72

for x, y, label, fc, ec in boxes:
    rect = mpatches.FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                    boxstyle='round,pad=0.05',
                                    facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', color='#e6edf3',
            fontsize=9.5, fontweight='bold', zorder=4)

# Arrows between boxes
arrow_xs = [0.5] * 5
arrow_ys = [(9.2 - box_h / 2, 7.8 + box_h / 2),
            (7.8 - box_h / 2, 6.4 + box_h / 2),
            (6.4 - box_h / 2, 5.0 + box_h / 2),
            (5.0 - box_h / 2, 3.6 + box_h / 2),
            (3.6 - box_h / 2, 2.2 + box_h / 2)]

for x, (y0, y1) in zip(arrow_xs, arrow_ys):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5), zorder=2)

# "No" loop back arrow
ax.annotate('', xy=(0.5, 7.8 + box_h / 2), xytext=(0.5, 2.2 - box_h / 2),
            arrowprops=dict(arrowstyle='->', color='#f0883e', lw=1.5,
                            connectionstyle='arc3,rad=-0.5'), zorder=2)
ax.text(0.08, 5.0, 'No\n(repeat for\nE epochs)', ha='center', va='center',
        color='#f0883e', fontsize=8.5)

# "Yes" arrow
ax.annotate('', xy=(0.5, 1.0), xytext=(0.5, 2.2 - box_h / 2),
            arrowprops=dict(arrowstyle='->', color='#3fb950', lw=1.5), zorder=2)
done = mpatches.FancyBboxPatch((0.5 - box_w / 2, 0.65), box_w, 0.45,
                                boxstyle='round,pad=0.05',
                                facecolor='#1a3a1a', edgecolor='#3fb950',
                                linewidth=1.8, zorder=3)
ax.add_patch(done)
ax.text(0.5, 0.875, 'Done — Model Trained', ha='center', va='center',
        color='#3fb950', fontsize=9.5, fontweight='bold', zorder=4)
ax.text(0.65, 1.55, 'Yes', ha='left', va='center', color='#3fb950', fontsize=9)

ax.set_xlim(0, 1)
ax.set_ylim(0.4, 9.8)
ax.axis('off')
ax.set_title('Universal Neural Network Training Loop', color='#e6edf3', fontsize=12, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'training_loop.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
