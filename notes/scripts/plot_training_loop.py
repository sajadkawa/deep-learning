import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(4, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

boxes = [
    (0.5, 9.2, 'Training Data',                          '#dbeafe', '#2563eb'),
    (0.5, 7.8, '1. Forward Pass\nz = Wᵀx + b,  ŷ = f(z)', '#dcfce7', '#15803d'),
    (0.5, 6.4, '2. Compute Loss\nL = Loss(y, ŷ)',          '#fee2e2', '#dc2626'),
    (0.5, 5.0, '3. Backward Pass\nCompute ∂L/∂W via Chain Rule', '#ffedd5', '#ea580c'),
    (0.5, 3.6, '4. Parameter Update\nW ← W − η (∂L/∂W)', '#f3e8ff', '#7c3aed'),
    (0.5, 2.2, 'Converged?',                              '#f9fafb', '#6b7280'),
]

box_h = 0.55
box_w = 0.72

for x, y, label, fc, ec in boxes:
    rect = mpatches.FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                    boxstyle='round,pad=0.05',
                                    facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', color='#111111',
            fontsize=9.5, fontweight='bold', zorder=4)

# Arrows between boxes
arrow_ys = [(9.2 - box_h / 2, 7.8 + box_h / 2),
            (7.8 - box_h / 2, 6.4 + box_h / 2),
            (6.4 - box_h / 2, 5.0 + box_h / 2),
            (5.0 - box_h / 2, 3.6 + box_h / 2),
            (3.6 - box_h / 2, 2.2 + box_h / 2)]

for y0, y1 in arrow_ys:
    ax.annotate('', xy=(0.5, y1), xytext=(0.5, y0),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.5), zorder=2)

# "No" loop back arrow
ax.annotate('', xy=(0.5, 7.8 + box_h / 2), xytext=(0.5, 2.2 - box_h / 2),
            arrowprops=dict(arrowstyle='->', color='#ea580c', lw=1.5,
                            connectionstyle='arc3,rad=-0.5'), zorder=2)
ax.text(0.08, 5.0, 'No\n(repeat for\nE epochs)', ha='center', va='center',
        color='#ea580c', fontsize=8.5)

# "Yes" arrow
ax.annotate('', xy=(0.5, 1.0), xytext=(0.5, 2.2 - box_h / 2),
            arrowprops=dict(arrowstyle='->', color='#15803d', lw=1.5), zorder=2)
done = mpatches.FancyBboxPatch((0.5 - box_w / 2, 0.65), box_w, 0.45,
                                boxstyle='round,pad=0.05',
                                facecolor='#dcfce7', edgecolor='#15803d',
                                linewidth=1.8, zorder=3)
ax.add_patch(done)
ax.text(0.5, 0.875, 'Done — Model Trained', ha='center', va='center',
        color='#15803d', fontsize=9.5, fontweight='bold', zorder=4)
ax.text(0.65, 1.55, 'Yes', ha='left', va='center', color='#15803d', fontsize=9)

ax.set_xlim(0, 1)
ax.set_ylim(0.4, 9.8)
ax.axis('off')
ax.set_title('Universal Neural Network Training Loop',
             color='#111111', fontsize=12, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'training_loop.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
