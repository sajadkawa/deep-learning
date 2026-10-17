import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

n_visible = 4
n_hidden  = 3

v_x = 1.5
h_x = 6.5
v_ys = [1.0, 2.0, 3.0, 4.0]
h_ys = [1.5, 2.5, 3.5]

node_radius = 0.35

for vy in v_ys:
    for hy in h_ys:
        ax.plot([v_x, h_x], [vy, hy], color='#d1d5db', linewidth=0.9, zorder=1)

for i, vy in enumerate(v_ys):
    circle = plt.Circle((v_x, vy), node_radius, facecolor='#dbeafe',
                         edgecolor='#2563eb', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(v_x, vy, f'v{i+1}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#111111', zorder=4)

for j, hy in enumerate(h_ys):
    circle = plt.Circle((h_x, hy), node_radius, facecolor='#fef3c7',
                         edgecolor='#b45309', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(h_x, hy, f'h{j+1}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#111111', zorder=4)

ax.text(v_x, 0.2, 'Visible layer', ha='center', va='center',
        fontsize=11, color='#2563eb')
ax.text(h_x, 0.2, 'Hidden layer', ha='center', va='center',
        fontsize=11, color='#b45309')

mid_x = (v_x + h_x) / 2
ax.text(mid_x, 3.55, 'w', ha='center', va='bottom',
        fontsize=12, color='#6b7280', style='italic')

ax.text(4.0, 4.6, 'No within-layer connections', ha='center', va='center',
        fontsize=9, color='#6b7280', style='italic')

ax.set_xlim(0, 8)
ax.set_ylim(-0.2, 5.1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'rbm_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved rbm_architecture.png")
