import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#1e1e1e')

n_visible = 4
n_hidden  = 3

v_x = 1.5
h_x = 6.5
v_ys = [1.0, 2.0, 3.0, 4.0]
h_ys = [1.5, 2.5, 3.5]

node_radius = 0.35
v_color = '#4fc3f7'
h_color = '#ce93d8'
edge_color = '#888888'
text_color = '#eeeeee'

# Edges — all v to all h
for vy in v_ys:
    for hy in h_ys:
        ax.plot([v_x, h_x], [vy, hy], color=edge_color, linewidth=0.9, zorder=1)

# Visible nodes
for i, vy in enumerate(v_ys):
    circle = plt.Circle((v_x, vy), node_radius, color=v_color, zorder=3)
    ax.add_patch(circle)
    ax.text(v_x, vy, f'v{i+1}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1e1e1e', zorder=4)

# Hidden nodes
for j, hy in enumerate(h_ys):
    circle = plt.Circle((h_x, hy), node_radius, color=h_color, zorder=3)
    ax.add_patch(circle)
    ax.text(h_x, hy, f'h{j+1}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#1e1e1e', zorder=4)

# Layer labels
ax.text(v_x, 0.2, 'Visible layer', ha='center', va='center',
        fontsize=11, color=v_color)
ax.text(h_x, 0.2, 'Hidden layer', ha='center', va='center',
        fontsize=11, color=h_color)

# Weight label on one edge
mid_x = (v_x + h_x) / 2
ax.text(mid_x, 3.55, 'w', ha='center', va='bottom',
        fontsize=12, color='#aaaaaa', style='italic')

# No within-layer connections annotation
ax.text(4.0, 4.6, 'No within-layer connections', ha='center', va='center',
        fontsize=9, color='#aaaaaa', style='italic')

ax.set_xlim(0, 8)
ax.set_ylim(-0.2, 5.1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'rbm_architecture.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved rbm_architecture.png")
