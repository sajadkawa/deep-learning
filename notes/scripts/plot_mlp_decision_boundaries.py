import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

np.random.seed(42)

c1    = '#1d4ed8'   # class 1 blue
c0    = '#b91c1c'   # class 0 red
bound = '#ca8a04'   # boundary amber

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#d1d5db')
    ax.tick_params(colors='#374151', labelsize=9)

def style_ax(ax, title):
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.22, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('x₁', color='#374151', fontsize=11, labelpad=4)
    ax.set_ylabel('x₂', color='#374151', fontsize=11, labelpad=4)
    ax.set_title(title, color='#111111', fontsize=10.5, pad=10)

def scatter_two_class(ax, pts1, pts0):
    ax.scatter(pts1[:,0], pts1[:,1], color=c1, s=55, zorder=5,
               edgecolors='white', linewidths=0.6)
    ax.scatter(pts0[:,0], pts0[:,1], color=c0, s=55, zorder=5,
               edgecolors='white', linewidths=0.6)

# ── Panel 1: Single neuron — one hyperplane ───────────────────────────────────
ax = axes[0]
pts1_a = np.random.rand(18, 2) * 0.45 + np.array([0.55, 0.55])
pts0_a = np.random.rand(18, 2) * 0.45 + np.array([0.0,  0.0 ])
scatter_two_class(ax, pts1_a, pts0_a)

x_line = np.linspace(0, 1, 100)
ax.plot(x_line, 0.9 - x_line, color=bound, linewidth=2.0, zorder=4)
ax.fill_between(x_line, 0.9 - x_line, 1.1, alpha=0.08, color=c1)
ax.fill_between(x_line, -0.1, 0.9 - x_line, alpha=0.08, color=c0)
style_ax(ax, '0 Hidden Layers\n(Single Perceptron)')
ax.text(0.5, -0.14, 'One hyperplane\n(half-space boundary)',
        ha='center', color='#6b7280', fontsize=8.5, style='italic')

# ── Panel 2: 1 hidden layer — convex region ───────────────────────────────────
ax = axes[1]
centre = np.array([0.5, 0.5])
angles = np.random.uniform(0, 2*np.pi, 20)
radii  = np.random.uniform(0.05, 0.18, 20)
pts1_b = np.clip(centre + np.stack([radii*np.cos(angles),
                                     radii*np.sin(angles)], axis=1), 0.05, 0.95)
pts0_b = np.random.rand(22, 2)
pts0_b = pts0_b[np.linalg.norm(pts0_b - centre, axis=1) > 0.25][:18]
scatter_two_class(ax, pts1_b, pts0_b)

theta = np.linspace(0, 2*np.pi, 200)
rx, ry = 0.22, 0.22
ax.plot(0.5 + rx*np.cos(theta), 0.5 + ry*np.sin(theta),
        color=bound, linewidth=2.0, linestyle='--', zorder=4)
ax.fill(0.5 + rx*np.cos(theta), 0.5 + ry*np.sin(theta), alpha=0.08, color=c1)
style_ax(ax, '1 Hidden Layer\n(Convex Regions)')
ax.text(0.5, -0.14, 'Intersection of half-spaces\n= convex region',
        ha='center', color='#6b7280', fontsize=8.5, style='italic')

# ── Panel 3: 2+ hidden layers — non-convex / disconnected ────────────────────
ax = axes[2]

def blob(centre, n=14, r=0.12):
    a   = np.random.uniform(0, 2*np.pi, n)
    rad = np.random.uniform(0.02, r, n)
    return np.clip(centre + np.stack([rad*np.cos(a), rad*np.sin(a)], axis=1), 0.05, 0.95)

pts1_c = np.vstack([blob(np.array([0.25, 0.75])), blob(np.array([0.75, 0.25]))])
pts0_c = np.random.rand(30, 2)
mask   = ((np.linalg.norm(pts0_c - [0.25,0.75], axis=1) > 0.18) &
          (np.linalg.norm(pts0_c - [0.75,0.25], axis=1) > 0.18))
pts0_c = pts0_c[mask][:20]
scatter_two_class(ax, pts1_c, pts0_c)

for centre_b in [[0.25, 0.75], [0.75, 0.25]]:
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(centre_b[0] + 0.18*np.cos(theta), centre_b[1] + 0.16*np.sin(theta),
            color=bound, linewidth=2.0, linestyle='--', zorder=4)
    ax.fill(centre_b[0] + 0.18*np.cos(theta), centre_b[1] + 0.16*np.sin(theta),
            alpha=0.08, color=c1)
style_ax(ax, '2+ Hidden Layers\n(Arbitrary Regions)')
ax.text(0.5, -0.14, 'Union of convex regions\n= arbitrary / disconnected',
        ha='center', color='#6b7280', fontsize=8.5, style='italic')

# ── shared legend ─────────────────────────────────────────────────────────────
p1 = mpatches.Patch(color=c1, label='Class 1')
p0 = mpatches.Patch(color=c0, label='Class 0')
pb = mpatches.Patch(color=bound, label='Decision boundary', fill=False,
                    linestyle='--', linewidth=1.5)
fig.legend(handles=[p1, p0, pb], loc='lower center', ncol=3,
           facecolor='white', edgecolor='#d1d5db', labelcolor='#111111',
           fontsize=9.5, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Geometric Evolution of Decision Boundaries with Depth',
             color='#111111', fontsize=12, y=1.01)

plt.tight_layout(rect=[0, 0.08, 1, 1])
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'mlp_decision_boundaries.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {os.path.abspath(out)}')
