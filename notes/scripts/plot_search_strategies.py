import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(0)

# 3x3 grid search — 9 evaluations
grid_vals = [0.1, 0.5, 0.9]
grid_points = [(x, y) for x in grid_vals for y in grid_vals]

# 9 random search evaluations
random_points = list(zip(rng.uniform(0, 1, 9), rng.uniform(0, 1, 9)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.patch.set_facecolor('#0d1117')

# Important hyperparameter is x-axis (hp1), unimportant is y-axis (hp2)
# Show this with a gradient band along x
for ax, points, title in [
    (axes[0], grid_points,   'Grid Search  (9 evaluations)'),
    (axes[1], random_points, 'Random Search  (9 evaluations)'),
]:
    ax.set_facecolor('#0d1117')

    # Background gradient — shows hp1 (x) matters, hp2 (y) does not
    x_bg = np.linspace(0, 1, 200)
    y_bg = np.linspace(0, 1, 200)
    X_bg, Y_bg = np.meshgrid(x_bg, y_bg)
    Z = X_bg   # loss depends only on x
    ax.contourf(X_bg, Y_bg, Z, levels=20, cmap='RdYlGn_r', alpha=0.25)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, color='#e6edf3', s=60, zorder=3, edgecolors='#388bfd', linewidths=1.2)

    # Mark unique x values explored
    unique_x = len(set(round(x, 2) for x in xs))
    ax.set_xlabel('Hyperparameter 1  (important)', color='#8b949e', fontsize=9)
    ax.set_ylabel('Hyperparameter 2  (unimportant)', color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_title(title, color='#e6edf3', fontsize=10, pad=8)
    ax.text(0.5, -0.18, f'{unique_x} distinct values of HP1 explored',
            ha='center', transform=ax.transAxes, color='#8b949e', fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

fig.suptitle('Grid Search vs Random Search — Coverage of Important Hyperparameter',
             color='#e6edf3', fontsize=11, y=1.02)
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'search_strategies.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
