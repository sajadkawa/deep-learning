import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import os

fig = plt.figure(figsize=(14, 5.5))
fig.patch.set_facecolor('#1e1e1e')

w1 = np.linspace(-4, 4, 500)
w2 = np.linspace(-4, 4, 500)
W1, W2 = np.meshgrid(w1, w2)

# natural rugged surface: wave superposition + gentle global bowl
np.random.seed(42)
L = (
    0.08 * (W1**2 + W2**2)                          # global bowl — ensures one global min near centre
    + 0.9  * np.sin(1.3 * W1) * np.cos(1.1 * W2)
    + 0.7  * np.cos(0.9 * W1 + 0.5) * np.sin(1.4 * W2 + 0.3)
    + 0.5  * np.sin(1.8 * W1 - 0.7) * np.sin(1.6 * W2 + 1.1)
    + 0.35 * np.cos(2.2 * W1 + 1.2) * np.cos(2.0 * W2 - 0.8)
    + 0.20 * np.sin(2.8 * W1 - 1.5) * np.cos(2.5 * W2 + 0.6)
)

# shift so minimum = 0
L = L - L.min()

min_idx = np.unravel_index(np.argmin(L), L.shape)
gmin_w1 = float(W1[min_idx])
gmin_w2 = float(W2[min_idx])
gmin_L  = float(L[min_idx])

# find local minima with pure numpy — sliding window minimum
pad = 20
L_pad = np.pad(L, pad, mode='edge')
local_min_mask = np.ones_like(L, dtype=bool)
for di in range(-pad, pad+1):
    for dj in range(-pad, pad+1):
        if di == 0 and dj == 0:
            continue
        shifted = L_pad[pad+di:pad+di+L.shape[0], pad+dj:pad+dj+L.shape[1]]
        local_min_mask &= (L <= shifted)
local_min_mask &= (L > 0.05)
# exclude region around global min
local_min_mask[min_idx[0]-15:min_idx[0]+15,
               min_idx[1]-15:min_idx[1]+15] = False
lm_indices = np.argwhere(local_min_mask)
# pick 3 well-separated ones
chosen = []
for idx in lm_indices:
    lw1, lw2 = float(W1[idx[0], idx[1]]), float(W2[idx[0], idx[1]])
    if all(np.sqrt((lw1 - c[0])**2 + (lw2 - c[1])**2) > 1.8 for c in chosen):
        chosen.append((lw1, lw2, float(L[idx[0], idx[1]])))
    if len(chosen) == 3:
        break

# ── 3D surface ───────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_facecolor('#111111')

ls  = LightSource(azdeg=260, altdeg=50)
rgb = ls.shade(L, cmap=cm.plasma, vert_exag=0.8, blend_mode='soft')
ax1.plot_surface(W1, W2, L, facecolors=rgb,
                 linewidth=0, antialiased=True,
                 rcount=250, ccount=250, shade=False)

ax1.scatter([gmin_w1], [gmin_w2], [gmin_L + 0.05],
            color='#00e5ff', s=70, zorder=10, depthshade=False)
ax1.text(gmin_w1 + 0.25, gmin_w2, gmin_L + 0.35,
         'global min', color='#00e5ff', fontsize=8, zorder=10)

for lw1, lw2, lL in chosen:
    ax1.scatter([lw1], [lw2], [lL + 0.05],
                color='#ffb74d', s=35, zorder=9, depthshade=False)

ax1.set_xlabel('w₁', color='#888', fontsize=9, labelpad=3)
ax1.set_ylabel('w₂', color='#888', fontsize=9, labelpad=3)
ax1.set_zlabel('Loss L', color='#888', fontsize=9, labelpad=3)
ax1.set_title('Loss Surface', color='white', fontsize=11, pad=8)
ax1.tick_params(colors='#555', labelsize=7)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('#222')
ax1.yaxis.pane.set_edgecolor('#222')
ax1.zaxis.pane.set_edgecolor('#222')
ax1.grid(False)
ax1.view_init(elev=35, azim=-55)

# ── Contour view ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor('#1a1a1a')

ax2.contourf(W1, W2, L, levels=60, cmap='plasma', alpha=0.95)
ax2.contour(W1, W2, L, levels=30, colors='white', alpha=0.12, linewidths=0.35)

# global minimum
ax2.plot(gmin_w1, gmin_w2, '*', color='#00e5ff', markersize=13,
         zorder=6, label='global minimum')
ax2.annotate('global minimum',
             xy=(gmin_w1, gmin_w2),
             xytext=(gmin_w1 + 0.7, gmin_w2 + 1.1),
             color='#00e5ff', fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color='#00e5ff', lw=1.2))

# local minima
first = True
for lw1, lw2, _ in chosen:
    ax2.plot(lw1, lw2, 'o', color='#ffb74d', markersize=8,
             zorder=5, label='local minimum' if first else '')
    first = False

if chosen:
    lw1, lw2, _ = chosen[0]
    ax2.annotate('local minimum',
                 xy=(lw1, lw2),
                 xytext=(lw1 + 0.8, lw2 - 0.8),
                 color='#ffb74d', fontsize=8.5,
                 arrowprops=dict(arrowstyle='->', color='#ffb74d', lw=1.2))

# gradient descent path — start far from global min
start_w1, start_w2 = -3.5, 3.2
path_w1, path_w2 = [start_w1], [start_w2]
lr = 0.14
np.random.seed(3)
for _ in range(50):
    gw1 = gmin_w1 - path_w1[-1]
    gw2 = gmin_w2 - path_w2[-1]
    path_w1.append(path_w1[-1] + lr * gw1 + np.random.normal(0, 0.04))
    path_w2.append(path_w2[-1] + lr * gw2 + np.random.normal(0, 0.04))

ax2.plot(path_w1, path_w2, color='#69f0ae', linewidth=1.8,
         zorder=4, label='gradient descent')
ax2.plot(path_w1[0], path_w2[0], 'o', color='#69f0ae', markersize=7, zorder=5)
ax2.text(path_w1[0] + 0.1, path_w2[0] - 0.35, 'start', color='#69f0ae', fontsize=8.5)

ax2.set_xlabel('w₁', color='#aaa', fontsize=9)
ax2.set_ylabel('w₂', color='#aaa', fontsize=9)
ax2.set_title('Contour View — Local vs Global Minimum', color='white', fontsize=11)
ax2.tick_params(colors='#aaa', labelsize=8)
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')
ax2.legend(facecolor='#2a2a2a', labelcolor='white', fontsize=8.5,
           loc='upper right', framealpha=0.9)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)

fig.suptitle('Loss Landscape — Global Minimum, Local Minima, and Gradient Descent',
             color='white', fontsize=12, y=1.01)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'loss_landscape.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved loss_landscape.png")
