import numpy as np
import matplotlib.pyplot as plt
import os

bg      = '#0f1117'
panel   = '#161b27'
spine_c = '#3a3f50'
txt     = '#d0d4e0'
c_s1    = '#4a9eda'   # first sigmoid
c_s2    = '#e05252'   # second sigmoid (flipped)
c_bump  = '#f0a500'   # bump = difference
c_sum   = '#2ecc71'   # sum of bumps

def sigmoid(z): return 1 / (1 + np.exp(-z))

x = np.linspace(-1, 5, 500)

# Two sigmoids: one rising at x=1, one rising at x=3 (flipped → falling)
w, b1, b2 = 8, -8, -24   # steep sigmoids
s1 =  sigmoid(w*x + b1)          # rises at x=1
s2 = -sigmoid(w*x + b2) + 1      # falls at x=3  (= 1 - sigmoid)
bump = s1 - (1 - s2)              # = s1 + s2 - 1  → bump between 1 and 3

# Three bumps tiling a target function
def make_bump(centre, width=1.2, height=1.0):
    w_ = 8
    s_left  =  sigmoid(w_*(x - (centre - width/2)))
    s_right =  sigmoid(w_*(x - (centre + width/2)))
    return height * (s_left - s_right)

bump1 = make_bump(0.5,  height=0.6)
bump2 = make_bump(2.0,  height=1.0)
bump3 = make_bump(3.5,  height=0.4)
approx = bump1 + bump2 + bump3

# Fake target function
target = 0.55*np.exp(-0.5*((x-2.0)/1.1)**2) + 0.35*np.exp(-0.5*((x-0.4)/0.4)**2)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor(bg)
for ax in axes:
    ax.set_facecolor(panel)
    for sp in ax.spines.values():
        sp.set_edgecolor(spine_c)
    ax.tick_params(colors='#888', labelsize=9)

# ── Left: how two sigmoids make one bump ─────────────────────────────────────
ax = axes[0]
ax.plot(x, s1,   color=c_s1,  linewidth=1.8, linestyle='--', label='σ₁  (rising at x=1)')
ax.plot(x, 1-s2, color=c_s2,  linewidth=1.8, linestyle='--', label='σ₂  (rising at x=3)')
ax.plot(x, bump, color=c_bump, linewidth=2.5, label='σ₁ − σ₂  (bump)')
ax.fill_between(x, 0, bump, alpha=0.12, color=c_bump)
ax.axhline(0, color='#555', linewidth=0.8)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.15, 1.25)
ax.set_xlabel('x', color=txt, fontsize=11)
ax.set_ylabel('activation', color=txt, fontsize=11)
ax.set_title('Two sigmoid neurons → one bump function', color='white', fontsize=10.5, pad=10)
ax.legend(fontsize=8.5, facecolor=panel, edgecolor=spine_c, labelcolor=txt, loc='upper right')
ax.text(1.0, 0.52, 'bump\nregion', ha='center', color=c_bump, fontsize=8, style='italic')

# ── Right: bumps tile an arbitrary function ───────────────────────────────────
ax = axes[1]
ax.plot(x, target, color='white', linewidth=2.2, label='target function f(x)', zorder=5)
ax.plot(x, bump1,  color='#7ec8e3', linewidth=1.4, linestyle=':', alpha=0.8, label='bump 1')
ax.plot(x, bump2,  color='#7ec8e3', linewidth=1.4, linestyle=':', alpha=0.8, label='bump 2')
ax.plot(x, bump3,  color='#7ec8e3', linewidth=1.4, linestyle=':', alpha=0.8, label='bump 3')
ax.plot(x, approx, color=c_sum, linewidth=2.2, linestyle='--', label='sum of bumps ≈ f(x)', zorder=4)
ax.fill_between(x, 0, bump1, alpha=0.07, color='#7ec8e3')
ax.fill_between(x, 0, bump2, alpha=0.07, color='#7ec8e3')
ax.fill_between(x, 0, bump3, alpha=0.07, color='#7ec8e3')
ax.axhline(0, color='#555', linewidth=0.8)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.15, 1.25)
ax.set_xlabel('x', color=txt, fontsize=11)
ax.set_ylabel('output', color=txt, fontsize=11)
ax.set_title('Enough bumps approximate any continuous function', color='white', fontsize=10.5, pad=10)
ax.legend(fontsize=8.5, facecolor=panel, edgecolor=spine_c, labelcolor=txt, loc='upper right')

fig.suptitle('Universal Approximation — the Bump Function Intuition',
             color='white', fontsize=12, y=1.01)
plt.tight_layout(rect=[0, 0.02, 1, 1])

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'uat_bump_functions.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'Saved: {os.path.abspath(out)}')
