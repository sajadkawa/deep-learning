import numpy as np
import matplotlib.pyplot as plt
import os

c_s1   = '#2563eb'   # first sigmoid — blue
c_s2   = '#dc2626'   # second sigmoid — red
c_bump = '#ca8a04'   # bump — amber
c_sum  = '#15803d'   # sum of bumps — green
c_ind  = '#93c5fd'   # individual bumps — light blue

def sigmoid(z): return 1 / (1 + np.exp(-z))

x = np.linspace(-1, 5, 500)

w, b1, b2 = 8, -8, -24
s1   =  sigmoid(w*x + b1)
s2   = -sigmoid(w*x + b2) + 1
bump =  s1 - (1 - s2)

def make_bump(centre, width=1.2, height=1.0):
    w_ = 8
    s_left  = sigmoid(w_*(x - (centre - width/2)))
    s_right = sigmoid(w_*(x - (centre + width/2)))
    return height * (s_left - s_right)

bump1  = make_bump(0.5,  height=0.6)
bump2  = make_bump(2.0,  height=1.0)
bump3  = make_bump(3.5,  height=0.4)
approx = bump1 + bump2 + bump3

target = 0.55*np.exp(-0.5*((x-2.0)/1.1)**2) + 0.35*np.exp(-0.5*((x-0.4)/0.4)**2)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#d1d5db')
    ax.tick_params(colors='#374151', labelsize=9)

# ── Left: how two sigmoids make one bump ─────────────────────────────────────
ax = axes[0]
ax.plot(x, s1,   color=c_s1,  linewidth=1.8, linestyle='--', label='σ₁  (rising at x=1)')
ax.plot(x, 1-s2, color=c_s2,  linewidth=1.8, linestyle='--', label='σ₂  (rising at x=3)')
ax.plot(x, bump, color=c_bump, linewidth=2.5, label='σ₁ − σ₂  (bump)')
ax.fill_between(x, 0, bump, alpha=0.12, color=c_bump)
ax.axhline(0, color='#e5e7eb', linewidth=0.8)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.15, 1.25)
ax.set_xlabel('x', color='#374151', fontsize=11)
ax.set_ylabel('activation', color='#374151', fontsize=11)
ax.set_title('Two sigmoid neurons → one bump function', color='#111111', fontsize=10.5, pad=10)
ax.legend(fontsize=8.5, facecolor='white', edgecolor='#d1d5db', labelcolor='#111111', loc='upper right')
ax.text(1.0, 0.52, 'bump\nregion', ha='center', color=c_bump, fontsize=8, style='italic')

# ── Right: bumps tile an arbitrary function ───────────────────────────────────
ax = axes[1]
ax.plot(x, target, color='#111111', linewidth=2.2, label='target function f(x)', zorder=5)
ax.plot(x, bump1,  color=c_ind, linewidth=1.4, linestyle=':', alpha=0.9, label='bump 1')
ax.plot(x, bump2,  color=c_ind, linewidth=1.4, linestyle=':', alpha=0.9, label='bump 2')
ax.plot(x, bump3,  color=c_ind, linewidth=1.4, linestyle=':', alpha=0.9, label='bump 3')
ax.plot(x, approx, color=c_sum, linewidth=2.2, linestyle='--', label='sum of bumps ≈ f(x)', zorder=4)
ax.fill_between(x, 0, bump1, alpha=0.08, color=c_ind)
ax.fill_between(x, 0, bump2, alpha=0.08, color=c_ind)
ax.fill_between(x, 0, bump3, alpha=0.08, color=c_ind)
ax.axhline(0, color='#e5e7eb', linewidth=0.8)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.15, 1.25)
ax.set_xlabel('x', color='#374151', fontsize=11)
ax.set_ylabel('output', color='#374151', fontsize=11)
ax.set_title('Enough bumps approximate any continuous function', color='#111111', fontsize=10.5, pad=10)
ax.legend(fontsize=8.5, facecolor='white', edgecolor='#d1d5db', labelcolor='#111111', loc='upper right')

fig.suptitle('Universal Approximation — the Bump Function Intuition',
             color='#111111', fontsize=12, y=1.01)
plt.tight_layout(rect=[0, 0.02, 1, 1])

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'uat_bump_functions.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {os.path.abspath(out)}')
