import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis('off')
fig.patch.set_facecolor('white')

TEXT_MAIN = '#111111'
TEXT_SUB  = '#374151'
ARROW     = '#9ca3af'
RED       = '#dc2626'
GREEN     = '#15803d'
AMBER     = '#b45309'

C_NOISE  = '#f3e8ff';  B_NOISE  = '#c084fc'
C_GEN    = '#fce7f3';  B_GEN    = '#f9a8d4'
C_FAKE   = '#fef3c7';  B_FAKE   = '#fcd34d'
C_REAL   = '#dbeafe';  B_REAL   = '#93c5fd'
C_DISC   = '#dcfce7';  B_DISC   = '#86efac'
C_OUT    = '#fef9c3';  B_OUT    = '#ca8a04'

ROW_Y  = 2.8   # main horizontal row y
REAL_Y = 4.5   # real data box y
BOX_H  = 0.75
BOX_W  = 1.55

# x-centres of boxes on the main row
X_NOISE = 1.2
X_GEN   = 3.2
X_FAKE  = 5.4
X_DISC  = 8.6
X_OUT   = 11.5

# Real data sits above the discriminator
X_REAL  = X_DISC


def box(cx, cy, w, h, fill, border, label, sub=''):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.07',
        facecolor=fill, edgecolor=border, linewidth=1.5, zorder=2))
    ax.text(cx, cy + (0.16 if sub else 0), label,
            ha='center', va='center', fontsize=9.5,
            color=TEXT_MAIN, fontweight='bold', zorder=3)
    if sub:
        ax.text(cx, cy - 0.22, sub, ha='center', va='center',
                fontsize=8, color=TEXT_SUB, zorder=3)


def harrow(x1, x2, y, color=ARROW):
    """Horizontal arrow from right edge of box at x1 to left edge of box at x2."""
    ax.annotate('', xy=(x2 - BOX_W/2, y), xytext=(x1 + BOX_W/2, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


# ── Draw boxes ────────────────────────────────────────────────────────────────
box(X_NOISE, ROW_Y, BOX_W, BOX_H, C_NOISE, B_NOISE, 'Noise',       r'$z \sim p(z)$')
box(X_GEN,   ROW_Y, BOX_W, BOX_H, C_GEN,   B_GEN,   'Generator',   r'$G_\theta$')
box(X_FAKE,  ROW_Y, BOX_W, BOX_H, C_FAKE,  B_FAKE,  'Fake sample', r'$\tilde{x}=G_\theta(z)$')
box(X_REAL,  REAL_Y, BOX_W, BOX_H, C_REAL, B_REAL,  'Real data',   r'$x \sim p_{\mathrm{data}}$')
box(X_DISC,  ROW_Y, BOX_W, BOX_H, C_DISC,  B_DISC,  'Discriminator', r'$D_\phi$')
box(X_OUT,   ROW_Y, BOX_W, BOX_H, C_OUT,   B_OUT,   'Real or Fake?', r'$D_\phi(\cdot)\in[0,1]$')

# ── Main row arrows ───────────────────────────────────────────────────────────
harrow(X_NOISE, X_GEN,  ROW_Y)   # noise → generator
harrow(X_GEN,   X_FAKE, ROW_Y)   # generator → fake sample

# fake sample → discriminator: horizontal gap between X_FAKE and X_DISC
# insert a merge point between them
MID_X = (X_FAKE + BOX_W/2 + X_DISC - BOX_W/2) / 2   # midpoint of the gap

# arrow from fake to merge point
ax.annotate('', xy=(MID_X, ROW_Y), xytext=(X_FAKE + BOX_W/2, ROW_Y),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.5))
# arrow from merge point to discriminator
ax.annotate('', xy=(X_DISC - BOX_W/2, ROW_Y), xytext=(MID_X, ROW_Y),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.5))

# label on fake→disc arrow
ax.text(MID_X, ROW_Y + 0.28, 'fake (label 0)',
        ha='center', va='bottom', fontsize=7.5, color='#b91c1c')

harrow(X_DISC, X_OUT, ROW_Y)     # discriminator → output

# ── Real data → discriminator: vertical arrow down ────────────────────────────
ax.annotate('', xy=(X_REAL, ROW_Y + BOX_H/2),
            xytext=(X_REAL, REAL_Y - BOX_H/2),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.5))
ax.text(X_REAL + 0.55, (ROW_Y + BOX_H/2 + REAL_Y - BOX_H/2) / 2,
        'real (label 1)', ha='left', va='center', fontsize=7.5, color='#2563eb')

# ── Gradient arrow: below the row, right to left ──────────────────────────────
GRAD_Y = 1.55
ax.annotate('', xy=(X_GEN + BOX_W/2, GRAD_Y),
            xytext=(X_DISC - BOX_W/2, GRAD_Y),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
ax.text((X_GEN + X_DISC) / 2, GRAD_Y - 0.28,
        r'gradient $\nabla_\theta \mathcal{L}_G$ flows back to $G_\theta$',
        ha='center', va='top', fontsize=8, color=RED)

# vertical stubs connecting boxes to gradient line
for xc in [X_GEN + BOX_W/2, X_DISC - BOX_W/2]:
    ax.plot([xc, xc], [ROW_Y - BOX_H/2, GRAD_Y], color=RED, lw=1.0, ls='--', zorder=1)

# ── Loss equations ────────────────────────────────────────────────────────────
ax.text(7.0, 0.85,
        r'$\mathcal{L}_D = -[\log D_\phi(x) + \log(1 - D_\phi(G_\theta(z)))]$',
        ha='center', va='center', fontsize=8.5, color=AMBER)
ax.text(7.0, 0.38,
        r'$\mathcal{L}_G = -\log D_\phi(G_\theta(z))$',
        ha='center', va='center', fontsize=8.5, color=GREEN)

ax.set_title('Generative Adversarial Network (GAN)', fontsize=12,
             color=TEXT_MAIN, fontweight='bold', pad=6)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'gan_architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved gan_architecture.png')
