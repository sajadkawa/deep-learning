import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colour palette ────────────────────────────────────────────────────────────
C_INPUT  = '#dbeafe'   # blue-100
C_ENC    = '#fce7f3'   # pink-100
C_LATENT = '#fef3c7'   # amber-100
C_DEC    = '#fce7f3'   # pink-100
C_OUTPUT = '#dbeafe'   # blue-100

B_INPUT  = '#93c5fd'
B_ENC    = '#f9a8d4'
B_LATENT = '#fcd34d'
B_DEC    = '#f9a8d4'
B_OUTPUT = '#93c5fd'

TEXT_MAIN = '#111111'
TEXT_SUB  = '#374151'
ARROW     = '#9ca3af'

def neuron(ax, cx, cy, fill, border, label='', sublabel=''):
    circle = plt.Circle((cx, cy), 0.28, facecolor=fill, edgecolor=border,
                         linewidth=1.4, zorder=3)
    ax.add_patch(circle)
    if label:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=7.5, color=TEXT_MAIN, fontweight='bold', zorder=4)

def layer_neurons(ax, x, ys, fill, border, labels=None):
    for i, y in enumerate(ys):
        lbl = labels[i] if labels else ''
        neuron(ax, x, y, fill, border, lbl)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW,
                                lw=1.2, connectionstyle='arc3,rad=0.0'))

def box(ax, cx, cy, w, h, fill, border, label, sublabel=''):
    rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                   boxstyle='round,pad=0.05',
                                   facecolor=fill, edgecolor=border,
                                   linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    ax.text(cx, cy + (0.18 if sublabel else 0), label,
            ha='center', va='center', fontsize=9, color=TEXT_MAIN,
            fontweight='bold', zorder=3)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel, ha='center', va='center',
                fontsize=7.5, color=TEXT_SUB, zorder=3)

# ── Layer x-positions ─────────────────────────────────────────────────────────
X_IN   = 1.5
X_EH   = 3.5    # encoder hidden
X_LAT  = 6.5    # latent / bottleneck
X_DH   = 9.5    # decoder hidden
X_OUT  = 11.5

# ── Input layer (d=4 neurons for visual clarity) ──────────────────────────────
in_ys = [1.5, 2.2, 2.9, 3.6]
layer_neurons(ax, X_IN, in_ys, C_INPUT, B_INPUT)
ax.text(X_IN, 0.7, 'Input', ha='center', va='center',
        fontsize=9, color=TEXT_MAIN, fontweight='bold')
ax.text(X_IN, 0.35, r'$x \in \mathbb{R}^d$', ha='center', va='center',
        fontsize=8, color=TEXT_SUB)

# ── Encoder hidden layer (3 neurons) ─────────────────────────────────────────
eh_ys = [1.8, 2.65, 3.5]
layer_neurons(ax, X_EH, eh_ys, C_ENC, B_ENC)
ax.text(X_EH, 0.7, 'Encoder', ha='center', va='center',
        fontsize=9, color=TEXT_MAIN, fontweight='bold')
ax.text(X_EH, 0.35, r'$f_\phi$', ha='center', va='center',
        fontsize=8, color=TEXT_SUB)

# ── Latent layer (2 neurons — bottleneck) ─────────────────────────────────────
lat_ys = [2.1, 3.0]
layer_neurons(ax, X_LAT, lat_ys, C_LATENT, B_LATENT)
ax.text(X_LAT, 0.7, 'Latent code', ha='center', va='center',
        fontsize=9, color=TEXT_MAIN, fontweight='bold')
ax.text(X_LAT, 0.35, r'$z \in \mathbb{R}^k,\ k < d$', ha='center', va='center',
        fontsize=8, color=TEXT_SUB)

# ── Decoder hidden layer (3 neurons) ─────────────────────────────────────────
dh_ys = [1.8, 2.65, 3.5]
layer_neurons(ax, X_DH, dh_ys, C_DEC, B_DEC)
ax.text(X_DH, 0.7, 'Decoder', ha='center', va='center',
        fontsize=9, color=TEXT_MAIN, fontweight='bold')
ax.text(X_DH, 0.35, r'$g_\theta$', ha='center', va='center',
        fontsize=8, color=TEXT_SUB)

# ── Output layer (4 neurons) ──────────────────────────────────────────────────
out_ys = [1.5, 2.2, 2.9, 3.6]
layer_neurons(ax, X_OUT, out_ys, C_OUTPUT, B_OUTPUT)
ax.text(X_OUT, 0.7, 'Output', ha='center', va='center',
        fontsize=9, color=TEXT_MAIN, fontweight='bold')
ax.text(X_OUT, 0.35, r'$\hat{x} \in \mathbb{R}^d$', ha='center', va='center',
        fontsize=8, color=TEXT_SUB)

# ── Arrows between layers ─────────────────────────────────────────────────────
for y1 in in_ys:
    for y2 in eh_ys:
        arrow(ax, X_IN + 0.28, y1, X_EH - 0.28, y2)

for y1 in eh_ys:
    for y2 in lat_ys:
        arrow(ax, X_EH + 0.28, y1, X_LAT - 0.28, y2)

for y1 in lat_ys:
    for y2 in dh_ys:
        arrow(ax, X_LAT + 0.28, y1, X_DH - 0.28, y2)

for y1 in dh_ys:
    for y2 in out_ys:
        arrow(ax, X_DH + 0.28, y1, X_OUT - 0.28, y2)

# ── Bottleneck brace annotation ───────────────────────────────────────────────
ax.annotate('', xy=(X_LAT + 0.5, 4.55), xytext=(X_LAT - 0.5, 4.55),
            arrowprops=dict(arrowstyle='<->', color='#b45309', lw=1.5))
ax.text(X_LAT, 4.78, 'Bottleneck', ha='center', va='center',
        fontsize=8.5, color='#b45309', fontweight='bold')

# ── Encoder / Decoder span brackets ──────────────────────────────────────────
ax.annotate('', xy=(X_EH + 0.4, 4.3), xytext=(X_IN - 0.4, 4.3),
            arrowprops=dict(arrowstyle='<->', color='#9ca3af', lw=1.2))
ax.text((X_IN + X_EH) / 2, 4.5, 'Encoder', ha='center', va='center',
        fontsize=8, color=TEXT_SUB)

ax.annotate('', xy=(X_OUT + 0.4, 4.3), xytext=(X_DH - 0.4, 4.3),
            arrowprops=dict(arrowstyle='<->', color='#9ca3af', lw=1.2))
ax.text((X_DH + X_OUT) / 2, 4.5, 'Decoder', ha='center', va='center',
        fontsize=8, color=TEXT_SUB)

# ── Loss annotation ───────────────────────────────────────────────────────────
ax.annotate('', xy=(X_IN - 0.4, 1.0), xytext=(X_OUT + 0.4, 1.0),
            arrowprops=dict(arrowstyle='<->', color='#dc2626', lw=1.3,
                            connectionstyle='arc3,rad=-0.25'))
ax.text(6.5, 0.02, r'Reconstruction loss: $\mathcal{L}(x,\hat{x})$',
        ha='center', va='center', fontsize=8.5, color='#dc2626')

ax.set_title('Autoencoder Architecture', fontsize=12, color=TEXT_MAIN,
             fontweight='bold', pad=6)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'autoencoder_architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved autoencoder_architecture.png')
