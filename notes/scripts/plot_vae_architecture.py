import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

C_INPUT  = '#dbeafe'
C_ENC    = '#fce7f3'
C_LAT    = '#fef3c7'
C_DEC    = '#fce7f3'
C_OUT    = '#dbeafe'
C_SAMPLE = '#dcfce7'

B_INPUT  = '#93c5fd'
B_ENC    = '#f9a8d4'
B_LAT    = '#fcd34d'
B_DEC    = '#f9a8d4'
B_OUT    = '#93c5fd'
B_SAMPLE = '#86efac'

TEXT_MAIN = '#111111'
TEXT_SUB  = '#374151'
ARROW     = '#9ca3af'
RED       = '#dc2626'
AMBER     = '#b45309'
GREEN     = '#15803d'


def draw_ae(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    def neuron(cx, cy, fill, border):
        ax.add_patch(plt.Circle((cx, cy), 0.26, facecolor=fill,
                                edgecolor=border, linewidth=1.4, zorder=3))

    def arr(x1, y1, x2, y2, color=ARROW):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # Input
    in_ys = [1.6, 2.4, 3.2]
    for y in in_ys:
        neuron(1.2, y, C_INPUT, B_INPUT)
    ax.text(1.2, 0.8, 'Input', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')
    ax.text(1.2, 0.45, r'$x$', ha='center', fontsize=8, color=TEXT_SUB)

    # Encoder
    eh_ys = [1.9, 2.8]
    for y in eh_ys:
        neuron(3.2, y, C_ENC, B_ENC)
    ax.text(3.2, 0.8, 'Encoder', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')

    # Latent — single point
    neuron(5.0, 2.4, C_LAT, B_LAT)
    ax.text(5.0, 0.8, 'Latent', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')
    ax.text(5.0, 0.45, r'$z$ (point)', ha='center', fontsize=8, color=TEXT_SUB)

    # Decoder
    dh_ys = [1.9, 2.8]
    for y in dh_ys:
        neuron(6.8, y, C_DEC, B_DEC)
    ax.text(6.8, 0.8, 'Decoder', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')

    # Output
    out_ys = [1.6, 2.4, 3.2]
    for y in out_ys:
        neuron(8.8, y, C_OUT, B_OUT)
    ax.text(8.8, 0.8, 'Output', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')
    ax.text(8.8, 0.45, r'$\hat{x}$', ha='center', fontsize=8, color=TEXT_SUB)

    # Arrows
    for y1 in in_ys:
        for y2 in eh_ys:
            arr(1.2+0.26, y1, 3.2-0.26, y2)
    for y1 in eh_ys:
        arr(3.2+0.26, y1, 5.0-0.26, 2.4)
    for y2 in dh_ys:
        arr(5.0+0.26, 2.4, 6.8-0.26, y2)
    for y1 in dh_ys:
        for y2 in out_ys:
            arr(6.8+0.26, y1, 8.8-0.26, y2)

    # Loss arrow
    ax.annotate('', xy=(1.2-0.3, 1.2), xytext=(8.8+0.3, 1.2),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.3,
                                connectionstyle='arc3,rad=-0.2'))
    ax.text(5.0, 0.05, r'$\mathcal{L}(x, \hat{x})$', ha='center',
            fontsize=8, color=RED)

    ax.set_title('Standard Autoencoder', fontsize=11,
                 color=TEXT_MAIN, fontweight='bold', pad=8)


def draw_vae(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def neuron(cx, cy, fill, border):
        ax.add_patch(plt.Circle((cx, cy), 0.26, facecolor=fill,
                                edgecolor=border, linewidth=1.4, zorder=3))

    def arr(x1, y1, x2, y2, color=ARROW):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    def box(cx, cy, w, h, fill, border, label, sub=''):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-w/2, cy-h/2), w, h,
            boxstyle='round,pad=0.05',
            facecolor=fill, edgecolor=border, linewidth=1.4, zorder=2))
        ax.text(cx, cy + (0.15 if sub else 0), label, ha='center', va='center',
                fontsize=8.5, color=TEXT_MAIN, fontweight='bold', zorder=3)
        if sub:
            ax.text(cx, cy - 0.22, sub, ha='center', va='center',
                    fontsize=7.5, color=TEXT_SUB, zorder=3)

    # Input
    in_ys = [1.8, 2.6, 3.4]
    for y in in_ys:
        neuron(1.2, y, C_INPUT, B_INPUT)
    ax.text(1.2, 1.0, 'Input', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')
    ax.text(1.2, 0.65, r'$x$', ha='center', fontsize=8, color=TEXT_SUB)

    # Encoder
    eh_ys = [2.1, 3.0]
    for y in eh_ys:
        neuron(3.0, y, C_ENC, B_ENC)
    ax.text(3.0, 1.0, 'Encoder', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')

    # mu and log sigma^2 boxes
    box(5.0, 3.5, 1.3, 0.55, C_LAT, B_LAT, r'$\mu$')
    box(5.0, 2.0, 1.3, 0.55, C_LAT, B_LAT, r'$\log\sigma^2$')
    ax.text(5.0, 1.0, 'Latent params', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')

    # Epsilon
    ax.text(6.6, 2.75, r'$\varepsilon \sim \mathcal{N}(0,I)$',
            ha='center', fontsize=8, color=GREEN, style='italic')

    # Sample box
    box(7.8, 2.75, 1.2, 0.55, C_SAMPLE, B_SAMPLE, r'$z = \mu + \sigma\odot\varepsilon$', '')
    ax.text(7.8, 1.0, 'Sample', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')

    # Decoder
    dh_ys = [2.1, 3.0]  # shifted up slightly
    for y in [2.3, 3.2]:
        neuron(9.8, y, C_DEC, B_DEC)
    ax.text(9.8, 1.0, 'Decoder', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')

    # Output
    out_ys = [1.8, 2.6, 3.4]
    for y in out_ys:
        neuron(11.5, y, C_OUT, B_OUT)
    ax.text(11.5, 1.0, 'Output', ha='center', fontsize=8.5,
            color=TEXT_MAIN, fontweight='bold')
    ax.text(11.5, 0.65, r'$\hat{x}$', ha='center', fontsize=8, color=TEXT_SUB)

    # Arrows: input → encoder
    for y1 in in_ys:
        for y2 in eh_ys:
            arr(1.2+0.26, y1, 3.0-0.26, y2)

    # encoder → mu and log sigma^2
    for y2 in [3.5, 2.0]:
        for y1 in eh_ys:
            arr(3.0+0.26, y1, 5.0-0.65, y2)

    # mu, log sigma^2 → sample
    arr(5.0+0.65, 3.5, 7.8-0.6, 2.85)
    arr(5.0+0.65, 2.0, 7.8-0.6, 2.65)

    # epsilon → sample
    arr(6.6, 2.95, 7.8-0.6, 2.78, color=GREEN)

    # sample → decoder
    for y2 in [2.3, 3.2]:
        arr(7.8+0.6, 2.75, 9.8-0.26, y2)

    # decoder → output
    for y1 in [2.3, 3.2]:
        for y2 in out_ys:
            arr(9.8+0.26, y1, 11.5-0.26, y2)

    # KL annotation
    ax.annotate('', xy=(5.0+0.65, 3.5), xytext=(5.0+0.65, 2.0),
                arrowprops=dict(arrowstyle='<->', color=AMBER, lw=1.2))
    ax.text(5.9, 2.75, r'$D_{\mathrm{KL}}$', ha='left', fontsize=8,
            color=AMBER, fontweight='bold')

    # Recon loss arrow
    ax.annotate('', xy=(1.2-0.3, 1.35), xytext=(11.5+0.3, 1.35),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.3,
                                connectionstyle='arc3,rad=-0.2'))
    ax.text(6.35, 0.3, r'$\mathcal{L} = \mathcal{L}_{\mathrm{recon}} + D_{\mathrm{KL}}$',
            ha='center', fontsize=8, color=RED)

    ax.set_title('Variational Autoencoder (VAE)', fontsize=11,
                 color=TEXT_MAIN, fontweight='bold', pad=8)


draw_ae(axes[0])
draw_vae(axes[1])

plt.tight_layout(pad=1.5)
out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'vae_architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved vae_architecture.png')
