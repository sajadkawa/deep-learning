import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
fig.patch.set_facecolor('white')

TEXT_MAIN = '#111111'
TEXT_SUB  = '#374151'
TEXT_DIM  = '#6b7280'
ARROW     = '#9ca3af'
SKIP_COL  = '#b45309'

# Colour scheme
C_ENC = '#dbeafe'; B_ENC = '#93c5fd'   # blue  — encoder conv blocks
C_POOL= '#fce7f3'; B_POOL= '#f9a8d4'   # pink  — pooling
C_BOT = '#fef3c7'; B_BOT = '#fcd34d'   # amber — bottleneck
C_UP  = '#dcfce7'; B_UP  = '#86efac'   # green — upsample
C_DEC = '#dbeafe'; B_DEC = '#93c5fd'   # blue  — decoder conv blocks
C_OUT = '#f3e8ff'; B_OUT = '#c084fc'   # purple— output


def block(cx, cy, w, h, fill, border, label, sub=''):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.05',
        facecolor=fill, edgecolor=border, linewidth=1.4, zorder=3))
    ax.text(cx, cy + (0.13 if sub else 0), label,
            ha='center', va='center', fontsize=7.5,
            color=TEXT_MAIN, fontweight='bold', zorder=4)
    if sub:
        ax.text(cx, cy - 0.18, sub, ha='center', va='center',
                fontsize=6.5, color=TEXT_SUB, zorder=4)


def arrow_down(cx, y1, y2):
    ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.3))


def arrow_up(cx, y1, y2):
    ax.annotate('', xy=(cx, y2), xytext=(cx, y1),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.3))


def arrow_right(x1, x2, cy):
    ax.annotate('', xy=(x2, cy), xytext=(x1, cy),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.3))


def skip_arrow(x1, x2, cy):
    """Dashed skip connection arrow from encoder to decoder."""
    ax.annotate('', xy=(x2, cy), xytext=(x1, cy),
                arrowprops=dict(arrowstyle='->', color=SKIP_COL, lw=1.3,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'))


# ── Layout: 5 encoder levels + bottleneck + 4 decoder levels ─────────────────
# Encoder x-positions (left side)
EX = [1.2, 2.4, 3.6, 4.8]   # 4 encoder levels
BX = 7.0                      # bottleneck x
# Decoder x-positions (right side, mirrored)
DX = [9.2, 10.4, 11.6, 12.8]

# y-positions per level (encoder goes down, decoder goes up)
EY = [6.0, 4.8, 3.6, 2.4]   # encoder levels top→bottom
BY = 1.2                      # bottleneck y
DY = [2.4, 3.6, 4.8, 6.0]   # decoder levels bottom→top

BW = 1.0   # block width
BH = 0.55  # block height

# ── Encoder blocks ────────────────────────────────────────────────────────────
enc_labels = ['64ch\n3×3 Conv×2', '128ch\n3×3 Conv×2',
              '256ch\n3×3 Conv×2', '512ch\n3×3 Conv×2']
for i, (ex, ey, lbl) in enumerate(zip(EX, EY, enc_labels)):
    block(ex, ey, BW, BH, C_ENC, B_ENC, f'Enc {i+1}', lbl)

# ── Pooling arrows (encoder down) ────────────────────────────────────────────
for i in range(len(EX) - 1):
    # arrow from enc block down to next level (via pool)
    mid_x = (EX[i] + EX[i+1]) / 2
    mid_y = (EY[i] + EY[i+1]) / 2
    # draw pool label between levels
    ax.text(EX[i] + 0.05, EY[i] - 0.45, '↓ MaxPool\n2×2', ha='center',
            va='center', fontsize=6, color=TEXT_DIM)

# arrows connecting encoder levels diagonally down-right
for i in range(len(EX) - 1):
    ax.annotate('', xy=(EX[i+1] - BW/2, EY[i+1] + BH/2),
                xytext=(EX[i] + BW/2, EY[i] - BH/2),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.2))

# ── Encoder to bottleneck ─────────────────────────────────────────────────────
ax.annotate('', xy=(BX - BW/2, BY + BH/2),
            xytext=(EX[-1] + BW/2, EY[-1] - BH/2),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.2))
ax.text(EX[-1] + 0.05, EY[-1] - 0.45, '↓ MaxPool\n2×2', ha='center',
        va='center', fontsize=6, color=TEXT_DIM)

# ── Bottleneck ────────────────────────────────────────────────────────────────
block(BX, BY, BW + 0.2, BH + 0.1, C_BOT, B_BOT, 'Bottleneck', '1024ch\n3×3 Conv×2')

# ── Decoder blocks ────────────────────────────────────────────────────────────
dec_labels = ['512ch\n3×3 Conv×2', '256ch\n3×3 Conv×2',
              '128ch\n3×3 Conv×2', '64ch\n3×3 Conv×2']
for i, (dx, dy, lbl) in enumerate(zip(DX, DY, dec_labels)):
    block(dx, dy, BW, BH, C_DEC, B_DEC, f'Dec {i+1}', lbl)

# ── Bottleneck to first decoder ───────────────────────────────────────────────
ax.annotate('', xy=(DX[0] - BW/2, DY[0] - BH/2),
            xytext=(BX + BW/2 + 0.1, BY + BH/2),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.2))
ax.text(BX + 0.9, BY + 0.55, '↑ Upsample', ha='center',
        va='center', fontsize=6, color=TEXT_DIM)

# ── Decoder upward arrows ─────────────────────────────────────────────────────
for i in range(len(DX) - 1):
    ax.annotate('', xy=(DX[i+1] - BW/2, DY[i+1] - BH/2),
                xytext=(DX[i] + BW/2, DY[i] + BH/2),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.2))
    ax.text(DX[i] + 0.05, DY[i] + 0.45, '↑ Upsample', ha='center',
            va='center', fontsize=6, color=TEXT_DIM)

# ── Skip connections (encoder → decoder, same level) ─────────────────────────
for i in range(4):
    ex, ey = EX[i], EY[i]
    dx, dy = DX[3 - i], DY[3 - i]   # mirrored: enc level 0 ↔ dec level 3
    # draw dashed horizontal arrow at the level y
    ax.annotate('', xy=(dx - BW/2, dy),
                xytext=(ex + BW/2, ey),
                arrowprops=dict(arrowstyle='->', color=SKIP_COL, lw=1.2,
                                linestyle='dashed'))

# skip connection label
ax.text(7.0, 6.35, 'skip connections (concat)', ha='center', va='center',
        fontsize=7.5, color=SKIP_COL, style='italic')

# ── Output block ──────────────────────────────────────────────────────────────
OUT_X = DX[-1]
OUT_Y = DY[-1] + 0.85
block(OUT_X, OUT_Y, BW + 0.2, BH - 0.1, C_OUT, B_OUT, 'Output', 'C-class\n1×1 Conv')
ax.annotate('', xy=(OUT_X, OUT_Y - BH/2 + 0.05),
            xytext=(OUT_X, DY[-1] + BH/2),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.2))

# ── Input label ───────────────────────────────────────────────────────────────
ax.text(EX[0], EY[0] + 0.6, 'Input image\nH × W × 3', ha='center',
        va='center', fontsize=7.5, color=TEXT_MAIN, fontweight='bold')
ax.annotate('', xy=(EX[0], EY[0] + BH/2),
            xytext=(EX[0], EY[0] + 0.42),
            arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.2))

# ── Section labels ────────────────────────────────────────────────────────────
ax.text(2.8, 0.35, 'Encoder (contracting path)',
        ha='center', va='center', fontsize=8, color='#2563eb', fontweight='bold')
ax.text(11.2, 0.35, 'Decoder (expanding path)',
        ha='center', va='center', fontsize=8, color='#15803d', fontweight='bold')
ax.text(BX, 0.35, 'Bottleneck',
        ha='center', va='center', fontsize=8, color='#b45309', fontweight='bold')

ax.set_title('U-Net Architecture', fontsize=12, color=TEXT_MAIN,
             fontweight='bold', pad=6)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'unet_architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved unet_architecture.png')
