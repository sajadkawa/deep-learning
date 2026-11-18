import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('white')

TEXT_MAIN = '#111111'
TEXT_SUB  = '#374151'
TEXT_DIM  = '#6b7280'
ARROW     = '#9ca3af'

C_INPUT  = '#dbeafe'; B_INPUT  = '#93c5fd'
C_HIDDEN = '#fef3c7'; B_HIDDEN = '#fcd34d'
C_OUTPUT = '#dcfce7'; B_OUTPUT = '#86efac'
C_ANN    = '#fef9c3'; B_ANN    = '#ca8a04'


# ─────────────────────────────────────────────────────────────────────────────
# LEFT: Skip-gram architecture
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 8)
ax.set_ylim(0, 6)
ax.axis('off')

def neuron(ax, cx, cy, fill, border, r=0.22):
    ax.add_patch(plt.Circle((cx, cy), r, facecolor=fill,
                             edgecolor=border, linewidth=1.3, zorder=3))

def box(ax, cx, cy, w, h, fill, border, label, sub=''):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h, boxstyle='round,pad=0.06',
        facecolor=fill, edgecolor=border, linewidth=1.4, zorder=2))
    ax.text(cx, cy+(0.14 if sub else 0), label, ha='center', va='center',
            fontsize=8.5, color=TEXT_MAIN, fontweight='bold', zorder=3)
    if sub:
        ax.text(cx, cy-0.2, sub, ha='center', va='center',
                fontsize=7, color=TEXT_SUB, zorder=3)

def arr(ax, x1, y1, x2, y2, color=ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3))

# Input one-hot
vocab = ['deep', 'learning', 'neural', 'network', 'model']
in_ys = np.linspace(1.2, 4.8, 5)
for i, (w, y) in enumerate(zip(vocab, in_ys)):
    neuron(ax, 1.2, y, C_INPUT, B_INPUT)
    ax.text(0.7, y, w, ha='right', va='center', fontsize=7.5, color=TEXT_MAIN)
    if i == 0:  # "deep" is the centre word — filled darker
        ax.add_patch(plt.Circle((1.2, y), 0.22, facecolor='#2563eb',
                                edgecolor='#1d4ed8', linewidth=1.3, zorder=4))
        ax.text(0.7, y, w, ha='right', va='center', fontsize=7.5,
                color='#2563eb', fontweight='bold')

ax.text(1.2, 0.55, 'Input\n(one-hot)', ha='center', fontsize=7.5,
        color=TEXT_SUB)

# Embedding layer (hidden)
emb_ys = [2.5, 3.5]
for y in emb_ys:
    neuron(ax, 3.2, y, C_HIDDEN, B_HIDDEN)
ax.text(3.2, 0.55, 'Embedding\n$v_w \\in \\mathbb{R}^d$', ha='center',
        fontsize=7.5, color=TEXT_SUB)

# Arrows input → embedding
for iy in in_ys:
    for ey in emb_ys:
        arr(ax, 1.2+0.22, iy, 3.2-0.22, ey)

# Output layer
out_ys = np.linspace(1.2, 4.8, 5)
for i, (w, y) in enumerate(zip(vocab, out_ys)):
    neuron(ax, 5.8, y, C_OUTPUT, B_OUTPUT)
    ax.text(6.3, y, w, ha='left', va='center', fontsize=7.5, color=TEXT_MAIN)
    if i == 1:  # "learning" is the target context word
        ax.add_patch(plt.Circle((5.8, y), 0.22, facecolor='#15803d',
                                edgecolor='#166534', linewidth=1.3, zorder=4))
        ax.text(6.3, y, w, ha='left', va='center', fontsize=7.5,
                color='#15803d', fontweight='bold')

ax.text(5.8, 0.55, 'Output\n(softmax)', ha='center', fontsize=7.5,
        color=TEXT_SUB)

# Arrows embedding → output
for ey in emb_ys:
    for oy in out_ys:
        arr(ax, 3.2+0.22, ey, 5.8-0.22, oy)

# Labels
ax.text(1.2, 5.35, 'Centre word', ha='center', fontsize=8,
        color='#2563eb', fontweight='bold')
ax.text(5.8, 5.35, 'Context word', ha='center', fontsize=8,
        color='#15803d', fontweight='bold')

ax.set_title('Skip-Gram Architecture', fontsize=11,
             color=TEXT_MAIN, fontweight='bold', pad=6)


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT: 2D embedding space after training
# ─────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.2, 1.4)
ax2.set_facecolor('white')
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color('#d1d5db')
ax2.tick_params(colors='#6b7280', labelsize=7)
ax2.set_xlabel('Dimension 1', fontsize=8, color=TEXT_SUB)
ax2.set_ylabel('Dimension 2', fontsize=8, color=TEXT_SUB)

# Illustrative embedding positions (not from the worked example — conceptual)
words_pos = {
    'deep':        (-0.2,  0.8),
    'learning':    ( 0.3,  0.9),
    'neural':      (-0.1,  0.5),
    'network':     ( 0.4,  0.6),
    'model':       ( 0.6,  0.3),
    'king':        (-0.9,  0.7),
    'queen':       (-0.6,  0.9),
    'man':         (-0.8,  0.2),
    'woman':       (-0.5,  0.4),
}

colors = {
    'deep': '#2563eb', 'learning': '#2563eb', 'neural': '#2563eb',
    'network': '#2563eb', 'model': '#2563eb',
    'king': '#b45309', 'queen': '#b45309',
    'man': '#15803d', 'woman': '#15803d',
}

for word, (x, y) in words_pos.items():
    ax2.scatter(x, y, color=colors[word], s=60, zorder=4)
    ax2.text(x + 0.06, y + 0.04, word, fontsize=7.5,
             color=colors[word], fontweight='bold', zorder=5)

# Analogy arrow: king - man + woman ≈ queen
kx, ky = words_pos['king']
mx, my = words_pos['man']
wx, wy = words_pos['woman']
qx, qy = words_pos['queen']

# draw king→queen arrow
ax2.annotate('', xy=(qx, qy), xytext=(kx, ky),
             arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.3,
                             linestyle='dashed'))
ax2.annotate('', xy=(wy, wy), xytext=(mx, my),
             arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.3,
                             linestyle='dashed'))

# annotation box
ax2.text(0.0, -0.85,
         r'$\vec{\mathrm{king}} - \vec{\mathrm{man}} + \vec{\mathrm{woman}} \approx \vec{\mathrm{queen}}$',
         ha='center', fontsize=8, color='#dc2626',
         bbox=dict(facecolor='#fef9c3', edgecolor='#ca8a04',
                   boxstyle='round,pad=0.3', linewidth=1))

ax2.set_title('Embedding Space (conceptual)', fontsize=11,
              color=TEXT_MAIN, fontweight='bold', pad=6)

plt.tight_layout(pad=1.5)
out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'word2vec_embeddings.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved word2vec_embeddings.png')
