import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5)
ax.axis('off')
fig.patch.set_facecolor('white')

TEXT_MAIN = '#111111'
TEXT_SUB  = '#374151'
TEXT_DIM  = '#6b7280'
ARROW     = '#9ca3af'
DIV       = '#d1d5db'

C_PT  = '#dbeafe'; B_PT  = '#93c5fd'   # blue   — PyTorch
C_TF  = '#fce7f3'; B_TF  = '#f9a8d4'   # pink   — TensorFlow
C_KR  = '#dcfce7'; B_KR  = '#86efac'   # green  — Keras
C_COM = '#fef3c7'; B_COM = '#fcd34d'   # amber  — common workflow


def box(cx, cy, w, h, fill, border, label, sub=''):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.07',
        facecolor=fill, edgecolor=border, linewidth=1.5, zorder=2))
    ax.text(cx, cy + (0.16 if sub else 0), label,
            ha='center', va='center', fontsize=9,
            color=TEXT_MAIN, fontweight='bold', zorder=3)
    if sub:
        ax.text(cx, cy - 0.22, sub, ha='center', va='center',
                fontsize=7.5, color=TEXT_SUB, zorder=3)


def arr(x1, y1, x2, y2, color=ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.4))


# ── Common workflow row (top) ─────────────────────────────────────────────────
steps = ['Define\nModel', 'Configure\nLoss & Opt', 'Prepare\nData',
         'Train\nLoop', 'Evaluate', 'Deploy']
xs_top = [1.1, 3.0, 4.9, 6.8, 8.7, 10.6]
for x, s in zip(xs_top, steps):
    box(x, 4.1, 1.5, 0.65, C_COM, B_COM, s)
for i in range(len(xs_top) - 1):
    arr(xs_top[i] + 0.75, 4.1, xs_top[i+1] - 0.75, 4.1)

ax.text(6.5, 4.72, 'Common Workflow (framework-independent)',
        ha='center', va='center', fontsize=9, color='#b45309', fontweight='bold')

# ── Divider ───────────────────────────────────────────────────────────────────
ax.plot([0.3, 12.7], [3.45, 3.45], color=DIV, lw=1.2, ls='--')

# ── Three framework columns ───────────────────────────────────────────────────
col_data = [
    (2.2,  C_PT, B_PT,  'PyTorch',     'Dynamic graph\ndefine-by-run',
     ['torch.Tensor', 'nn.Module', 'torch.optim', 'DataLoader'],
     'Research\ndebugging'),
    (6.5,  C_TF, B_TF,  'TensorFlow',  'Eager + @tf.function\nstatic graph opt',
     ['tf.Tensor', 'tf.Variable', 'GradientTape', 'tf.data'],
     'Production\nGoogle ecosystem'),
    (10.8, C_KR, B_KR,  'Keras',       'High-level API\non TensorFlow',
     ['Sequential', 'Functional API', 'compile/fit', 'callbacks'],
     'Rapid\nprototyping'),
]

for cx, fill, border, name, subtitle, items, usecase in col_data:
    # Header box
    box(cx, 3.0, 3.2, 0.65, fill, border, name, subtitle)
    # Item list
    for i, item in enumerate(items):
        y = 2.25 - i * 0.42
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - 1.4, y - 0.16), 2.8, 0.32,
            boxstyle='round,pad=0.04',
            facecolor='white', edgecolor=border, linewidth=1.0, zorder=2))
        ax.text(cx, y, item, ha='center', va='center',
                fontsize=8, color=TEXT_MAIN, zorder=3)
    # Use case label
    ax.text(cx, 0.38, usecase, ha='center', va='center',
            fontsize=7.5, color=TEXT_SUB, style='italic')

# ── Keras sits on TF arrow ────────────────────────────────────────────────────
ax.annotate('', xy=(8.9, 3.0), xytext=(7.8, 3.0),
            arrowprops=dict(arrowstyle='->', color='#15803d', lw=1.3,
                            linestyle='dashed'))
ax.text(8.35, 3.22, 'built on', ha='center', fontsize=7, color='#15803d')

ax.set_title('Deep Learning Frameworks', fontsize=12,
             color=TEXT_MAIN, fontweight='bold', pad=6)

out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'dl_frameworks.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Saved dl_frameworks.png')
