import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)
ax.axis('off')

# --- colours: light pastel fills, darker borders, dark text ---
# box: (fill, border)
c_x     = ('#dbeafe', '#1d4ed8')   # blue
c_f     = ('#ffedd5', '#c2410c')   # orange
c_theta = ('#f3e8ff', '#7e22ce')   # purple
c_y     = ('#dcfce7', '#15803d')   # green
c_loss  = ('#fee2e2', '#b91c1c')   # red
col_arrow = '#374151'
col_title = '#111111'
col_label = '#111111'   # main label inside box
col_sub   = '#374151'   # sublabel inside box
col_annot = '#111111'   # training equation text

def box(ax, cx, cy, w, h, label, sublabel, fc, ec, fontsize=11):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle='round,pad=0.10',
                          facecolor=fc, edgecolor=ec,
                          linewidth=2, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(cx, cy + 0.15, label, ha='center', va='center',
                color=col_label, fontsize=fontsize, fontweight='bold', zorder=4)
        ax.text(cx, cy - 0.22, sublabel, ha='center', va='center',
                color=col_sub, fontsize=9.5, fontweight='bold', zorder=4)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                color=col_label, fontsize=fontsize, fontweight='bold', zorder=4)

def arrow(ax, x1, y1, x2, y2, style='arc3,rad=0'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=col_arrow,
                                lw=1.8, mutation_scale=16,
                                connectionstyle=style), zorder=2)

y_main  = 3.2
y_theta = 4.7
y_true  = 1.8
y_annot = 0.35

# --- boxes ---
box(ax, 1.4,  y_main,  1.3,  1.0,  'x',          'input vector',            *c_x,     fontsize=12)
box(ax, 5.0,  y_main,  2.6,  1.15, 'F(x ; θ)',   'neural network',          *c_f,     fontsize=14)
box(ax, 5.0,  y_theta, 2.4,  0.85, 'θ  (parameters)', 'weights & biases — learned', *c_theta, fontsize=11)
box(ax, 8.6,  y_main,  1.3,  1.0,  'ŷ',          'prediction',              *c_y,     fontsize=12)
box(ax, 8.6,  y_true,  1.3,  0.85, 'y',           'true label',              *c_y,     fontsize=12)
box(ax, 11.3, y_main,  1.9,  1.0,  'L(y, ŷ)',    'loss — scalar',           *c_loss,  fontsize=12)

# --- arrows ---
arrow(ax, 2.05, y_main,        3.7,   y_main)
arrow(ax, 5.0,  y_theta - 0.36, 5.0,  y_main + 0.50)
arrow(ax, 6.3,  y_main,        7.95,  y_main)
arrow(ax, 9.25, y_main,        10.35, y_main)
arrow(ax, 8.6,  y_true + 0.36, 8.6,   y_main - 0.43)
arrow(ax, 8.6,  y_main - 0.43, 10.35, y_main - 0.28, style='arc3,rad=-0.3')

# --- arrow labels ---
ax.text(2.87, y_main + 0.20, 'data',           ha='center', color=col_sub, fontsize=8)
ax.text(5.30, y_theta - 0.72, 'fixed during\nforward pass', ha='left', color=col_sub, fontsize=7.5)
ax.text(7.12, y_main + 0.20, 'ŷ = F(x ; θ)', ha='center', color=col_sub, fontsize=8)

# --- training objective ---
ax.text(6.5, y_annot,
        "Training  =  find  θ*  =  argmin  L( y,  F(x ; θ) )",
        ha='center', va='center', color=col_annot,
        fontsize=10.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#f9fafb',
                  edgecolor='#9ca3af', linewidth=1.5))

ax.set_title('A Neural Network as a Parametric Function',
             color=col_title, fontsize=13, pad=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'network_as_function.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved network_as_function.png")
