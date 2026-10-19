import matplotlib.pyplot as plt
import numpy as np
import os

# Three panels side by side, tall enough for all content
fig, axes = plt.subplots(1, 3, figsize=(17, 9))
fig.patch.set_facecolor('white')

inp = np.array([
    [1, 0, 2, 1, 0, 1],
    [0, 1, 1, 0, 2, 0],
    [2, 1, 0, 1, 1, 2],
    [1, 0, 1, 2, 0, 1],
    [0, 2, 1, 0, 1, 0],
    [1, 1, 0, 1, 2, 1]
])
filt = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
conv_out = np.array([[6, 4, 3, 7], [4, 3, 5, 4], [3, 5, 5, 3], [5, 5, 3, 6]])

# Fixed vertical zones (bottom to top):
#   0.0 – 1.2  : result label  ("= 6")
#   1.2 – 1.6  : gap
#   1.6 – 4.5  : filter grid  (3 rows × cell_f)
#   4.5 – 5.0  : gap + "Filter (3×3)" title
#   5.0 – 5.5  : "patch at (r,c)" label
#   5.5 – 9.5  : input grid   (6 rows × cell_i)
#   9.5 – 10.0 : "Input (6×6)" title

cell_i = 0.72   # input cell size
cell_f = 0.80   # filter cell size  (slightly larger for clarity)

INPUT_TOP  = 9.2   # y of top-left corner of input grid (row 0)
FILTER_TOP = 4.2   # y of top-left corner of filter grid
RESULT_Y   = 1.0   # y of result text

def draw_grid(ax, data, x0, y0, cell, highlight=None, hl_color='#ffcc44',
              edge_hl='#cc6600', fontsize=12):
    rows, cols = data.shape
    for r in range(rows):
        for c in range(cols):
            in_hl = highlight and (r, c) in highlight
            fc = hl_color if in_hl else '#f3f4f6'
            ec = edge_hl  if in_hl else '#d1d5db'
            lw = 2.5      if in_hl else 0.8
            rect = plt.Rectangle((x0 + c * cell, y0 - r * cell), cell, cell,
                                  facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
            ax.add_patch(rect)
            ax.text(x0 + c * cell + cell / 2, y0 - r * cell + cell / 2,
                    str(data[r, c]), ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color='#111111', zorder=3)

def draw_panel(ax, panel_title, patch_col,
               inp_hl_color, inp_edge_color,
               result_val, result_color):

    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_title(panel_title, fontsize=11, fontweight='bold', color='#111111', pad=10)

    x0 = 0.3   # left margin for both grids

    # ── INPUT GRID ──────────────────────────────────────────────
    hl_inp = {(r, c + patch_col) for r in range(3) for c in range(3)}
    draw_grid(ax, inp, x0, INPUT_TOP, cell_i,
              highlight=hl_inp, hl_color=inp_hl_color, edge_hl=inp_edge_color)

    # "Input (6×6)" title — above the grid
    ax.text(x0 + 3 * cell_i, INPUT_TOP + 0.45,
            'Input (6×6)', ha='center', fontsize=11, fontweight='bold', color='#111111')

    # Dashed border around the active patch
    patch_x = x0 + patch_col * cell_i
    patch_y_top = INPUT_TOP
    rect_b = plt.Rectangle((patch_x, patch_y_top - 2 * cell_i),
                             3 * cell_i, 3 * cell_i,
                             facecolor='none', edgecolor=inp_edge_color,
                             linewidth=2.8, linestyle='--', zorder=5)
    ax.add_patch(rect_b)

    # "patch at (0, col)" label — BELOW the input grid, clear of it
    label_y = INPUT_TOP - 5 * cell_i - 0.45   # below bottom row of input
    ax.text(x0 + 3 * cell_i, label_y,
            f'patch at (0,{patch_col})',
            ha='center', va='top', fontsize=10, color=inp_edge_color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=inp_edge_color, linewidth=1.5))

    # ── FILTER GRID ──────────────────────────────────────────────
    draw_grid(ax, filt, x0, FILTER_TOP, cell_f,
              highlight={(r, c) for r in range(3) for c in range(3)},
              hl_color='#dcfce7', edge_hl='#15803d')

    # "Filter (3×3)" title — above the filter grid
    ax.text(x0 + 1.5 * cell_f, FILTER_TOP + 0.42,
            'Filter (3×3)', ha='center', fontsize=11, fontweight='bold', color='#15803d')

    # ── RESULT ───────────────────────────────────────────────────
    # Arrow from right edge of filter to result
    arrow_x0 = x0 + 3 * cell_f + 0.15
    arrow_x1 = arrow_x0 + 0.65
    arrow_y  = FILTER_TOP - cell_f   # middle of filter grid
    ax.annotate('', xy=(arrow_x1, arrow_y), xytext=(arrow_x0, arrow_y),
                arrowprops=dict(arrowstyle='->', color=result_color, lw=2.2))
    ax.text(arrow_x1 + 0.15, arrow_y,
            f'= {result_val}',
            ha='left', va='center', fontsize=16, fontweight='bold', color=result_color)

    # ── AXIS LIMITS ──────────────────────────────────────────────
    ax.set_xlim(-0.2, 7.0)
    ax.set_ylim(0.2, INPUT_TOP + 0.8)

# ── Panel 1: patch at (0,0) ──────────────────────────────────────
draw_panel(axes[0],
           'Step 1: Filter at position (0,0)',
           patch_col=0,
           inp_hl_color='#fef3c7', inp_edge_color='#b45309',
           result_val=6, result_color='#b45309')

# ── Panel 2: patch at (0,1) ──────────────────────────────────────
draw_panel(axes[1],
           'Step 2: Filter slides one step right',
           patch_col=1,
           inp_hl_color='#dbeafe', inp_edge_color='#2563eb',
           result_val=4, result_color='#2563eb')

# ── Panel 3: full output feature map ─────────────────────────────
ax3 = axes[2]
ax3.set_facecolor('white')
ax3.axis('off')
ax3.set_title('Result: Full output feature map (4×4)', fontsize=11, fontweight='bold', color='#111111', pad=10)
ax3.set_xlim(-0.2, 6.0)
ax3.set_ylim(0.2, INPUT_TOP + 0.8)

cell_o = 1.0
x0_o = 0.5
y0_o = 8.0

draw_grid(ax3, conv_out, x0_o, y0_o, cell_o, fontsize=13)
ax3.text(x0_o + 2 * cell_o, y0_o + 0.45,
         'Output (4×4)', ha='center', fontsize=11, fontweight='bold', color='#111111')

# Formula box — centred below the output grid with plenty of room
formula_y = y0_o - 4 * cell_o - 0.7
ax3.text(x0_o + 2 * cell_o, formula_y,
         'Output size formula:\n\n(W − F) / S + 1\n\n= (6 − 3) / 1 + 1 = 4',
         ha='center', va='top', fontsize=10.5, color='#333', linespacing=1.6,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#fef9c3',
                   edgecolor='#ca8a04', linewidth=1.8))

ax3.text(x0_o + 2 * cell_o, 0.55,
         'W = input size    F = filter size    S = stride',
         ha='center', fontsize=9, color='#6b7280')

plt.tight_layout(pad=2.0)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'cnn_conv_layer_mechanics.png'), dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved cnn_conv_layer_mechanics.png")
