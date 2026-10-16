import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

c_chain = '#dc2626'   # red — chain rule path
c_fwd   = '#9ca3af'   # gray — forward arrows
c_blind = '#b45309'   # amber — what rule cannot see
c_known = '#15803d'   # green — what rule knows

fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

x_in, x_h, x_out, x_L = 1.8, 5.2, 8.6, 12.0
y_x1, y_x2             = 6.8, 4.2
y_h1, y_h2             = 6.8, 4.2
y_out                  = 5.5
y_L                    = 5.5
node_r                 = 0.52

def node(x, y, label, border='#93c5fd', lw=1.5, bg_col='#eff6ff', fc='#111111'):
    ax.add_patch(plt.Circle((x, y), node_r, color=bg_col, zorder=4,
                             linewidth=lw, ec=border))
    ax.text(x, y, label, ha='center', va='center',
            fontsize=11, color=fc, zorder=5, fontweight='bold')

def fwd_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c_fwd, lw=1.3,
                                shrinkA=node_r*72, shrinkB=node_r*72),
                zorder=3)

def back_arrow(x1, y1, x2, y2, rad=-0.25):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<-', color=c_chain, lw=2.3,
                                shrinkA=node_r*72, shrinkB=node_r*72,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=6)

def chain_label(x, y, text):
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5,
            color=c_chain, fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor='none', alpha=0.95))

# ── forward arrows ────────────────────────────────────────────────────────────
for (xa, ya, xb, yb) in [
    (x_in, y_x1, x_h,   y_h1),
    (x_in, y_x1, x_h,   y_h2),
    (x_in, y_x2, x_h,   y_h1),
    (x_in, y_x2, x_h,   y_h2),
    (x_h,  y_h1, x_out, y_out),
    (x_h,  y_h2, x_out, y_out),
    (x_out, y_out, x_L, y_L),
]:
    fwd_arrow(xa, ya, xb, yb)

# ── backward chain arrows ─────────────────────────────────────────────────────
back_arrow(x_L,   y_L   + 0.25, x_out, y_out + 0.25, rad=-0.3)
back_arrow(x_out, y_out + 0.25, x_h,   y_h1  + 0.25, rad=-0.25)
back_arrow(x_h,   y_h1  + 0.25, x_in,  y_x1  + 0.25, rad=-0.25)

# ── chain factor labels ───────────────────────────────────────────────────────
chain_label((x_out + x_L) / 2, y_out + 1.15, '∂L/∂ŷ')
chain_label(x_out,              y_out + 1.15, "f '(z⁽²⁾)")
chain_label((x_h + x_out) / 2, y_h1  + 1.15, 'w⁽²⁾₁₁')
chain_label(x_h,                y_h1  + 1.15, "f '(z₁⁽¹⁾)")
chain_label((x_in + x_h) / 2,  y_x1  + 1.15, 'x₁')

# ── nodes ─────────────────────────────────────────────────────────────────────
node(x_in, y_x1, 'x₁', border='#93c5fd', bg_col='#eff6ff')
node(x_in, y_x2, 'x₂', border='#93c5fd', bg_col='#eff6ff')
node(x_h,  y_h1, 'h₁', border=c_chain,   bg_col='#fef2f2', lw=2.5)
node(x_h,  y_h2, 'h₂', border='#d1d5db', bg_col='#f9fafb')
node(x_out, y_out, 'ŷ', border=c_chain,  bg_col='#fef2f2', lw=2.5)
node(x_L,  y_L,  'L',  border=c_chain,   bg_col='#fef2f2', lw=2.5, fc=c_chain)

# ── weight labels ─────────────────────────────────────────────────────────────
ax.text((x_in + x_h)/2, (y_x1 + y_h1)/2 - 0.45,
        'w⁽¹⁾₁₁', ha='center', fontsize=9.5, color='white',
        fontweight='bold', zorder=7,
        bbox=dict(boxstyle='round,pad=0.22', facecolor=c_chain,
                  edgecolor=c_chain, linewidth=1.4))
ax.text((x_h + x_out)/2, (y_h1 + y_out)/2 - 0.45,
        'w⁽²⁾₁₁', ha='center', fontsize=9, color='#9ca3af', zorder=7)

# ── column labels ─────────────────────────────────────────────────────────────
for xp, lbl in [(x_in, 'Input Layer'), (x_h, 'Hidden Layer'),
                (x_out, 'Output Layer'), (x_L, 'Loss')]:
    ax.text(xp, 3.45, lbl, ha='center', va='center',
            fontsize=8.5, color='#9ca3af', style='italic')

# ── chain rule equation box ───────────────────────────────────────────────────
ax.text(7.0, 8.55,
        "∂L/∂w⁽¹⁾₁₁  =  ∂L/∂ŷ  ·  f '(z⁽²⁾)  ·  w⁽²⁾₁₁  ·  f '(z₁⁽¹⁾)  ·  x₁",
        ha='center', va='center', fontsize=11, color='#111111', zorder=8,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9c3',
                  edgecolor='#ca8a04', linewidth=1.8))

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(7.0, 9.55, 'Why the Perceptron Rule Cannot Train Hidden Weights',
        ha='center', va='center', fontsize=13,
        color='#111111', fontweight='bold', zorder=8)
ax.text(7.0, 9.1,
        'The gradient ∂L/∂w⁽¹⁾₁₁ requires five terms — the perceptron rule has access to only two.',
        ha='center', va='center', fontsize=9, color='#6b7280', style='italic', zorder=8)

# ── bottom boxes ─────────────────────────────────────────────────────────────
# Green — what the rule knows
bx, by, bw, bh = 0.3, 0.25, 4.2, 2.8
ax.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh,
    boxstyle='round,pad=0.12', linewidth=1.5,
    edgecolor=c_known, facecolor='#f0fdf4', zorder=4))
ax.text(bx + bw/2, by + bh - 0.30, 'Perceptron rule knows:',
        ha='center', fontsize=9.5, color=c_known, fontweight='bold', zorder=5)
ax.text(bx + bw/2, by + bh - 0.72, 'wᵢ ← wᵢ + η · (y − ŷ) · xᵢ',
        ha='center', fontsize=9.5, color='#111111', fontfamily='monospace', zorder=5)
ax.text(bx + bw/2, by + bh - 1.18, '─────────────────────',
        ha='center', fontsize=8, color='#d1d5db', zorder=5)
ax.text(bx + bw/2, by + bh - 1.55, '✓  (y − ŷ)  — output error',
        ha='center', fontsize=9, color=c_known, zorder=5)
ax.text(bx + bw/2, by + bh - 1.95, '✓  xᵢ  — input value',
        ha='center', fontsize=9, color=c_known, zorder=5)
ax.text(bx + bw/2, by + bh - 2.35, '✓  η  — learning rate',
        ha='center', fontsize=9, color=c_known, zorder=5)

# Amber — what the rule cannot see
bx2, by2, bw2, bh2 = 5.0, 0.25, 8.6, 2.8
ax.add_patch(mpatches.FancyBboxPatch((bx2, by2), bw2, bh2,
    boxstyle='round,pad=0.12', linewidth=1.5,
    edgecolor=c_blind, facecolor='#fffbeb', zorder=4))
ax.text(bx2 + bw2/2, by2 + bh2 - 0.30, 'Perceptron rule cannot see:',
        ha='center', fontsize=9.5, color=c_blind, fontweight='bold', zorder=5)
ax.text(bx2 + bw2/2, by2 + bh2 - 0.72,
        '(required by the chain rule — structurally inaccessible)',
        ha='center', fontsize=8.5, color='#6b7280', style='italic', zorder=5)
ax.text(bx2 + bw2/2, by2 + bh2 - 1.18,
        '─────────────────────────────────────────────',
        ha='center', fontsize=8, color='#d1d5db', zorder=5)

blind_items = [
    ("✗  ∂L/∂ŷ",      "gradient of loss w.r.t. output prediction"),
    ("✗  f '(z⁽²⁾)",  "derivative of output activation function"),
    ("✗  w⁽²⁾₁₁",     "weight in the layer above the hidden weight"),
    ("✗  f '(z₁⁽¹⁾)", "derivative of hidden neuron activation"),
]
for i, (sym, desc) in enumerate(blind_items):
    y_item = by2 + bh2 - 1.55 - i * 0.40
    ax.text(bx2 + 0.25, y_item, sym,
            ha='left', fontsize=9, color=c_blind, zorder=5, fontweight='bold')
    ax.text(bx2 + 2.55, y_item, f'— {desc}',
            ha='left', fontsize=8.5, color='#6b7280', zorder=5)

plt.tight_layout(pad=0.3)
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'credit_assignment.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {os.path.abspath(out)}')
