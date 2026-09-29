import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title("MLP Architecture — 2 inputs, 2 hidden neurons, 1 output", fontsize=13, fontweight='bold')

# ── Neuron positions ──────────────────────────────────────────────────────────
input_neurons  = [(2, 5.5), (2, 3.5)]
hidden_neurons = [(5, 6), (5, 3)]
output_neuron  = [(8, 4.5)]

layer_colors = {
    'input':  '#d1ecf1',
    'hidden': '#fff3cd',
    'output': '#d4edda',
}

def draw_neuron(ax, x, y, label, color, fontsize=10):
    circle = plt.Circle((x, y), 0.45, color=color, ec='gray', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=4)

def draw_connection(ax, x1, y1, x2, y2, label='', color='gray'):
    ax.annotate("", xy=(x2 - 0.45, y2), xytext=(x1 + 0.45, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.15, label, ha='center', fontsize=8, color='dimgray', style='italic')

# ── Draw connections ──────────────────────────────────────────────────────────
weight_labels_1 = [
    ('w¹₁₁', input_neurons[0], hidden_neurons[0]),
    ('w¹₁₂', input_neurons[0], hidden_neurons[1]),
    ('w¹₂₁', input_neurons[1], hidden_neurons[0]),
    ('w¹₂₂', input_neurons[1], hidden_neurons[1]),
]
for label, (x1,y1), (x2,y2) in weight_labels_1:
    draw_connection(ax, x1, y1, x2, y2, label=label)

weight_labels_2 = [
    ('w²₁₁', hidden_neurons[0], output_neuron[0]),
    ('w²₂₁', hidden_neurons[1], output_neuron[0]),
]
for label, (x1,y1), (x2,y2) in weight_labels_2:
    draw_connection(ax, x1, y1, x2, y2, label=label)

# ── Draw neurons ──────────────────────────────────────────────────────────────
for i, (x, y) in enumerate(input_neurons):
    draw_neuron(ax, x, y, f'x{i+1}', layer_colors['input'])

for i, (x, y) in enumerate(hidden_neurons):
    draw_neuron(ax, x, y, f'h{i+1}', layer_colors['hidden'])

draw_neuron(ax, *output_neuron[0], 'ŷ', layer_colors['output'])

# ── Input arrows ──────────────────────────────────────────────────────────────
for x, y in input_neurons:
    ax.annotate("", xy=(x - 0.45, y), xytext=(x - 1.2, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# ── Output arrow ──────────────────────────────────────────────────────────────
ox, oy = output_neuron[0]
ax.annotate("", xy=(ox + 1.2, oy), xytext=(ox + 0.45, oy),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# ── Layer labels ──────────────────────────────────────────────────────────────
ax.text(2,   7.3, "Input",             ha='center', fontsize=10, color='steelblue', fontweight='bold')
ax.text(5,   7.3, "Layer 1 — Hidden",  ha='center', fontsize=10, color='goldenrod', fontweight='bold')
ax.text(8,   7.3, "Layer 2 — Output",  ha='center', fontsize=10, color='seagreen',  fontweight='bold')

# ── Bias annotations ─────────────────────────────────────────────────────────
ax.text(5, 2.2, "(+b₁⁽¹⁾, +b₂⁽¹⁾)", ha='center', fontsize=9, color='gray', style='italic')
ax.text(8, 3.7, "(+b⁽²⁾)",           ha='center', fontsize=9, color='gray', style='italic')

# ── Formula box — general to specific ────────────────────────────────────────
formulas = [
    "ŷ = f(x)",
    "h = f( W⁽¹⁾ᵀx + b⁽¹⁾ )",
    "ŷ = f( W⁽²⁾ᵀh + b⁽²⁾ )",
]

line_y = 1.9
for line in formulas:
    ax.text(5, line_y, line, ha='center', fontsize=9.5, family='monospace')
    line_y -= 0.4

from matplotlib.patches import FancyBboxPatch
box = FancyBboxPatch((2.5, line_y + 0.15), 5.0, 1.9 - (line_y + 0.15),
                     boxstyle='round,pad=0.1', linewidth=0.8,
                     edgecolor='gray', facecolor='lightyellow', alpha=0.5, zorder=0)
ax.add_patch(box)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'mlp_architecture.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {os.path.abspath(output_path)}")

