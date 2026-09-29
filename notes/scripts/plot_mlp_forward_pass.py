import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title("MLP Forward Pass — XOR input x=[0,1], expected output=1", fontsize=12, fontweight='bold')

# ── Neuron positions ──────────────────────────────────────────────────────────
input_neurons  = [(2, 4.0), (2, 2.0)]
hidden_neurons = [(5.5, 4.5), (5.5, 1.5)]
output_neuron  = [(9, 3.0)]

layer_colors = {'input': '#d1ecf1', 'hidden': '#fff3cd', 'output': '#d4edda'}

def draw_neuron(ax, x, y, top_label, bottom_label, color):
    circle = plt.Circle((x, y), 0.5, color=color, ec='gray', linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y + 0.15, top_label,    ha='center', va='center', fontsize=9,  fontweight='bold', zorder=4)
    ax.text(x, y - 0.18, bottom_label, ha='center', va='center', fontsize=8,  color='dimgray',   zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, label, label_color='dimgray'):
    ax.annotate("", xy=(x2 - 0.5, y2), xytext=(x1 + 0.5, y1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(mx, my + 0.18, label, ha='center', fontsize=8, color=label_color, style='italic')

# ── Input neurons ─────────────────────────────────────────────────────────────
draw_neuron(ax, *input_neurons[0], 'x₁', '0', layer_colors['input'])
draw_neuron(ax, *input_neurons[1], 'x₂', '1', layer_colors['input'])

# ── Hidden neuron h1: w=[1,1], b=-0.5 → z=0.5 → output=1 ────────────────────
draw_neuron(ax, *hidden_neurons[0], 'h₁', 'z=0.5 → 1', layer_colors['hidden'])
# ── Hidden neuron h2: w=[-1,-1], b=1.5 → z=0.5 → output=1 ───────────────────
draw_neuron(ax, *hidden_neurons[1], 'h₂', 'z=0.5 → 1', layer_colors['hidden'])

# ── Output neuron: w=[1,1], b=-1.5 → z=0.5 → output=1 ───────────────────────
draw_neuron(ax, *output_neuron[0], 'ŷ', 'z=0.5 → 1', layer_colors['output'])

# ── Connections input → hidden ────────────────────────────────────────────────
draw_arrow(ax, *input_neurons[0], *hidden_neurons[0], 'w=1, x=0 → 0')
draw_arrow(ax, *input_neurons[1], *hidden_neurons[0], 'w=1, x=1 → 1')
draw_arrow(ax, *input_neurons[0], *hidden_neurons[1], 'w=-1, x=0 → 0')
draw_arrow(ax, *input_neurons[1], *hidden_neurons[1], 'w=-1, x=1 → -1')

# ── Connections hidden → output ───────────────────────────────────────────────
draw_arrow(ax, *hidden_neurons[0], *output_neuron[0], 'w=1, h=1 → 1')
draw_arrow(ax, *hidden_neurons[1], *output_neuron[0], 'w=1, h=1 → 1')

# ── Input arrows ──────────────────────────────────────────────────────────────
for x, y in input_neurons:
    ax.annotate("", xy=(x - 0.5, y), xytext=(x - 1.3, y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# ── Output arrow ──────────────────────────────────────────────────────────────
ox, oy = output_neuron[0]
ax.annotate("", xy=(ox + 1.3, oy), xytext=(ox + 0.5, oy),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
ax.text(ox + 1.5, oy, '1 ✓', fontsize=12, fontweight='bold', color='seagreen', va='center')

# ── Layer labels ──────────────────────────────────────────────────────────────
ax.text(2,   5.3, "Input",             ha='center', fontsize=10, color='steelblue', fontweight='bold')
ax.text(5.5, 5.3, "Layer 1 — Hidden",  ha='center', fontsize=10, color='goldenrod', fontweight='bold')
ax.text(9,   5.3, "Layer 2 — Output",  ha='center', fontsize=10, color='seagreen',  fontweight='bold')

# ── Bias annotations ─────────────────────────────────────────────────────────
ax.text(5.5, 5.0, "b₁=-0.5   b₂=+1.5", ha='center', fontsize=8, color='gray', style='italic')
ax.text(9,   2.3, "b=-1.5",             ha='center', fontsize=8, color='gray', style='italic')

# ── Computation box ───────────────────────────────────────────────────────────
ax.text(5.5, 0.35,
        "h₁: z = 1(0)+1(1)−0.5 = 0.5 → 1     h₂: z = −1(0)−1(1)+1.5 = 0.5 → 1     out: z = 1(1)+1(1)−1.5 = 0.5 → 1",
        ha='center', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'mlp_forward_pass.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved to {os.path.abspath(output_path)}")

