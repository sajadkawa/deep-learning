import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(2.2, 8.2)
ax.axis('off')

# Layer x positions
x_input  = 1.5
x_h1     = 3.5
x_h2     = 5.5
x_output = 7.5

# Neuron y positions per layer
y_input  = [6.5, 5.7, 4.9, 4.1]   # 4 input neurons
y_h1     = [6.5, 5.7, 4.9, 4.1]   # 4 hidden neurons (layer 1)
y_h2     = [6.2, 5.3, 4.4]        # 3 hidden neurons (layer 2)
y_output = [6.2, 5.3, 4.4]        # 3 output neurons

neuron_r = 0.28
colors = {
    'input':  '#dbeafe',
    'hidden': '#fef3c7',
    'output': '#dcfce7',
}
borders = {
    'input':  '#2563eb',
    'hidden': '#b45309',
    'output': '#15803d',
}

def draw_layer(ax, x, ys, color, border, label, sublabels):
    for i, y in enumerate(ys):
        circle = plt.Circle((x, y), neuron_r, facecolor=color, zorder=3, linewidth=1.2,
                             edgecolor=border)
        ax.add_patch(circle)
        ax.text(x, y, sublabels[i], ha='center', va='center', fontsize=8,
                fontweight='bold', color='#111111', zorder=4)
    ax.text(x, min(ys) - 0.55, label, ha='center', va='center', fontsize=9,
            color='#111111', fontweight='bold')

def draw_connections(ax, x1, ys1, x2, ys2):
    for y1 in ys1:
        for y2 in ys2:
            ax.plot([x1 + neuron_r, x2 - neuron_r], [y1, y2],
                    color='#d1d5db', linewidth=0.6, zorder=1)

# Draw connections first (behind neurons)
draw_connections(ax, x_input, y_input, x_h1, y_h1)
draw_connections(ax, x_h1,    y_h1,    x_h2, y_h2)
draw_connections(ax, x_h2,    y_h2,    x_output, y_output)

# Draw layers
draw_layer(ax, x_input,  y_input,  colors['input'],  borders['input'],  'Input\na⁽⁰⁾',
           ['x₁','x₂','x₃','x₄'])
draw_layer(ax, x_h1,     y_h1,     colors['hidden'], borders['hidden'], 'Hidden 1\na⁽¹⁾',
           ['h₁','h₂','h₃','h₄'])
draw_layer(ax, x_h2,     y_h2,     colors['hidden'], borders['hidden'], 'Hidden 2\na⁽²⁾',
           ['h₁','h₂','h₃'])
draw_layer(ax, x_output, y_output, colors['output'], borders['output'], 'Output\nŷ = a⁽³⁾',
           ['ŷ₁','ŷ₂','ŷ₃'])

# Layer labels at top
for x, lbl in [(x_input,'n₀ = 4'), (x_h1,'n₁ = 4'),
               (x_h2,'n₂ = 3'), (x_output,'n₃ = 3')]:
    ax.text(x, 7.1, lbl, ha='center', va='center', fontsize=8.5, color='#374151')

# Arrows between layers (one representative arrow per gap)
for x1, x2 in [(x_input, x_h1), (x_h1, x_h2), (x_h2, x_output)]:
    ax.annotate('', xy=(x2 - neuron_r - 0.05, 5.0),
                xytext=(x1 + neuron_r + 0.05, 5.0),
                arrowprops=dict(arrowstyle='->', color='#9ca3af', lw=1.2))

ax.text(5.0, 7.8, 'Deep Feedforward Network — fully connected layers',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#111111')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'deep_ff_architecture.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved deep_ff_architecture.png")
