import matplotlib.pyplot as plt
import os

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.patch.set_facecolor('white')

titles = ['Positive Phase', 'Negative Phase', 'Weight Update']
subtitles = [
    'Clamp v to data\nInfer h ~ P(h|v)',
    'Reconstruct v̂ ~ P(v|h)\nInfer ĥ ~ P(h|v̂)',
    'Δw = η(⟨vhᵀ⟩ₐₑₐₐ − ⟨v̂ĥᵀ⟩ₘₒₑₑₗ)',
]

n_v, n_h = 3, 2
v_ys = [0.8, 1.6, 2.4]
h_ys = [1.2, 2.0]
v_x, h_x = 0.6, 2.0

v_colors = ['#dbeafe', '#dcfce7']   # positive phase: blue; negative phase: green
h_colors = ['#fef3c7', '#fee2e2']   # positive phase: amber; negative phase: red

for col, ax in enumerate(axes):
    ax.set_facecolor('white')
    ax.set_xlim(0, 2.6)
    ax.set_ylim(0.2, 3.2)
    ax.axis('off')

    for vy in v_ys:
        for hy in h_ys:
            ax.plot([v_x, h_x], [vy, hy], color='#d1d5db', linewidth=1.0, zorder=1)

    vc = v_colors[0] if col == 0 else v_colors[1]
    hc = h_colors[0] if col == 0 else h_colors[1]
    ve = '#2563eb' if col == 0 else '#15803d'
    he = '#b45309' if col == 0 else '#dc2626'

    for i, vy in enumerate(v_ys):
        c = plt.Circle((v_x, vy), 0.22, facecolor=vc, edgecolor=ve, linewidth=1.5, zorder=3)
        ax.add_patch(c)
        ax.text(v_x, vy, f'v{i+1}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='#111111', zorder=4)

    for j, hy in enumerate(h_ys):
        c = plt.Circle((h_x, hy), 0.22, facecolor=hc, edgecolor=he, linewidth=1.5, zorder=3)
        ax.add_patch(c)
        ax.text(h_x, hy, f'h{j+1}', ha='center', va='center',
                fontsize=8, fontweight='bold', color='#111111', zorder=4)

    ax.text(v_x, 0.35, 'v', ha='center', fontsize=9, color='#374151')
    ax.text(h_x, 0.35, 'h', ha='center', fontsize=9, color='#374151')

    ax.text(1.3, 3.05, titles[col], ha='center', va='center',
            fontsize=10, fontweight='bold', color='#111111')
    ax.text(1.3, 2.72, subtitles[col], ha='center', va='center',
            fontsize=7.5, color='#374151', linespacing=1.5)

for spine_x in [0.355, 0.688]:
    fig.add_artist(plt.matplotlib.patches.FancyArrowPatch(
        (spine_x, 0.5), (spine_x + 0.005, 0.5),
        transform=fig.transFigure,
        arrowstyle='->', color='#9ca3af',
        mutation_scale=15, linewidth=1.5
    ))

plt.tight_layout(pad=1.2)
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'cd1_training.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved cd1_training.png")
