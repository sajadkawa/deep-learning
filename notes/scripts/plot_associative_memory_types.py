import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#f8f8f8')

def draw_pattern(ax, x, y, pattern, label, color, size=0.32):
    n = len(pattern)
    for i, val in enumerate(pattern):
        col = '#2c7bb6' if val == 1 else '#d7191c'
        rect = mpatches.FancyBboxPatch(
            (x + i * (size + 0.04), y), size, size,
            boxstyle="round,pad=0.02",
            facecolor=col, edgecolor='white', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + i * (size + 0.04) + size / 2, y + size / 2,
                '+1' if val == 1 else '-1',
                ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    total_w = n * (size + 0.04) - 0.04
    ax.text(x + total_w / 2, y - 0.18, label, ha='center', va='top', fontsize=9, color='#333333')

# ---- Left panel: Auto-associative ----
ax = axes[0]
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('#f8f8f8')
ax.set_title('Auto-Associative Memory', fontsize=12, fontweight='bold', pad=10, color='#222222')

# Stored pattern
draw_pattern(ax, 0.3, 2.8, [1, 1, -1, 1], 'stored pattern', '#2c7bb6')
# Noisy probe
draw_pattern(ax, 0.3, 1.6, [-1, 1, -1, 1], 'noisy probe\n(1 bit flipped)', '#e08020')
# Retrieved pattern
draw_pattern(ax, 0.3, 0.3, [1, 1, -1, 1], 'retrieved pattern', '#2c7bb6')

# Arrows
ax.annotate('', xy=(1.7, 1.55), xytext=(1.7, 1.0),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8))
ax.text(1.85, 1.28, 'network\nupdate', fontsize=8, color='#555555', va='center')

# Brace / label showing same pattern
ax.annotate('', xy=(3.7, 2.95), xytext=(3.7, 0.45),
            arrowprops=dict(arrowstyle='->', color='#2c7bb6', lw=1.5, linestyle='dashed'))
ax.text(3.82, 1.7, 'same\npattern', fontsize=8, color='#2c7bb6', va='center', rotation=90)

# ---- Right panel: Hetero-associative ----
ax = axes[1]
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('#f8f8f8')
ax.set_title('Hetero-Associative Memory', fontsize=12, fontweight='bold', pad=10, color='#222222')

# Input pattern xa
draw_pattern(ax, 0.3, 2.8, [1, 1, -1, 1], 'input pattern xa', '#2c7bb6')
# Output pattern ya (different)
draw_pattern(ax, 0.3, 0.3, [1, -1, 1, -1], 'output pattern ya', '#1a9641')

# Arrow
ax.annotate('', xy=(1.7, 0.85), xytext=(1.7, 1.55),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8))
ax.text(1.85, 1.2, 'W = (1/N)\nSUM y(x)^T', fontsize=8, color='#555555', va='center')

# Second pair
draw_pattern(ax, 0.3, 1.9, [-1, 1, 1, 1], 'input pattern xb', '#2c7bb6', size=0.28)
ax.annotate('', xy=(1.7, 1.55), xytext=(1.7, 2.15),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=1.2, linestyle='dotted'))

plt.tight_layout(pad=1.5)
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'associative_memory_types.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved associative_memory_types.png")
