import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

ax.axhline(0, color='#8b949e', linewidth=1.5)
ax.axvline(0, color='#8b949e', linewidth=1.5)

ax.annotate('', xy=(0, 1.05), xytext=(0, -1.05),
            arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))
ax.annotate('', xy=(1.05, 0), xytext=(-1.05, 0),
            arrowprops=dict(arrowstyle='->', color='#8b949e', lw=1.5))

ax.text(0, 1.1, 'THINKING', ha='center', va='bottom', color='#8b949e', fontsize=11, fontweight='bold')
ax.text(0, -1.1, 'ACTING', ha='center', va='top', color='#8b949e', fontsize=11, fontweight='bold')
ax.text(-1.1, 0, 'HUMANLY', ha='right', va='center', color='#8b949e', fontsize=11, fontweight='bold')
ax.text(1.1, 0, 'RATIONALLY', ha='left', va='center', color='#8b949e', fontsize=11, fontweight='bold')

quadrants = [
    (-0.55, 0.45, 'Thinking\nHumanly', '#388bfd'),
    (0.55, 0.45, 'Thinking\nRationally', '#3fb950'),
    (-0.55, -0.45, 'Acting\nHumanly', '#388bfd'),
    (0.55, -0.45, 'Acting\nRationally', '#f0883e'),
]

for x, y, label, color in quadrants:
    ax.text(x, y, label, ha='center', va='center', color=color,
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22', edgecolor=color, linewidth=1.5))

subtexts = [
    (-0.55, 0.18, 'Cognitive science\nBrain simulation', '#8b949e'),
    (0.55, 0.18, 'Logic systems\nTheorem provers', '#8b949e'),
    (-0.55, -0.72, 'Turing Test', '#8b949e'),
    (0.55, -0.72, 'Most of modern AI\n← This course', '#f0883e'),
]
for x, y, label, color in subtexts:
    ax.text(x, y, label, ha='center', va='center', color=color, fontsize=9)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis('off')
ax.set_title("Russell & Norvig's AI Definition Space", color='#e6edf3', fontsize=13, pad=12)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'ai_quadrants.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
