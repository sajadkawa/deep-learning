import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.axhline(0, color='#9ca3af', linewidth=1.5)
ax.axvline(0, color='#9ca3af', linewidth=1.5)

ax.annotate('', xy=(0, 1.05), xytext=(0, -1.05),
            arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.5))
ax.annotate('', xy=(1.05, 0), xytext=(-1.05, 0),
            arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.5))

ax.text(0, 1.1,  'THINKING',   ha='center', va='bottom', color='#374151', fontsize=11, fontweight='bold')
ax.text(0, -1.1, 'ACTING',     ha='center', va='top',    color='#374151', fontsize=11, fontweight='bold')
ax.text(-1.1, 0, 'HUMANLY',    ha='right',  va='center', color='#374151', fontsize=11, fontweight='bold')
ax.text(1.1,  0, 'RATIONALLY', ha='left',   va='center', color='#374151', fontsize=11, fontweight='bold')

# quadrant boxes: light pastel fills, darker borders
quadrants = [
    (-0.55,  0.45, 'Thinking\nHumanly',   '#dbeafe', '#1d4ed8'),
    ( 0.55,  0.45, 'Thinking\nRationally', '#dcfce7', '#15803d'),
    (-0.55, -0.45, 'Acting\nHumanly',     '#dbeafe', '#1d4ed8'),
    ( 0.55, -0.45, 'Acting\nRationally',  '#ffedd5', '#c2410c'),
]

for x, y, label, fc, ec in quadrants:
    ax.text(x, y, label, ha='center', va='center', color='#111111',
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=fc, edgecolor=ec, linewidth=1.5))

subtexts = [
    (-0.55,  0.18, 'Cognitive science\nBrain simulation', '#6b7280'),
    ( 0.55,  0.18, 'Logic systems\nTheorem provers',      '#6b7280'),
    (-0.55, -0.72, 'Turing Test',                         '#6b7280'),
    ( 0.55, -0.72, 'The rational\nagent approach', '#6b7280'),
]
for x, y, label, color in subtexts:
    ax.text(x, y, label, ha='center', va='center', color=color,
            fontsize=9, fontweight='bold')

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis('off')
ax.set_title("Russell & Norvig's AI Definition Space",
             color='#111111', fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'images', 'ai_quadrants.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved ai_quadrants.png")
