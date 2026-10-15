import matplotlib.pyplot as plt
import os

events = [
    (1958, "Perceptron\n(Rosenblatt)",          0.3,  '#fef9c3', '#ca8a04'),
    (1969, "Minsky & Papert\nPerceptrons book",  -0.3, '#fee2e2', '#b91c1c'),
    (1986, "Backpropagation\n(Rumelhart et al)", 0.3,  '#fef9c3', '#ca8a04'),
    (1998, "LeNet CNN\n(LeCun)",                 -0.3, '#dbeafe', '#1d4ed8'),
    (2006, "Deep Belief Nets\n(Hinton)",          0.3,  '#fef9c3', '#ca8a04'),
    (2012, "AlexNet — GPU\nImageNet win",         -0.3, '#dcfce7', '#15803d'),
    (2017, "Transformer\n(Attention is All)",     0.3,  '#dcfce7', '#15803d'),
    (2020, "GPT-3, DALL-E\nLarge scale models",  -0.3, '#dcfce7', '#15803d'),
]

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
fig.suptitle("Deep Learning — Historical Timeline",
             fontsize=13, fontweight="bold", color='#111111')

ax.set_xlim(1945, 2028)
ax.set_ylim(-0.85, 0.85)
ax.axis("off")

# Main timeline line
ax.annotate("", xy=(2026, 0), xytext=(1948, 0),
            arrowprops=dict(arrowstyle="->", color='#374151', lw=2))

for year, label, y_pos, fc, ec in events:
    ax.plot(year, 0, "o", color='#374151', markersize=7, zorder=5)
    ax.plot([year, year], [0, y_pos * 0.75], color='#9ca3af', lw=1, linestyle="--")
    ax.text(year, y_pos, label, ha="center", va="center", fontsize=8,
            fontweight='bold', color='#111111',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=fc, edgecolor=ec, linewidth=1.2))
    y_year = -0.12 if y_pos > 0 else 0.12
    ax.text(year, y_year, str(year), ha="center", va="center",
            fontsize=8, fontweight="bold", color='#374151')

# Era labels
ax.text(1963, -0.72, "Early theory",          fontsize=8.5, color='#6b7280', style="italic")
ax.text(1990, -0.72, "AI winters & revivals",  fontsize=8.5, color='#6b7280', style="italic")
ax.text(2013, -0.72, "Deep Learning era",      fontsize=8.5, color='#15803d',
        style="italic", fontweight="bold")

# Era dividers
ax.axvline(1975, color='#e5e7eb', lw=1, linestyle=":")
ax.axvline(2011, color='#e5e7eb', lw=1, linestyle=":")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "..", "images", "dl_timeline.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white', edgecolor='none')
plt.close()
print("Saved dl_timeline.png")
