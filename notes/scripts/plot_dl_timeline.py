import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

events = [
    (1958, "Perceptron\n(Rosenblatt)",          0.3,  "#fff3cd"),
    (1969, "Minsky & Papert\nPerceptrons book",  -0.3, "#f8d7da"),
    (1986, "Backpropagation\n(Rumelhart et al)", 0.3,  "#fff3cd"),
    (1998, "LeNet CNN\n(LeCun)",                 -0.3, "#d1ecf1"),
    (2006, "Deep Belief Nets\n(Hinton)",          0.3,  "#fff3cd"),
    (2012, "AlexNet — GPU\nImageNet win",         -0.3, "#d4edda"),
    (2017, "Transformer\n(Attention is All)",     0.3,  "#d4edda"),
    (2020, "GPT-3, DALL-E\nLarge scale models",  -0.3, "#d4edda"),
]

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Deep Learning — Historical Timeline", fontsize=13, fontweight="bold")

ax.set_xlim(1945, 2028)
ax.set_ylim(-0.85, 0.85)
ax.axis("off")

# Main timeline line
ax.annotate("", xy=(2026, 0), xytext=(1948, 0),
            arrowprops=dict(arrowstyle="->", color="black", lw=2))

for year, label, y_pos, color in events:
    # Dot on timeline
    ax.plot(year, 0, "o", color="black", markersize=7, zorder=5)

    # Vertical line to box
    ax.plot([year, year], [0, y_pos * 0.75], color="gray", lw=1, linestyle="--")

    # Label box
    ax.text(year, y_pos, label, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="gray", linewidth=0.8))

    # Year label
    y_year = -0.12 if y_pos > 0 else 0.12
    ax.text(year, y_year, str(year), ha="center", va="center", fontsize=8, fontweight="bold")

# Era labels
ax.text(1963, -0.72, "Early theory", fontsize=8.5, color="gray", style="italic")
ax.text(1990, -0.72, "AI winters & revivals", fontsize=8.5, color="gray", style="italic")
ax.text(2013, -0.72, "Deep Learning era", fontsize=8.5, color="#28a745", style="italic", fontweight="bold")

# Era dividers
ax.axvline(1975, color="lightgray", lw=1, linestyle=":")
ax.axvline(2011, color="lightgray", lw=1, linestyle=":")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "..", "images", "dl_timeline.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {os.path.abspath(output_path)}")
plt.show()
