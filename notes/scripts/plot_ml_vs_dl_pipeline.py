import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.suptitle("Traditional ML vs Deep Learning — Pipeline Comparison", fontsize=13, fontweight="bold")

def draw_box(ax, text, y, color, width=0.6, height=0.07):
    x = 0.5
    box = mpatches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                   boxstyle="round,pad=0.01",
                                   facecolor=color, edgecolor="gray", linewidth=1)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, fontweight="bold")

def draw_arrow(ax, y_start, y_end):
    ax.annotate("", xy=(0.5, y_end + 0.035), xytext=(0.5, y_start - 0.035),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

# ── Traditional ML ────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Traditional ML", fontsize=12, pad=10)

ml_steps = [
    (0.88, "Raw Data",                   "#f0f0f0"),
    (0.74, "Human understands problem",  "#fff3cd"),
    (0.60, "Human designs FEATURES",     "#ffe0b2"),
    (0.46, "Engineered Features",        "#fff3cd"),
    (0.32, "ML Algorithm",               "#d1ecf1"),
    (0.18, "Prediction",                 "#d4edda"),
]

for i, (y, label, color) in enumerate(ml_steps):
    draw_box(ax, label, y, color)
    if i < len(ml_steps) - 1:
        draw_arrow(ax, y, ml_steps[i+1][0])

ax.text(0.5, 0.04, "Human does the representation work", ha="center",
        fontsize=8.5, color="gray", style="italic")

# Brace for human work
ax.annotate("", xy=(0.88, 0.57), xytext=(0.88, 0.77),
            arrowprops=dict(arrowstyle="-", color="tomato", lw=2))
ax.text(0.92, 0.67, "Human\nwork", fontsize=8, color="tomato", va="center")

# ── Deep Learning ─────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Deep Learning", fontsize=12, pad=10)

dl_steps = [
    (0.88, "Raw Data",                        "#f0f0f0"),
    (0.74, "Layer 1 — low-level features",    "#d1ecf1"),
    (0.60, "Layer 2 — mid-level features",    "#c3e6f5"),
    (0.46, "Layer 3 — high-level features",   "#b3d9f7"),
    (0.32, "Learned Representation",          "#9ecbf5"),
    (0.18, "Prediction",                      "#d4edda"),
]

for i, (y, label, color) in enumerate(dl_steps):
    draw_box(ax, label, y, color)
    if i < len(dl_steps) - 1:
        draw_arrow(ax, y, dl_steps[i+1][0])

ax.text(0.5, 0.04, "Model learns the representation", ha="center",
        fontsize=8.5, color="gray", style="italic")

# Brace for model learning
ax.annotate("", xy=(0.88, 0.29), xytext=(0.88, 0.77),
            arrowprops=dict(arrowstyle="-", color="steelblue", lw=2))
ax.text(0.92, 0.53, "Model\nlearns", fontsize=8, color="steelblue", va="center")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "..", "images", "ml_vs_dl_pipeline.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {os.path.abspath(output_path)}")
plt.show()
