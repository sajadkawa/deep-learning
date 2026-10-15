import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor('white')
fig.suptitle("Traditional ML vs Deep Learning — Pipeline Comparison",
             fontsize=13, fontweight="bold", color='#111111')

def draw_box(ax, text, y, fc, ec, width=0.6, height=0.07):
    x = 0.5
    box = mpatches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                   boxstyle="round,pad=0.01",
                                   facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color='#111111')

def draw_arrow(ax, y_start, y_end):
    ax.annotate("", xy=(0.5, y_end + 0.035), xytext=(0.5, y_start - 0.035),
                arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.5))

# ── Traditional ML ────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('white')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Traditional ML", fontsize=12, fontweight='bold', color='#111111', pad=10)

ml_steps = [
    (0.88, "Raw Data",                  '#f3f4f6', '#6b7280'),
    (0.74, "Human understands problem", '#fef9c3', '#ca8a04'),
    (0.60, "Human designs FEATURES",    '#ffedd5', '#c2410c'),
    (0.46, "Engineered Features",       '#fef9c3', '#ca8a04'),
    (0.32, "ML Algorithm",              '#dbeafe', '#1d4ed8'),
    (0.18, "Prediction",                '#dcfce7', '#15803d'),
]

for i, (y, label, fc, ec) in enumerate(ml_steps):
    draw_box(ax, label, y, fc, ec)
    if i < len(ml_steps) - 1:
        draw_arrow(ax, y, ml_steps[i+1][0])

ax.text(0.5, 0.04, "Human does the representation work", ha="center",
        fontsize=8.5, color="#6b7280", style="italic")

ax.annotate("", xy=(0.88, 0.57), xytext=(0.88, 0.77),
            arrowprops=dict(arrowstyle="-", color="#c2410c", lw=2))
ax.text(0.92, 0.67, "Human\nwork", fontsize=8, color="#c2410c",
        va="center", fontweight='bold')

# ── Deep Learning ─────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('white')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Deep Learning", fontsize=12, fontweight='bold', color='#111111', pad=10)

dl_steps = [
    (0.88, "Raw Data",                       '#f3f4f6', '#6b7280'),
    (0.74, "Layer 1 — low-level features",   '#dbeafe', '#1d4ed8'),
    (0.60, "Layer 2 — mid-level features",   '#bfdbfe', '#1d4ed8'),
    (0.46, "Layer 3 — high-level features",  '#93c5fd', '#1d4ed8'),
    (0.32, "Learned Representation",         '#60a5fa', '#1d4ed8'),
    (0.18, "Prediction",                     '#dcfce7', '#15803d'),
]

for i, (y, label, fc, ec) in enumerate(dl_steps):
    draw_box(ax, label, y, fc, ec)
    if i < len(dl_steps) - 1:
        draw_arrow(ax, y, dl_steps[i+1][0])

ax.text(0.5, 0.04, "Model learns the representation", ha="center",
        fontsize=8.5, color="#6b7280", style="italic")

ax.annotate("", xy=(0.88, 0.29), xytext=(0.88, 0.77),
            arrowprops=dict(arrowstyle="-", color="#1d4ed8", lw=2))
ax.text(0.92, 0.53, "Model\nlearns", fontsize=8, color="#1d4ed8",
        va="center", fontweight='bold')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "..", "images", "ml_vs_dl_pipeline.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white', edgecolor='none')
plt.close()
print("Saved ml_vs_dl_pipeline.png")
