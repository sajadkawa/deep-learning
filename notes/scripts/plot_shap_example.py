import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

features = ["Age", "Loan amount", "Income", "Missed payments"]
values   = [-0.10, +0.31, -0.42, +0.85]
colors   = ['#dcfce7' if v < 0 else '#fee2e2' for v in values]
edges    = ['#15803d' if v < 0 else '#b91c1c' for v in values]

fig, ax = plt.subplots(figsize=(6, 3.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

fig.suptitle("SHAP Feature Contributions — Loan Default Prediction",
             fontsize=12, fontweight="bold", color='#111111')

bars = ax.barh(features, values, color=colors, edgecolor=edges,
               linewidth=1.5, height=0.5)

for bar, val in zip(bars, values):
    offset = 0.03 if val >= 0 else -0.03
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", ha=ha,
            fontsize=10, fontweight="bold", color='#111111')

ax.axvline(0, color='#374151', linewidth=1)
ax.set_xlabel("SHAP value (contribution to prediction)",
              fontsize=10, color='#374151')
ax.set_xlim(-0.7, 1.1)
ax.tick_params(colors='#374151', labelsize=9)

low_risk  = mpatches.Patch(facecolor='#dcfce7', edgecolor='#15803d',
                            label='Pushes toward LOW RISK')
high_risk = mpatches.Patch(facecolor='#fee2e2', edgecolor='#b91c1c',
                            label='Pushes toward HIGH RISK')
ax.legend(handles=[low_risk, high_risk], loc="lower right", fontsize=9,
          facecolor='white', edgecolor='#d1d5db')

ax.text(0.01, -0.22, "Prediction: HIGH RISK  (sum of contributions > 0)",
        transform=ax.transAxes, fontsize=9, color='#6b7280', style="italic")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_edgecolor('#d1d5db')
ax.spines["bottom"].set_edgecolor('#d1d5db')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "..", "images", "shap_example.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor='white', edgecolor='none')
plt.close()
print("Saved shap_example.png")
