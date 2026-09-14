import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

features = ["Age", "Loan amount", "Income", "Missed payments"]
values   = [-0.10, +0.31, -0.42, +0.85]
colors   = ["#d4edda" if v < 0 else "#f8d7da" for v in values]
edge     = ["#28a745" if v < 0 else "#dc3545" for v in values]

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle("SHAP Feature Contributions — Loan Default Prediction", fontsize=12, fontweight="bold")

bars = ax.barh(features, values, color=colors, edgecolor=edge, linewidth=1.2, height=0.5)

for bar, val in zip(bars, values):
    offset = 0.03 if val >= 0 else -0.03
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", ha=ha, fontsize=10, fontweight="bold")

ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("SHAP value (contribution to prediction)", fontsize=10)
ax.set_xlim(-0.7, 1.1)

low_risk  = mpatches.Patch(facecolor="#d4edda", edgecolor="#28a745", label="Pushes toward LOW RISK")
high_risk = mpatches.Patch(facecolor="#f8d7da", edgecolor="#dc3545", label="Pushes toward HIGH RISK")
ax.legend(handles=[low_risk, high_risk], loc="lower right", fontsize=9)

ax.text(0.01, -0.18, "Prediction: HIGH RISK  (sum of contributions > 0)",
        transform=ax.transAxes, fontsize=9, color="gray", style="italic")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "..", "images", "shap_example.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {os.path.abspath(output_path)}")
plt.show()
