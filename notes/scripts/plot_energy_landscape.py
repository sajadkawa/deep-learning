import os
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 500)

# Energy landscape: two valleys (stable states) with a hill between
energy = 0.5 * (x**2 - 2)**2 - 0.3 * x

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(x, energy, color="#2196F3", linewidth=2)

# Mark the two local minima
minima_x = [x[np.argmin(energy[:250])], x[250 + np.argmin(energy[250:])]]
for mx in minima_x:
    my = 0.5 * (mx**2 - 2)**2 - 0.3 * mx
    ax.plot(mx, my, "o", color="#4CAF50", markersize=8, zorder=5)
    ax.annotate("Low energy\n(stable state)", xy=(mx, my),
                xytext=(mx + 0.3, my + 1.2),
                fontsize=8, color="#4CAF50",
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1))

ax.set_xlabel("Network State")
ax.set_ylabel("Energy")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "..", "images", "energy_landscape.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", os.path.abspath(out))
