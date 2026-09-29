import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

z = np.linspace(-6, 6, 400)

# --- functions ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_deriv(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

# --- layout: 4 rows x 2 cols ---
fig = plt.figure(figsize=(12, 14))
fig.patch.set_facecolor('#1e1e1e')
gs = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.35)

configs = [
    ("Sigmoid",      sigmoid(z),           sigmoid_deriv(z),    "#4fc3f7", (-0.1, 1.1),  (-0.05, 0.30)),
    ("Tanh",         np.tanh(z),           tanh_deriv(z),       "#81c784", (-1.1, 1.1),  (-0.05, 1.10)),
    ("ReLU",         relu(z),              relu_deriv(z),       "#ffb74d", (-0.5, 6.5),  (-0.1,  1.2)),
    ("Leaky ReLU",   leaky_relu(z),        leaky_relu_deriv(z), "#ce93d8", (-6.5, 6.5),  (-0.1,  1.2)),
]

for row, (name, f, df, color, ylim_f, ylim_df) in enumerate(configs):
    # function
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.set_facecolor('#2d2d2d')
    ax1.plot(z, f, color=color, linewidth=2)
    ax1.axhline(0, color='#666', linewidth=0.8, linestyle='--')
    ax1.axvline(0, color='#666', linewidth=0.8, linestyle='--')
    ax1.set_title(f"{name}  —  f(z)", color='white', fontsize=11, pad=6)
    ax1.set_xlabel("z", color='#aaa', fontsize=9)
    ax1.set_ylabel("f(z)", color='#aaa', fontsize=9)
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(*ylim_f)
    ax1.tick_params(colors='#aaa', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor('#555')

    # derivative
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.set_facecolor('#2d2d2d')
    ax2.plot(z, df, color=color, linewidth=2, linestyle='--')
    ax2.axhline(0, color='#666', linewidth=0.8, linestyle='--')
    ax2.axvline(0, color='#666', linewidth=0.8, linestyle='--')
    ax2.set_title(f"{name}  —  f'(z)", color='white', fontsize=11, pad=6)
    ax2.set_xlabel("z", color='#aaa', fontsize=9)
    ax2.set_ylabel("f'(z)", color='#aaa', fontsize=9)
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(*ylim_df)
    ax2.tick_params(colors='#aaa', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555')

fig.suptitle("Activation Functions and Their Derivatives", color='white', fontsize=13, y=0.995)

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'activation_functions.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Saved activation_functions.png")
