import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

z = np.linspace(-6, 6, 400)

def sigmoid(z):       return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): s = sigmoid(z); return s * (1 - s)
def tanh_deriv(z):    return 1 - np.tanh(z) ** 2
def relu(z):          return np.maximum(0, z)
def relu_deriv(z):    return (z > 0).astype(float)
def leaky_relu(z, alpha=0.1):       return np.where(z > 0, z, alpha * z)
def leaky_relu_deriv(z, alpha=0.1): return np.where(z > 0, 1.0, alpha)

configs = [
    ("Sigmoid",    sigmoid(z),      sigmoid_deriv(z),    '#2563eb', (-0.1, 1.1),  (-0.05, 0.30)),
    ("Tanh",       np.tanh(z),      tanh_deriv(z),       '#15803d', (-1.1, 1.1),  (-0.05, 1.10)),
    ("ReLU",       relu(z),         relu_deriv(z),       '#b45309', (-0.5, 6.5),  (-0.1,  1.2)),
    ("Leaky ReLU", leaky_relu(z),   leaky_relu_deriv(z), '#7c3aed', (-0.6, 6.5),  (-0.05, 1.2)),
]

fig = plt.figure(figsize=(11, 12))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.35)

for row, (name, f, df, color, ylim_f, ylim_df) in enumerate(configs):
    for col, (data, ylim, ylabel, title_suffix) in enumerate([
        (f,  ylim_f,  "f(z)",  "f(z)"),
        (df, ylim_df, "f'(z)", "f'(z)"),
    ]):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('white')
        ls = '--' if col == 1 else '-'
        ax.plot(z, data, color=color, linewidth=2, linestyle=ls)
        ax.axhline(0, color='#e5e7eb', linewidth=0.8)
        ax.axvline(0, color='#e5e7eb', linewidth=0.8)
        ax.set_title(f"{name}  —  {title_suffix}", color='#111111', fontsize=11, pad=6)
        ax.set_xlabel("z", color='#374151', fontsize=9)
        ax.set_ylabel(ylabel, color='#374151', fontsize=9)
        ax.set_xlim(-6, 6)
        ax.set_ylim(*ylim)
        ax.tick_params(colors='#374151', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#d1d5db')

fig.suptitle("Activation Functions and Their Derivatives",
             color='#111111', fontsize=13, y=0.995)

out = os.path.join(os.path.dirname(__file__), '..', 'images', 'activation_functions.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved activation_functions.png")
