import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

np.random.seed(42)

def simulate_capacity(N, P_values, n_trials=200):
    success_rates = []
    for P in P_values:
        successes = 0
        for _ in range(n_trials):
            # Generate P random bipolar patterns
            patterns = np.sign(np.random.randn(P, N))
            patterns[patterns == 0] = 1
            # Build weight matrix
            W = np.zeros((N, N))
            for m in range(P):
                W += np.outer(patterns[m], patterns[m])
            W /= N
            np.fill_diagonal(W, 0)
            # Test retrieval of first pattern with 1 bit flipped
            probe = patterns[0].copy()
            flip = np.random.randint(N)
            probe[flip] *= -1
            # Asynchronous update until stable
            s = probe.copy()
            for _ in range(50):
                changed = False
                order = np.random.permutation(N)
                for i in order:
                    net_i = W[i] @ s
                    new_si = 1.0 if net_i >= 0 else -1.0
                    if new_si != s[i]:
                        s[i] = new_si
                        changed = True
                if not changed:
                    break
            if np.array_equal(s, patterns[0]):
                successes += 1
        success_rates.append(successes / n_trials)
    return success_rates

N = 100
P_values = list(range(1, 31))
pn_ratios = [p / N for p in P_values]
success_rates = simulate_capacity(N, P_values)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#f8f8f8')
ax.set_facecolor('#f8f8f8')

ax.plot(pn_ratios, success_rates, color='#2c7bb6', linewidth=2.5, marker='o', markersize=5, label='Retrieval success rate')

# Shade reliable region
ax.axvspan(0, 0.138, alpha=0.12, color='#1a9641', label='Reliable region (P/N < 0.14)')
ax.axvspan(0.138, max(pn_ratios), alpha=0.10, color='#d7191c', label='Degraded region (P/N > 0.14)')

# Capacity line
ax.axvline(x=0.138, color='#d7191c', linewidth=1.8, linestyle='--', label='Capacity limit P/N = 0.14')

ax.set_xlabel('P / N  (patterns stored per neuron)', fontsize=11)
ax.set_ylabel('Retrieval success rate', fontsize=11)
ax.set_title('Hopfield Network Storage Capacity\n(N = 100 neurons, 200 trials per point)', fontsize=11, fontweight='bold')
ax.set_xlim(0, max(pn_ratios))
ax.set_ylim(-0.05, 1.1)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

for spine in ax.spines.values():
    spine.set_edgecolor('#cccccc')

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
plt.savefig(os.path.join(out_dir, 'associative_memory_capacity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved associative_memory_capacity.png")
