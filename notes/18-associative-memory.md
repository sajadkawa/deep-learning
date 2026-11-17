# 18 — Associative Memory

---

## Part 1: Two Kinds of Associative Memory

**Note 15** introduced the Hopfield network as a system that retrieves a stored pattern from a partial or noisy input. That is one specific form of associative memory — but the concept is broader. Before examining the mechanics, it is worth distinguishing the two fundamental types.

**Auto-associative memory** maps a pattern to itself. The input is a corrupted or incomplete version of a stored pattern, and the output is the full, clean version of that same pattern. The Hopfield network is an auto-associative memory: you present a noisy probe, and the network converges to the nearest stored pattern.

**Hetero-associative memory** maps a pattern to a different stored pattern. The input is one pattern, and the output is a paired but distinct pattern. This models associations between different objects — a name and a face, a word and its meaning, a cue and a response.

```
Auto-associative:    x_noisy  →  x_stored     (same pattern, cleaned up)

Hetero-associative:  x_a      →  y_a          (different pattern, paired at storage time)
                     x_b      →  y_b
```

The diagram below shows both types side by side: auto-associative corrects a noisy probe back to the stored pattern; hetero-associative maps an input pattern to its paired output.

![Auto-associative vs hetero-associative memory — two types of pattern association](images/associative_memory_types.png)

Both types use the same underlying mechanism: a weight matrix constructed from stored pattern pairs using the Hebbian outer product rule. The difference is only in what is paired with what at storage time.

> In auto-associative memory, each pattern is paired with itself. In hetero-associative memory, each input pattern is paired with a distinct output pattern. The weight construction is identical — only the pairing changes.

---

## Part 2: The Hebbian Storage Rule

The weight matrix for an associative memory is built from the stored associations using the **Hebbian outer product rule**. The rule encodes co-activation: if two neurons are active together in a stored pattern, their connection is strengthened.

### Auto-Associative Storage

To store $P$ patterns $\mathbf{x}^1, \mathbf{x}^2, \ldots, \mathbf{x}^P$ (each a vector of $\pm 1$ values of length $N$) as auto-associations:

$$W = \frac{1}{N} \sum_m \mathbf{x}^m (\mathbf{x}^m)^\top$$

with $w_{ii} = 0$ (zero diagonal enforced after construction)

Each term $\mathbf{x}^m(\mathbf{x}^m)^\top$ is the $N \times N$ outer product of pattern $m$ with itself. Entry $(i, j)$ of this matrix is $x_i^m \times x_j^m$ — positive when neurons $i$ and $j$ have the same sign in pattern $m$, negative when they differ. Summing over all patterns accumulates these co-activation statistics across the full set of stored patterns.

The factor $1/N$ normalises by network size so that the signal-to-noise ratio (Part 4) scales predictably with $N$.

### Worked Example — Auto-Associative Storage

Store two patterns of length N = 4:

```
x¹ = (+1, +1, −1, +1)
x² = (−1, +1, +1, +1)
```

Note that $\mathbf{x}^1 \cdot \mathbf{x}^2 = (1)(-1) + (1)(1) + (-1)(1) + (1)(1) = 0$. The patterns are orthogonal — this minimises crosstalk (Part 4).

**Step 1 — Outer product for x¹:**

```
x¹(x¹)ᵀ =  [ +1  +1  −1  +1 ]
            [ +1  +1  −1  +1 ]
            [ −1  −1  +1  −1 ]
            [ +1  +1  −1  +1 ]
```

**Step 2 — Outer product for x²:**

```
x²(x²)ᵀ =  [ +1  −1  −1  −1 ]
            [ −1  +1  +1  +1 ]
            [ −1  +1  +1  +1 ]
            [ −1  +1  +1  +1 ]
```

**Step 3 — Sum and scale by 1/N = 1/4:**

```
x¹(x¹)ᵀ + x²(x²)ᵀ =  [ +2   0  −2   0 ]
                       [  0  +2   0  +2 ]
                       [ −2   0  +2   0 ]
                       [  0  +2   0  +2 ]

W = (1/4) × above =   [ 0.5   0   −0.5   0  ]
                       [  0   0.5    0   0.5 ]
                       [−0.5   0    0.5   0  ]
                       [  0   0.5    0   0.5 ]
```

**Step 4 — Zero the diagonal:**

The diagonal of each outer product xᵐ(xᵐ)ᵀ is always all-ones (since (±1)² = 1). After summing and scaling, the diagonal is 0.5, 0.5, 0.5, 0.5. Set these to zero:

```
W =  [  0     0   −0.5   0  ]
     [  0     0     0   0.5 ]
     [ −0.5   0     0    0  ]
     [  0    0.5    0    0  ]
```

This is the weight matrix that stores both patterns as stable attractors.

---

## Part 3: Retrieval Dynamics

Once the weight matrix is constructed, retrieval proceeds by the same asynchronous update rule used in the Hopfield network (**Note 15**, Part 2):

$$\text{net}_i = \sum_j w_{ij} s_j$$

$$s_i \leftarrow \begin{cases} +1 & \text{if } \text{net}_i \geq 0 \\ -1 & \text{if } \text{net}_i < 0 \end{cases}$$

Neurons update one at a time. Each update either decreases the energy or leaves it unchanged. The network converges to a local energy minimum — a stored pattern (or a spurious state, discussed in Part 4).

### Why Stored Patterns Are Fixed Points

A stored pattern $\mathbf{x}^m$ is a fixed point of the update rule if, for every neuron $i$:

$$\text{sign}(\text{net}_i) = x_i^m$$

Substituting the weight matrix $W = (1/N) \sum_k \mathbf{x}^k(\mathbf{x}^k)^\top$ (diagonal zeroed):

$$\text{net}_i = \sum_j w_{ij} x_j^m = \frac{1}{N} \sum_j \sum_k x_i^k x_j^k x_j^m \quad (j \neq i)$$

$$= \frac{1}{N} \sum_k x_i^k \left( \sum_j x_j^k x_j^m \right) = \frac{1}{N} \sum_k x_i^k (\mathbf{x}^k \cdot \mathbf{x}^m)$$

For the target pattern $k = m$, the dot product $\mathbf{x}^m \cdot \mathbf{x}^m = N$ (since each component is $\pm 1$). This gives a **signal term**:

$$\text{Signal:} \quad \frac{1}{N} \times x_i^m \times N = x_i^m$$

For all other patterns $k \neq m$, the dot products $\mathbf{x}^k \cdot \mathbf{x}^m$ are the **crosstalk terms** — interference from the other stored patterns. If the patterns are orthogonal, these terms are zero and retrieval is exact. In general, they are small but nonzero.

> The signal term always points in the direction of the stored pattern. The crosstalk terms are noise from the other stored patterns. Retrieval succeeds when the signal dominates the crosstalk — which holds when P is small relative to N.

### Worked Example — Retrieval

Using the weight matrix from Part 2, retrieve x¹ = (+1, +1, −1, +1) from a noisy probe with s₁ flipped:

```
probe = (−1, +1, −1, +1)   ← x¹ with s₁ flipped from +1 to −1
```

Asynchronous update, neurons in order 1 → 4:

```
Neuron 1:  net₁ = w₁₂s₂ + w₁₃s₃ + w₁₄s₄
                = (0)(+1) + (−0.5)(−1) + (0)(+1)
                = 0.5 > 0   →   s₁ → +1

Neuron 2:  net₂ = w₂₁s₁ + w₂₃s₃ + w₂₄s₄
                = (0)(+1) + (0)(−1) + (0.5)(+1)
                = 0.5 > 0   →   s₂ stays +1

Neuron 3:  net₃ = w₃₁s₁ + w₃₂s₂ + w₃₄s₄
                = (−0.5)(+1) + (0)(+1) + (0)(+1)
                = −0.5 < 0  →   s₃ stays −1

Neuron 4:  net₄ = w₄₁s₁ + w₄₂s₂ + w₄₃s₃
                = (0)(+1) + (0.5)(+1) + (0)(−1)
                = 0.5 > 0   →   s₄ stays +1
```

After one pass: s = (+1, +1, −1, +1) = x¹ ✓

The network recovered the stored pattern in a single update pass.

---

## Part 4: Performance Measures

Three quantities characterise how well an associative memory performs: storage capacity, error correction radius, and crosstalk noise.

### Storage Capacity

The maximum number of patterns that can be stored and reliably retrieved is approximately:

$$P_{\max} \approx 0.14 \times N$$

This is the same capacity limit derived for the Hopfield network in **Note 15**. It arises from the crosstalk terms in the net input. When $P$ is small, the crosstalk is weak and the signal dominates. As $P$ grows, the crosstalk accumulates and eventually overwhelms the signal, causing retrieval to fail.

The diagram below shows retrieval success rate as a function of P/N for a network of N = 100 neurons. The transition near P/N = 0.14 is sharp — below it, retrieval is reliable; above it, it collapses.

![Retrieval success rate vs P/N ratio — the capacity cliff at P/N ≈ 0.14](images/associative_memory_capacity.png)

### Error Correction Radius

The error correction radius is the maximum number of bit errors in the probe that the network can still correct. For a network storing $P$ patterns with $N$ neurons, the radius is approximately:

$$r \approx \frac{N}{2} \left(1 - \frac{P}{P_{\max}}\right)$$

When $P$ is well below capacity ($P \ll P_{\max}$), the radius approaches $N/2$ — the network can correct nearly half the bits. As $P$ approaches $P_{\max}$, the radius shrinks to zero.

### Crosstalk Noise

The signal-to-noise ratio for retrieving pattern $\mathbf{x}^m$ is:

$$\text{SNR} = \frac{\text{signal}}{\text{crosstalk}} = \frac{N}{\sqrt{P-1}}$$

This follows from the signal term being $x_i^m$ (magnitude 1, scaled by $N/N = 1$) and the crosstalk terms being a sum of $P-1$ random $\pm 1$ contributions, each scaled by $1/N$. By the central limit theorem, the crosstalk has standard deviation $\sqrt{P-1}/N$, giving $\text{SNR} = N/\sqrt{P-1}$.

- $P = 1$: $\text{SNR} = \infty$ (no crosstalk, perfect retrieval)
- $P = 14$: $\text{SNR} \approx N/\sqrt{13} \approx 0.27N$
- $P = 0.14N$: $\text{SNR} \approx \sqrt{N/0.14} \approx 2.7$ for $N = 100$ (marginal)

> The capacity limit $P_{\max} \approx 0.14N$ is the point where the SNR drops below the threshold for reliable retrieval. It is not a design choice — it is a consequence of the Hebbian weight construction.

---

## Part 5: Hetero-Associative Memory

In hetero-associative memory, each input pattern $\mathbf{x}^m$ is paired with a distinct output pattern $\mathbf{y}^m$. The weight matrix is constructed from the cross-outer products of output with input:

$$W = \frac{1}{N} \sum_m \mathbf{y}^m (\mathbf{x}^m)^\top$$

Entry $(i, j)$ of this matrix is $\frac{1}{N} \sum_m y_i^m x_j^m$ — the co-activation of output neuron $i$ and input neuron $j$ across all stored pairs.

Retrieval is a single forward pass — not an iterative update. Given input $\mathbf{x}$, the output is:

$$\mathbf{y} = \text{sign}(W \mathbf{x})$$

There is no energy function and no convergence dynamics. The hetero-associative memory is a one-shot lookup: present the input, compute the weighted sum, threshold.

### Worked Example — Hetero-Associative Storage and Retrieval

Store two input-output pairs with N = 4:

```
xᵃ = (+1, +1, −1, +1)   →   yᵃ = (+1, −1, +1, −1)
xᵇ = (−1, +1, +1, +1)   →   yᵇ = (−1, −1, +1, +1)
```

**Step 1 — Outer products yᵐ(xᵐ)ᵀ:**

```
yᵃ(xᵃ)ᵀ =  [ +1  +1  −1  +1 ]
            [ −1  −1  +1  −1 ]
            [ +1  +1  −1  +1 ]
            [ −1  −1  +1  −1 ]

yᵇ(xᵇ)ᵀ =  [ +1  −1  −1  −1 ]
            [ +1  −1  −1  −1 ]
            [ −1  +1  +1  +1 ]
            [ −1  +1  +1  +1 ]
```

**Step 2 — Sum and scale by 1/4:**

```
W = (1/4) × [ +2   0  −2   0 ]   =  [ 0.5   0   −0.5   0  ]
             [  0  −2   0  −2 ]      [  0  −0.5    0  −0.5 ]
             [  0  +2   0  +2 ]      [  0   0.5    0   0.5 ]
             [ −2   0  +2   0 ]      [−0.5   0    0.5   0  ]
```

Note: the diagonal is not zeroed for hetero-associative memory — the input and output are different vectors, so self-connections are not a concern.

**Step 3 — Retrieve yᵃ from xᵃ:**

```
net = W xᵃ = [ (0.5)(+1) + (0)(+1) + (−0.5)(−1) + (0)(+1)   ]   =  [ +1.0 ]
             [ (0)(+1) + (−0.5)(+1) + (0)(−1) + (−0.5)(+1)   ]      [ −1.0 ]
             [ (0)(+1) + (0.5)(+1) + (0)(−1) + (0.5)(+1)     ]      [ +1.0 ]
             [ (−0.5)(+1) + (0)(+1) + (0.5)(−1) + (0)(+1)    ]      [ −1.0 ]

y = sign(net) = (+1, −1, +1, −1) = yᵃ  ✓
```

**Step 4 — Retrieve yᵇ from xᵇ:**

```
net = W xᵇ = (−1.0, −1.0, +1.0, +1.0)

y = sign(net) = (−1, −1, +1, +1) = yᵇ  ✓
```

Both pairs retrieved exactly.

---

## Part 6: Relationship to Hopfield Networks and Modern Attention

### Relationship to Hopfield Networks

The Hopfield network (**Note 15**) is a special case of auto-associative memory. The differences are:

```
Hopfield network:          Auto-associative memory (general):
  Symmetric weights            Symmetric weights (same)
  No self-connections          No self-connections (same)
  Asynchronous update          Asynchronous update (same)
  Energy function              Energy function (same)
  Binary states {−1, +1}       Binary states {−1, +1} (same)
  Hebbian outer product rule   Hebbian outer product rule (same)
```

They are the same model. "Hopfield network" refers to the specific architecture and energy analysis; "auto-associative memory" refers to the functional role it plays. The capacity limit $0.14N$, the spurious states, and the convergence guarantee all carry over directly.

The hetero-associative memory is a generalisation: it relaxes the constraint that input and output must be the same pattern. The weight matrix is no longer symmetric ($W \neq W^\top$ in general), so there is no energy function and no convergence dynamics — retrieval is a single matrix-vector product.

### Relationship to Modern Attention

The attention mechanism in transformers (**Note 27**) can be understood as a continuous, differentiable generalisation of hetero-associative memory.

In hetero-associative memory:

$$\mathbf{y} = \text{sign}(W \mathbf{x}) \quad \text{where} \quad W = \frac{1}{N} \sum_m \mathbf{y}^m (\mathbf{x}^m)^\top$$

Substituting $W$:

$$\mathbf{y} = \text{sign}\!\left( \frac{1}{N} \sum_m \mathbf{y}^m (\mathbf{x}^m \cdot \mathbf{x}) \right)$$

The term $\mathbf{x}^m \cdot \mathbf{x}$ is the dot product between the stored input pattern $\mathbf{x}^m$ and the query $\mathbf{x}$ — a similarity score. The output is a weighted combination of stored output patterns $\mathbf{y}^m$, where the weights are the similarity scores.

Scaled dot-product attention does exactly this, but continuously:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

Here $Q$ is the query, $K$ are the keys (stored input patterns), $V$ are the values (stored output patterns), and the softmax replaces the hard sign threshold with a soft weighted sum. The division by $\sqrt{d}$ is a normalisation analogous to the $1/N$ factor.

```
Associative memory:   hard threshold,  discrete retrieval,  exact match
Attention:            soft threshold,  continuous retrieval, weighted blend
```

> Attention is a soft, differentiable associative memory. The keys are the stored input patterns, the values are the stored output patterns, and the query is the probe. The softmax replaces the sign function, making the lookup differentiable and trainable end-to-end.

This connection was made explicit by Ramsauer et al. (2020), who showed that modern Hopfield networks with continuous states and exponential energy functions are equivalent to the attention mechanism.

---

## Summary

Key takeaways:

- Auto-associative memory maps a noisy or partial input to the same stored pattern; hetero-associative memory maps an input pattern to a paired but different output pattern
- Both use the Hebbian outer product rule: auto-associative stores $W = (1/N) \sum_m \mathbf{x}^m(\mathbf{x}^m)^\top$; hetero-associative stores $W = (1/N) \sum_m \mathbf{y}^m(\mathbf{x}^m)^\top$
- Auto-associative retrieval is iterative — asynchronous updates converge to a stored attractor via energy descent; hetero-associative retrieval is a single forward pass $\mathbf{y} = \text{sign}(W\mathbf{x})$
- The signal term in the net input is always $x_i^m$ (pointing toward the stored pattern); the crosstalk terms are interference from other stored patterns
- Storage capacity is approximately $0.14N$ patterns — the same limit as the Hopfield network — arising from the SNR dropping below the reliable retrieval threshold
- The error correction radius shrinks as P approaches capacity; orthogonal patterns minimise crosstalk and maximise capacity
- The Hopfield network is a special case of auto-associative memory — same architecture, same rule, same energy function
- Scaled dot-product attention is a continuous, differentiable generalisation of hetero-associative memory: keys are stored input patterns, values are stored output patterns, the query is the probe, and softmax replaces the sign threshold
