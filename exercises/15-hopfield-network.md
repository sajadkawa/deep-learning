# Exercise 15 — Hopfield Network

---

## Section A — Multiple Choice

**1.** A Hopfield network with N = 200 neurons uses the Hebbian outer product rule. What is the maximum number of patterns that can be stored and reliably retrieved?

- a) 200
- b) 100
- c) 28
- d) 14

---

**2.** Which property of the weight matrix is required for the energy function to guarantee convergence under asynchronous update?

- a) All weights must be positive
- b) The weight matrix must be symmetric
- c) The weight matrix must be orthogonal
- d) All diagonal entries must equal 1

---

**3.** A Hopfield network is initialised with a state that is a noisy version of a stored pattern. Under asynchronous update, which of the following is guaranteed?

- a) The network converges to the stored pattern
- b) The network converges to some local energy minimum
- c) The network converges to the global energy minimum
- d) The network converges within N update steps

---

**4.** The reversed pattern −x¹ is always a spurious attractor in a Hopfield network. What property of the energy function explains this?

- a) The energy function is convex
- b) The energy function is symmetric under s → −s
- c) The weight matrix has zero trace
- d) The Hebbian rule stores both x¹ and −x¹ explicitly

---

**5.** A Hopfield network stores P patterns using the Hebbian outer product rule. As P increases beyond 0.14N, what happens to the energy landscape?

- a) The stored patterns become deeper energy minima
- b) The network begins to ignore noisy inputs
- c) Crosstalk between patterns destroys the clean energy minima
- d) The network switches to synchronous update automatically

---

## Section B — Short Answer

**1.** Explain why the weight matrix in a Hopfield network must have zeros on the diagonal. What would happen to the energy function if self-connections were allowed?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**2.** The Hebbian outer product rule stores patterns as W = (1/N) Σₘ xᵐ(xᵐ)ᵀ. Explain in terms of co-activation statistics why this rule makes stored patterns into energy minima.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**3.** Synchronous update can produce 2-cycles even with symmetric weights. Describe what a 2-cycle is and explain why it does not violate the energy argument from Part 3 of the note.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**4.** A Hopfield network is used to store images of handwritten digits (0–9). A user reports that when they present a noisy version of the digit "3", the network sometimes converges to the digit "8" instead. Give two distinct explanations for why this might happen.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

Store two patterns of length N = 5 using the Hebbian outer product rule:

```
x¹ = (+1, +1, −1, +1, −1)
x² = (−1, +1, +1, +1, −1)
```

**(a)** Compute the outer product x¹(x¹)ᵀ. Show the full 5×5 matrix.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute the outer product x²(x²)ᵀ. Show the full 5×5 matrix.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Compute the weight matrix W = (1/5)(x¹(x¹)ᵀ + x²(x²)ᵀ) and zero the diagonal. Show the final W.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** Start from the following noisy input (one bit of x¹ is flipped — neuron 3 is wrong):

```
s = (+1, +1, +1, +1, −1)
```

Update neuron 3 asynchronously. Compute net₃ = Σⱼ w₃ⱼ sⱼ (j ≠ 3) using your W from part (c). What does s₃ become? Does the network recover x¹?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** The Hopfield network stores patterns as energy minima using the Hebbian rule. A student proposes an alternative: instead of the outer product rule, train the weights using gradient descent to minimise a loss that penalises the network for not converging to the stored patterns. What would be the advantage of this approach over the Hebbian rule? What would be lost?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**2.** The storage capacity of a Hopfield network is approximately 0.14N. A researcher wants to store 1000 patterns reliably. They propose building a single Hopfield network with N = 10,000 neurons. A colleague suggests instead using 10 separate Hopfield networks of N = 1,000 neurons each, with a routing mechanism that selects which network to query. Compare these two approaches in terms of storage capacity, spurious states, and retrieval reliability.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
