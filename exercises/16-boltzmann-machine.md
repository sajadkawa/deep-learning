# Exercise 16 — Boltzmann Machine and Restricted Boltzmann Machine

---

## Section A — Multiple Choice

**1.** In a Boltzmann machine, the probability of a state s is given by P(s) = e^(−E(s)/T) / Z. What happens to the probability distribution as T → 0?

- a) All states become equally probable
- b) The distribution concentrates on the lowest-energy state(s)
- c) The partition function Z approaches infinity
- d) The network stops updating its weights

&nbsp;

**2.** A stochastic neuron in a Boltzmann machine has net input netᵢ = 2.0 and temperature T = 2.0. What is P(sᵢ = 1)?

- a) σ(4.0)
- b) σ(2.0)
- c) σ(1.0)
- d) σ(0.5)

&nbsp;

**3.** In contrastive Hebbian learning, the negative phase measures:

- a) The co-activation of visible and hidden units when visible units are clamped to training data
- b) The co-activation of visible and hidden units when the network runs freely
- c) The gradient of the energy function with respect to the weights
- d) The reconstruction error between the input and the visible unit activations after one Gibbs step

&nbsp;

**4.** What property of the RBM makes the positive phase of CD-1 exact (computable in a single parallel step)?

- a) The visible units are binary
- b) The hidden units are conditionally independent given the visible units
- c) The weight matrix is symmetric
- d) The energy function is convex

&nbsp;

**5.** In greedy layer-wise pre-training of a Deep Belief Network, what is used as the input to the second RBM?

- a) The raw training data
- b) The reconstruction of the visible units after one Gibbs step
- c) The hidden activations P(h⁽¹⁾=1 | v) produced by the first trained RBM
- d) The weight matrix of the first RBM

&nbsp;

---

## Section B — Short Answer

**1.** Explain why the full Boltzmann machine is computationally intractable to train, and identify the specific step in contrastive Hebbian learning that causes this problem.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A Boltzmann machine has 3 visible units and 2 hidden units, all biases zero, and all weights equal to 1.0. Without computing all 32 states, identify which state has the lowest energy and explain why.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Explain the role of the partition function Z in the Boltzmann distribution. Why is computing Z exactly intractable for large networks?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** CD-1 is described as an approximation of the true log-likelihood gradient. What exactly is being approximated, and what makes this approximation acceptable in practice?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A small RBM has 2 visible units (v₁, v₂) and 2 hidden units (h₁, h₂). Weights and biases:

```
W = [[0.6, 0.4],   (row i = visible unit i, col j = hidden unit j)
     [0.3, 0.8]]

Visible biases:  a = [0.0, 0.0]
Hidden biases:   b = [0.0, 0.0]
Learning rate:   η = 0.1
```

Training example: v⁽⁰⁾ = (1, 0).

**(a)** Compute P(h₁=1 | v⁽⁰⁾) and P(h₂=1 | v⁽⁰⁾). Use these expectations directly (do not sample).

**(b)** Using h⁽⁰⁾ = (P(h₁=1|v⁽⁰⁾), P(h₂=1|v⁽⁰⁾)), compute P(v₁=1 | h⁽⁰⁾) and P(v₂=1 | h⁽⁰⁾).

**(c)** Using v⁽¹⁾ = (P(v₁=1|h⁽⁰⁾), P(v₂=1|h⁽⁰⁾)), compute P(h₁=1 | v⁽¹⁾) and P(h₂=1 | v⁽¹⁾).

**(d)** Compute the CD-1 weight updates Δw₁₁, Δw₁₂, Δw₂₁, Δw₂₂ and the new weight matrix W after one update.

&nbsp;

---

## Section D — Critical Thinking

**1.** A colleague argues: "Since CD-1 is only an approximation of the true gradient, we should always use CD-10 or CD-100 for better results." Evaluate this argument. Under what conditions might more Gibbs steps help, and under what conditions might they not?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Deep Belief Networks solved the vanishing gradient problem through pre-training rather than by fixing backpropagation. Modern networks solve it through better initialisation, batch normalisation, and ReLU activations — without any pre-training. Does this mean the DBN approach was wrong, or does it reveal something deeper about what makes deep networks trainable? Argue your position.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
