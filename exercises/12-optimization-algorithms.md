# Exercise 12 — Optimization Algorithms: SGD, Momentum, RMSProp, Adam

---

## Section A — Multiple Choice

**1.** Why does vanilla gradient descent oscillate in a ravine?

- a) The learning rate is always too large for deep networks
- b) The same learning rate governs both the steep and flat directions — a rate that works for one causes problems in the other
- c) Gradient descent cannot handle non-convex loss surfaces
- d) The gradient is zero in a ravine, so no progress is made

&nbsp;

**2.** In momentum, what happens to gradients that alternate in sign across steps?

- a) They accumulate and produce large velocity in that direction
- b) They cancel out in the exponential moving average, keeping velocity small
- c) They are ignored and replaced by the previous gradient
- d) They cause the learning rate to decrease automatically

&nbsp;

**3.** RMSProp divides the gradient by √sₜ where sₜ is the exponential moving average of squared gradients. What is the effect on a parameter that has been receiving consistently large gradients?

- a) It receives a larger effective learning rate to compensate
- b) It receives a smaller effective learning rate to prevent overshooting
- c) Its gradient is set to zero to prevent instability
- d) Its learning rate is increased by β each step

&nbsp;

**4.** Adam uses bias correction on its moment estimates. At step t = 1 with β₁ = 0.9, what is the bias correction factor applied to m₁?

- a) 0.9
- b) 0.1
- c) 10
- d) 1

&nbsp;

**5.** Which optimizer is generally preferred for RNNs and why?

- a) SGD with momentum — because RNNs have smooth loss surfaces
- b) Adam — because it requires no learning rate tuning
- c) RMSProp — because it handles non-stationary gradients well
- d) Vanilla gradient descent — because RNNs are trained on small datasets

&nbsp;

---

## Section B — Short Answer

**1.** Explain why saddle points are more problematic for vanilla gradient descent in high-dimensional spaces than local minima.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Adam's bias correction has no effect after the first few steps. Explain why — what happens to the correction factor (1 − β₁ᵗ) as t increases?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Cosine annealing is generally preferred over step decay for learning rate scheduling. What property of cosine annealing makes it more stable, and what does step decay do that can cause instability?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** The note states that AdamW should be used instead of Adam when L2 regularization is needed. Why does standard L2 regularization interact poorly with Adam's adaptive learning rates?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A single parameter w is being optimized. Starting values: w₀ = 2.0. The gradients at each step are:

```
g₁ = 0.4,   g₂ = 0.6,   g₃ = 0.5
```

Use η = 0.1 throughout. Use ε = 0 unless stated.

**(a)** Trace 3 steps of **SGD with momentum** (β = 0.9, v₀ = 0). Compute v₁, w₁, v₂, w₂, v₃, w₃.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Trace 3 steps of **RMSProp** (β = 0.9, s₀ = 0, ε = 10⁻⁸). Compute s₁, w₁, s₂, w₂, s₃, w₃.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Trace 3 steps of **Adam** (β₁ = 0.9, β₂ = 0.999, m₀ = v₀ = 0, ε = 10⁻⁸). Compute m₁, v₁, m̂₁, v̂₁, w₁ for step 1 only.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** At step 1, Adam's bias-corrected update is approximately equal to η = 0.1 regardless of the gradient magnitude. Verify this using your result from (c) — compute m̂₁ / (√v̂₁ + ε) and confirm it is close to 1.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A colleague trains a deep CNN on ImageNet using Adam with η = 0.001 and achieves 72% top-1 accuracy. A paper reports 76% on the same architecture using SGD with momentum and a cosine annealing schedule. Your colleague argues Adam is strictly better because it requires less tuning. Construct a counter-argument — explain the mechanism by which SGD with momentum and a schedule can outperform Adam on vision tasks despite requiring more tuning.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Momentum uses β = 0.9 as the standard default, meaning the current gradient contributes only 10% to the velocity. Consider a scenario where this causes a problem — describe a specific situation where a high β value would hurt training, and explain what value you would use instead and why.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
