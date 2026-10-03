# 12 — Optimization Algorithms: SGD, Momentum, RMSProp, Adam

---

## Part 1: Limitations of Vanilla Gradient Descent

Gradient descent updates every parameter by the same rule at every step:

```
w ← w − η · ∂L/∂w
```

This works in principle — if the loss surface were a smooth bowl, gradient descent would roll straight to the minimum. In practice, the loss surfaces of deep networks are not smooth bowls. They have structure that makes vanilla gradient descent slow, unstable, or stuck.

### Ravines

A ravine is a region of the loss surface where curvature is much higher in one direction than another — like a narrow valley. Gradient descent oscillates across the steep walls while making slow progress along the valley floor.

```
High curvature direction:   large gradient → large step → overshoots → oscillates
Low curvature direction:    small gradient → small step → slow progress
```

The same learning rate η governs both directions. A large η fixes the slow direction but causes oscillation in the steep direction. A small η fixes the oscillation but makes the slow direction even slower. There is no single η that works well for both.

### Saddle Points

A saddle point is a location where the gradient is zero but it is not a minimum — the loss decreases in some directions and increases in others. In high-dimensional spaces, saddle points are far more common than local minima.

```
At a saddle point:
  ∂L/∂w = 0   for all parameters
  → gradient descent stops updating
  → training stalls
```

Vanilla gradient descent has no mechanism to escape a saddle point once it arrives there — the gradient is zero, so the update is zero.

### Flat Regions

In flat regions of the loss surface, gradients are very small. Gradient descent makes tiny steps and converges extremely slowly. This is common early in training with poor initialization, and near the end of training when approaching a minimum.

> These three failure modes — ravines, saddle points, and flat regions — all have the same root cause: vanilla gradient descent uses only the current gradient, with no memory of where it has been and no adaptation to the local geometry of the loss surface. Every optimizer in this note addresses one or more of these failures.

---

## Part 2: Momentum

The problem with ravines is that gradient descent oscillates — it overshoots in the steep direction and undershoots in the flat direction. The fix is to give the optimizer memory: instead of updating based on the current gradient alone, accumulate a running average of past gradients.

This is **momentum**. The update rule becomes:

```
vₜ = β · vₜ₋₁ + (1 − β) · ∂L/∂w
wₜ = wₜ₋₁ − η · vₜ
```

Where vₜ is the **velocity** — an exponential moving average of past gradients — and β is the momentum coefficient (typically 0.9).

### How Momentum Helps

In a ravine, gradients in the steep direction alternate in sign (positive, negative, positive, ...) because the optimizer oscillates across the walls. When these are averaged, they cancel out — the velocity in the steep direction stays small. Gradients in the flat direction are consistently the same sign, so they accumulate — the velocity in the flat direction grows.

```
Steep direction:   gradients alternate sign → cancel in average → small velocity → less oscillation
Flat direction:    gradients same sign → accumulate → large velocity → faster progress
```

The β parameter controls how much history is retained. β = 0.9 means the current gradient contributes 10% and the history contributes 90%.

> Momentum dampens oscillation in high-curvature directions and accelerates progress in low-curvature directions — by accumulating gradient history rather than reacting to each gradient independently.

### Worked Example

One parameter w, learning rate η = 0.1, momentum β = 0.9. Starting at w₀ = 1.0, v₀ = 0.

Gradients at each step (simulating a ravine — alternating sign in one direction):

```
Step 1:  g₁ = 0.8
Step 2:  g₂ = 0.6
Step 3:  g₃ = 0.7
```

**Step 1:**
```
v₁ = 0.9 × 0.0 + 0.1 × 0.8 = 0.08
w₁ = 1.0 − 0.1 × 0.08 = 1.0 − 0.008 = 0.992
```

**Step 2:**
```
v₂ = 0.9 × 0.08 + 0.1 × 0.6 = 0.072 + 0.06 = 0.132
w₂ = 0.992 − 0.1 × 0.132 = 0.992 − 0.0132 = 0.9788
```

**Step 3:**
```
v₃ = 0.9 × 0.132 + 0.1 × 0.7 = 0.1188 + 0.07 = 0.1888
w₃ = 0.9788 − 0.1 × 0.1888 = 0.9788 − 0.01888 = 0.9599
```

The velocity grows across steps because the gradients are consistently positive — momentum is accelerating progress in this direction.

---

## Part 3: RMSProp

Momentum addresses oscillation by smoothing the gradient direction. But it does not address the ravine problem directly — it still uses the same learning rate η for all parameters. What we want is a **per-parameter learning rate** that is large for parameters with small gradients and small for parameters with large gradients.

**RMSProp** (Root Mean Square Propagation) achieves this by tracking the magnitude of recent gradients for each parameter:

```
sₜ = β · sₜ₋₁ + (1 − β) · (∂L/∂w)²
wₜ = wₜ₋₁ − (η / √(sₜ + ε)) · ∂L/∂w
```

Where sₜ is an exponential moving average of squared gradients, and ε is a small constant (e.g. 10⁻⁸) for numerical stability.

### How RMSProp Helps

The effective learning rate for each parameter is η / √sₜ:

```
Large gradients → large sₜ → small effective learning rate → prevents overshooting
Small gradients → small sₜ → large effective learning rate → accelerates slow directions
```

This is exactly what is needed in a ravine: the steep direction has large gradients and gets a small effective learning rate (reducing oscillation), while the flat direction has small gradients and gets a large effective learning rate (accelerating progress).

> RMSProp adapts the learning rate per parameter based on the recent history of gradient magnitudes — parameters that have been receiving large gradients get smaller updates, and parameters that have been receiving small gradients get larger updates.

### Worked Example

Same parameter, same gradients as Part 2. η = 0.1, β = 0.9, ε = 10⁻⁸, s₀ = 0, w₀ = 1.0.

**Step 1:**
```
s₁ = 0.9 × 0.0 + 0.1 × (0.8)² = 0.064
w₁ = 1.0 − (0.1 / √(0.064 + ε)) × 0.8
   = 1.0 − (0.1 / 0.2530) × 0.8
   = 1.0 − 0.3953 × 0.8
   = 1.0 − 0.3162 = 0.6838
```

**Step 2:**
```
s₂ = 0.9 × 0.064 + 0.1 × (0.6)² = 0.0576 + 0.036 = 0.0936
w₂ = 0.6838 − (0.1 / √0.0936) × 0.6
   = 0.6838 − (0.1 / 0.3060) × 0.6
   = 0.6838 − 0.1961 = 0.4877
```

**Step 3:**
```
s₃ = 0.9 × 0.0936 + 0.1 × (0.7)² = 0.08424 + 0.049 = 0.13324
w₃ = 0.4877 − (0.1 / √0.13324) × 0.7
   = 0.4877 − (0.1 / 0.3650) × 0.7
   = 0.4877 − 0.1918 = 0.2959
```

RMSProp takes much larger steps than momentum here because the effective learning rate η/√s is larger than η alone when s < 1 — which is typical early in training.

---

## Part 4: Adam

Adam (Adaptive Moment Estimation) combines momentum and RMSProp into a single optimizer. It maintains two moving averages per parameter:

```
mₜ = β₁ · mₜ₋₁ + (1 − β₁) · ∂L/∂w        ← first moment  (like momentum)
vₜ = β₂ · vₜ₋₁ + (1 − β₂) · (∂L/∂w)²     ← second moment (like RMSProp)
```

Where β₁ = 0.9 and β₂ = 0.999 are the standard defaults.

### Bias Correction

There is a problem with these moving averages at the start of training. Both m and v are initialised to zero. In the first few steps, the averages are heavily biased toward zero — they have not had enough steps to accumulate a meaningful estimate.

For example, at step 1 with β₁ = 0.9:

```
m₁ = 0.9 × 0 + 0.1 × g₁ = 0.1 × g₁
```

The estimate m₁ is only 10% of the true gradient — it is biased toward zero by a factor of 10. Adam corrects for this:

```
m̂ₜ = mₜ / (1 − β₁ᵗ)
v̂ₜ = vₜ / (1 − β₂ᵗ)
```

At step 1: 1 − β₁¹ = 1 − 0.9 = 0.1, so m̂₁ = m₁ / 0.1 = 10 × m₁ — restoring the true scale. As t grows, β₁ᵗ → 0 and the correction factor → 1, so bias correction has no effect after the first few steps.

The full Adam update:

```
mₜ = β₁ · mₜ₋₁ + (1 − β₁) · ∂L/∂w
vₜ = β₂ · vₜ₋₁ + (1 − β₂) · (∂L/∂w)²

m̂ₜ = mₜ / (1 − β₁ᵗ)
v̂ₜ = vₜ / (1 − β₂ᵗ)

wₜ = wₜ₋₁ − η · m̂ₜ / (√v̂ₜ + ε)
```

### Worked Example

Same parameter, same gradients. η = 0.1, β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, m₀ = v₀ = 0, w₀ = 1.0.

**Step 1** (g₁ = 0.8):
```
m₁ = 0.9 × 0 + 0.1 × 0.8 = 0.08
v₁ = 0.999 × 0 + 0.001 × (0.8)² = 0.00064

m̂₁ = 0.08 / (1 − 0.9¹)   = 0.08 / 0.1    = 0.8
v̂₁ = 0.00064 / (1 − 0.999¹) = 0.00064 / 0.001 = 0.64

w₁ = 1.0 − 0.1 × 0.8 / (√0.64 + ε)
   = 1.0 − 0.1 × 0.8 / 0.8
   = 1.0 − 0.1 = 0.9
```

**Step 2** (g₂ = 0.6):
```
m₂ = 0.9 × 0.08 + 0.1 × 0.6 = 0.072 + 0.06 = 0.132
v₂ = 0.999 × 0.00064 + 0.001 × (0.6)² = 0.00063936 + 0.00036 = 0.00099936

m̂₂ = 0.132 / (1 − 0.9²)   = 0.132 / 0.19   = 0.6947
v̂₂ = 0.00099936 / (1 − 0.999²) = 0.00099936 / 0.001999 = 0.4999

w₂ = 0.9 − 0.1 × 0.6947 / (√0.4999 + ε)
   = 0.9 − 0.1 × 0.6947 / 0.7071
   = 0.9 − 0.0982 = 0.8018
```

**Step 3** (g₃ = 0.7):
```
m₃ = 0.9 × 0.132 + 0.1 × 0.7 = 0.1188 + 0.07 = 0.1888
v₃ = 0.999 × 0.00099936 + 0.001 × (0.7)² = 0.00099836 + 0.00049 = 0.00148836

m̂₃ = 0.1888 / (1 − 0.9³)   = 0.1888 / 0.271  = 0.6967
v̂₃ = 0.00148836 / (1 − 0.999³) = 0.00148836 / 0.002997 = 0.4966

w₃ = 0.8018 − 0.1 × 0.6967 / (√0.4966 + ε)
   = 0.8018 − 0.1 × 0.6967 / 0.7047
   = 0.8018 − 0.0989 = 0.7029
```

### Comparing the Three Optimizers

The three-step trace on the same parameter and gradients:

```
Step    Momentum w    RMSProp w    Adam w
──────  ────────────  ───────────  ──────────
0       1.0000        1.0000       1.0000
1       0.9920        0.6838       0.9000
2       0.9788        0.4877       0.8018
3       0.9599        0.2959       0.7029
```

Momentum makes small, consistent steps — it accumulates velocity but the effective step size is still governed by η. RMSProp makes large steps early because the running average s is small, giving a large effective learning rate. Adam takes steps of approximately η = 0.1 per step — the bias correction normalises the update so the effective step size is approximately constant and equal to η, regardless of gradient magnitude.

> Adam's bias correction gives it a well-defined effective step size from the very first update — approximately η per step. This is why Adam is less sensitive to the choice of η than RMSProp or momentum alone.

---

## Part 5: Learning Rate Schedules

Even with a good optimizer, a fixed learning rate is rarely optimal throughout training. Early in training, a larger learning rate explores the loss surface quickly. Later in training, a smaller learning rate allows fine-grained convergence near a minimum.

A **learning rate schedule** reduces η over the course of training according to a predefined rule.

The diagram below shows step decay and cosine annealing over 90 epochs, illustrating the difference between discontinuous and smooth schedules.

![Learning rate schedules — step decay vs cosine annealing](images/lr_schedules.png)

### Step Decay

Reduce the learning rate by a fixed factor every k epochs:

```
η(epoch) = η₀ × γ^⌊epoch / k⌋
```

Where γ is the decay factor (e.g. 0.1) and k is the step size (e.g. every 30 epochs).

```
Example: η₀ = 0.1, γ = 0.1, k = 30

Epochs  0–29:   η = 0.1
Epochs 30–59:   η = 0.01
Epochs 60–89:   η = 0.001
```

Step decay is simple and widely used. The discontinuous drops can cause instability at each step boundary.

### Cosine Annealing

Reduce the learning rate smoothly following a cosine curve from η₀ to η_min over T epochs:

```
η(t) = η_min + (1/2)(η₀ − η_min)(1 + cos(πt / T))
```

The learning rate starts at η₀, decreases smoothly to η_min, and can be restarted periodically (cosine annealing with warm restarts). The smooth decay avoids the instability of step decay.

```
t=0:    η = η₀          (maximum)
t=T/2:  η ≈ (η₀+η_min)/2  (midpoint)
t=T:    η = η_min       (minimum)
```

: Choosing an Optimizer in Practice

```
Situation                                    Recommendation
───────────────────────────────────────────  ──────────────────────────────────────────
Default starting point                       Adam (β₁=0.9, β₂=0.999, η=0.001)
Computer vision, CNNs                        SGD + momentum (β=0.9) with schedule
                                             Often outperforms Adam with tuning
NLP, transformers                            Adam or AdamW (Adam + weight decay)
RNNs                                         RMSProp — handles non-stationary gradients
Small dataset, risk of overfitting           SGD + momentum — less aggressive adaptation
Need fast convergence, less tuning           Adam — robust to η choice
```

### Common Pitfalls

```
Pitfall                                      Fix
───────────────────────────────────────────  ──────────────────────────────────────────
Learning rate too large                      Loss diverges or oscillates — reduce η
Learning rate too small                      Training is very slow — increase η
No learning rate schedule                    Loss plateaus early — add step decay or
                                             cosine annealing
Using Adam with very large η                 Adam is robust but not immune — η=0.001
                                             is the standard starting point
Forgetting weight decay                      Use AdamW instead of Adam for L2
                                             regularization — Adam's adaptive rates
                                             interact poorly with standard L2
```

> No optimizer is universally best. Adam is the robust default — it works well across most tasks with minimal tuning. SGD with momentum and a schedule often achieves better final performance on vision tasks but requires more careful tuning. Start with Adam; switch to SGD if you have the time to tune.

---

## Summary

Key takeaways:

- Vanilla gradient descent fails on deep network loss surfaces due to ravines (oscillation across steep walls), saddle points (zero gradient, training stalls), and flat regions (tiny gradients, slow progress)
- Momentum accumulates an exponential moving average of past gradients — gradients that alternate in sign cancel out (reducing oscillation), gradients that are consistently signed accumulate (accelerating progress). β = 0.9 is the standard default
- RMSProp adapts the learning rate per parameter by dividing by the root mean square of recent gradients — large gradients get smaller effective learning rates, small gradients get larger effective learning rates
- Adam combines momentum (first moment m) and RMSProp (second moment v) with bias correction — the correction removes the zero-initialisation bias in the first few steps and gives Adam an approximately constant effective step size of η from the start
- Bias correction is essential in Adam's early steps — without it, m and v are heavily biased toward zero, producing undersized updates
- Learning rate schedules reduce η during training: step decay reduces by a fixed factor every k epochs (simple but discontinuous), cosine annealing reduces smoothly following a cosine curve (preferred for stability)
- Adam is the robust default for most tasks; SGD with momentum and a schedule often achieves better final performance on vision tasks with careful tuning; RMSProp is preferred for RNNs
