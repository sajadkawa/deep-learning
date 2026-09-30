# Exercise 10 — Weight Initialization

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** A fully connected layer has 128 inputs and 256 outputs. You are using ReLU activations. Which initialization is correct?

- a) wᵢⱼ ~ N(0, 1 / 128)
- b) wᵢⱼ ~ N(0, 2 / 128)
- c) wᵢⱼ ~ N(0, 2 / (128 + 256))
- d) wᵢⱼ ~ N(0, 1 / (128 + 256))

---

**Q2.** A network is initialized with all weights set to the same small constant c ≠ 0. Which of the following correctly describes what happens during training?

- a) Training proceeds normally — the constant is non-zero so symmetry is broken
- b) All neurons in each layer remain identical throughout training
- c) The network converges but more slowly than with random initialization
- d) Only the output layer is affected — hidden layers train correctly

---

**Q3.** Xavier initialization sets Var(w) = 2 / (nₗ₋₁ + nₗ). The denominator is a compromise between two conditions. What are those two conditions?

- a) Preserving variance in the forward pass (requires 1/nₗ₋₁) and preserving variance in the backward pass (requires 1/nₗ)
- b) Preventing saturation (requires 1/nₗ₋₁) and preventing vanishing gradients (requires 1/nₗ)
- c) Matching fan-in (requires nₗ₋₁) and matching fan-out (requires nₗ) for symmetry
- d) Ensuring the weight matrix is orthogonal in the forward direction and the backward direction

---

**Q4.** You initialize a deep ReLU network with Xavier initialization instead of He initialization. What is the most likely symptom after the first forward pass?

- a) Activations explode — Xavier variance is too large for ReLU
- b) Activations gradually shrink across layers — Xavier does not compensate for ReLU zeroing half its inputs
- c) Activations are stable — Xavier and He produce identical results for ReLU
- d) The output layer saturates — Xavier pushes activations into the sigmoid saturation region

---

**Q5.** Which of the following correctly states why biases can be safely initialized to zero while weights cannot?

- a) Biases are smaller in magnitude than weights, so zero is a better starting point
- b) Biases do not connect neurons to each other, so identical bias values do not create the symmetry problem
- c) Biases are updated by a different rule than weights, so their initial value does not matter
- d) Biases only affect the output layer, where symmetry is not a concern

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** The note proves that zero initialization causes permanent symmetry — neurons that start identical remain identical forever. A student argues: *"This only applies to the first layer. By the time the signal reaches deeper layers, the activations will differ because the loss is different for each output."* Identify the flaw in this argument.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** Xavier initialization is derived from two conditions — one from the forward pass and one from the backward pass — that cannot both be satisfied simultaneously. Explain precisely why they cannot both be satisfied, and what assumption Glorot and Bengio made to resolve the conflict.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** He initialization differs from Xavier (fan-in only version) by a factor of 2 in the numerator. Derive in plain terms where this factor of 2 comes from — what property of ReLU makes it necessary?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** The note describes initialization and activation function choice as two axes that both contribute to vanishing and exploding gradients. Explain what it means for two axes to "multiply through the chain rule" — give a concrete example of a combination that produces doubly vanishing gradients and explain why it is worse than either axis alone.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** A network has the following architecture: n₀ = 64, n₁ = 32, n₂ = 16, n₃ = 1. All hidden layers use ReLU. The output layer uses a linear activation.

**(a)** Compute the He initialization standard deviation σ for layers 1 and 2 (the two ReLU hidden layers). Show your working.

**(b)** Compute the Xavier initialization standard deviation σ for layer 3 (the linear output layer). Show your working.

**(c)** Now suppose instead that all three layers are initialized with a fixed σ = 0.01 (too small). Trace the approximate activation variance across layers 1, 2, and 3, assuming the input has variance 1.0 and each layer has the fan-in given above.

Use: Var(z⁽ˡ⁾) = nₗ₋₁ · Var(w) · Var(a⁽ˡ⁻¹⁾), and Var(ReLU(z)) = Var(z) / 2.

Show the variance after each layer and state what has happened to the signal by layer 3.

**(d)** Repeat part (c) using He initialization for layers 1 and 2 and Xavier for layer 3. Show that the variance is preserved.

You may use: √(2/64) ≈ 0.177, √(2/32) ≈ 0.250, √(2/48) ≈ 0.204.

---

## Section D — Critical Thinking

**Q11.** Batch normalization (introduced in Note 11) normalizes activations at each layer during training, actively correcting variance drift. A student concludes: *"If we use batch normalization, initialization doesn't matter — the normalization will fix any variance problem within the first few steps."*

Construct a precise counter-argument. Your answer should address what happens during the first forward and backward pass before batch normalization has had any effect, and whether batch normalization can recover from a catastrophic starting point.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** Consider two mismatched initialization choices:

- **Case A:** A deep network with tanh activations, initialized with He initialization (Var(w) = 2 / nₗ₋₁)
- **Case B:** A deep network with ReLU activations, initialized with Xavier initialization (Var(w) = 2 / (nₗ₋₁ + nₗ))

For each case, reason through what happens to activation variance across layers. Does the mismatch cause vanishing activations, exploding activations, or something else? Which case is more damaging in practice, and why?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
