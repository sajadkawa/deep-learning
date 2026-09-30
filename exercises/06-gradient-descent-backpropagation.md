# Exercise 06 — Gradient Descent & Backpropagation

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** A network weight currently has gradient ∂L/∂w = −0.8 and the learning rate is η = 0.1. What is the updated weight if the current value is w = 0.5?

- a) 0.42
- b) 0.58
- c) 0.50
- d) 0.66

---

**Q2.** A training dataset has 1000 samples and mini-batch gradient descent is used with batch size B = 50. How many weight updates occur per epoch?

- a) 1
- b) 50
- c) 20
- d) 1000

---

**Q3.** In the backpropagation worked example from the note, the input is x = [0, 1] and the gradient ∂L/∂w⁽¹⁾₁₁ (the weight from x₁ to h₁) is zero. Why?

- a) The sigmoid derivative at z = 0.5 is zero
- b) The weight w⁽¹⁾₁₁ is initialised to zero
- c) The input x₁ = 0 makes the gradient zero regardless of the upstream signal
- d) The loss is zero for this training sample

---

**Q4.** A 10-layer network uses sigmoid activations throughout. Assuming each sigmoid derivative is at its maximum value of 0.25, what is the approximate gradient magnitude at layer 1 if the gradient at the output is 1.0?

- a) 0.25
- b) 0.001
- c) 10⁻⁶
- d) 0.1

---

**Q5.** Gradient clipping rescales the gradient vector when its norm exceeds a threshold. Which of the following correctly describes what clipping preserves and what it changes?

- a) Preserves the gradient magnitude, changes the direction
- b) Preserves the gradient direction, changes the magnitude
- c) Changes both the direction and the magnitude
- d) Preserves both — it only clips individual components that exceed the threshold

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** Explain why the learning rate η is described as the most important hyperparameter in training. What happens concretely when η is too large, and what happens when it is too small? Why is there no universally correct value?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** Batch gradient descent computes the exact gradient over the full dataset before each update. Despite having the most accurate gradient, it is rarely used in practice. Explain why, and state what property of SGD and mini-batch GD makes them preferable despite their noisier gradients.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** In the backpropagation worked example, the network uses MSE loss with sigmoid activations. The note states that in practice BCE would be used for classification. Using what you know from Note 05, explain precisely what would change in the backward pass if BCE replaced MSE — specifically, what happens to the gradient at the output layer and why this is an improvement.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** Vanishing gradients are described as a training failure, not just a slowdown. Explain the distinction. What does a network with vanishing gradients actually learn, and what does it fail to learn?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** Consider a simplified 1-hidden-layer network with a single hidden neuron and a single output neuron:

```
Architecture:  1 input → 1 hidden neuron → 1 output
Activation:    sigmoid σ(z) = 1/(1+e⁻ᶻ) in both layers
Loss:          MSE  L = (y − ŷ)²
```

Parameters:

```
w⁽¹⁾ = 0.5,   b⁽¹⁾ = 0.0
w⁽²⁾ = 1.0,   b⁽²⁾ = 0.0
Learning rate η = 1.0
```

Training sample: x = 1.0, y = 1.0

You may use: σ(0.5) = 0.6225, σ(0.6225) = 0.6508

**(a)** Perform the forward pass. Compute z⁽¹⁾, h, z⁽²⁾, ŷ, and L. Show all arithmetic.

**(b)** Perform the backward pass. Compute the following gradients in order, showing each chain rule step:

- ∂L/∂ŷ
- ∂L/∂z⁽²⁾ = ∂L/∂ŷ × σ(z⁽²⁾)(1 − σ(z⁽²⁾))
- ∂L/∂w⁽²⁾ = ∂L/∂z⁽²⁾ × h  and  ∂L/∂b⁽²⁾ = ∂L/∂z⁽²⁾
- ∂L/∂h = ∂L/∂z⁽²⁾ × w⁽²⁾
- ∂L/∂z⁽¹⁾ = ∂L/∂h × σ(z⁽¹⁾)(1 − σ(z⁽¹⁾))
- ∂L/∂w⁽¹⁾ = ∂L/∂z⁽¹⁾ × x  and  ∂L/∂b⁽¹⁾ = ∂L/∂z⁽¹⁾

You may use: σ(0.6225)(1 − σ(0.6225)) = 0.2273, σ(0.5)(1 − σ(0.5)) = 0.2350

**(c)** Apply the gradient descent update with η = 1.0. State the new values of all four parameters.

**(d)** Perform a second forward pass with the new weights. Compute the new ŷ and new loss L. Has the loss decreased? You may use: σ(0.5746) = 0.6398, σ(0.8618) = 0.7030

---

## Section D — Critical Thinking

**Q11.** A student argues: *"Backpropagation is just the chain rule — there is nothing special about it. You could compute the same gradients by perturbing each weight slightly and measuring the change in loss."* This alternative is called numerical differentiation. Construct a precise argument for why backpropagation is not just a convenience but a practical necessity. Your argument must address computational cost and must give a concrete example with numbers.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** The note states that in high-dimensional networks, true local minima are rare — most apparent traps are saddle points. A saddle point has zero gradient in some directions but not others.

- **(a)** Why does this distinction matter for gradient descent? What happens at a true local minimum versus a saddle point when training continues?
- **(b)** Mini-batch gradient descent introduces noise into the gradient estimate. A student argues this noise is purely harmful — it makes the gradient less accurate. Construct a counter-argument: describe a specific scenario where gradient noise is beneficial, and explain the mechanism.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
