# Exercise 04 — Activation Functions

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** A network has 6 hidden layers, each using sigmoid activation. Assuming the gradient at the output layer is 1.0 and each sigmoid derivative is at its maximum value of 0.25, what is the approximate gradient magnitude reaching the first hidden layer?

- a) 0.25
- b) 0.0039
- c) 0.000244
- d) 0.000015

---

**Q2.** Which of the following is the correct derivative of the sigmoid function σ(z)?

- a) σ(z) · σ(−z)
- b) σ(z) · (1 + σ(z))
- c) σ(z) · (1 − σ(z))
- d) (1 − σ(z))²

---

**Q3.** A neuron using ReLU activation has weights w = [0.5, −2.0] and bias b = −1.0. For input x = [1, 1]ᵀ, what is the output of this neuron?

- a) −2.5
- b) 0.0
- c) 2.5
- d) 1.0

---

**Q4.** Softmax is applied at the output layer of a 4-class classifier. The raw scores are z = [1.0, 1.0, 1.0, 1.0]. What is the softmax output?

- a) [1.0, 0.0, 0.0, 0.0]
- b) [0.5, 0.5, 0.0, 0.0]
- c) [0.25, 0.25, 0.25, 0.25]
- d) [0.4, 0.3, 0.2, 0.1]

---

**Q5.** Which activation function introduced the dying neuron problem, and which property of that function causes it?

- a) Sigmoid — because its output is always positive
- b) Tanh — because it saturates at both ends
- c) ReLU — because its gradient is exactly zero for all negative inputs
- d) Leaky ReLU — because its slope α is too small to recover dead neurons

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** Explain why the Heaviside step function — which worked correctly for the single perceptron — cannot be used as the activation function in the hidden layers of an MLP trained with backpropagation. Your answer must reference the chain rule explicitly.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** Sigmoid and tanh both saturate and both cause vanishing gradients. State one concrete advantage tanh has over sigmoid for hidden layers, and explain the mechanism behind it. Then state why neither is the preferred choice for deep feedforward hidden layers today.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** A colleague proposes using softmax activation in every hidden layer of a deep network, arguing that it produces well-normalised outputs between 0 and 1 at every layer. Identify two distinct problems this would cause and explain each.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** ReLU is not differentiable at z = 0. In practice this is handled by setting f'(0) = 0. Explain why this is acceptable in training despite being mathematically imprecise. What would happen if f'(0) were set to 1 instead?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** Consider a single hidden neuron with two inputs. The weights and bias are:

```
w₁ = 0.8,   w₂ = −1.2,   b = 0.5
```

**(a)** Compute the pre-activation z and the output f(z) for each of the following inputs, using the activation function specified. Show all arithmetic.

| Input | Activation | z | f(z) |
|---|---|---|---|
| x = [1.0, 1.0]ᵀ | Sigmoid | | |
| x = [1.0, 1.0]ᵀ | Tanh | | |
| x = [1.0, 1.0]ᵀ | ReLU | | |
| x = [1.0, 1.0]ᵀ | Leaky ReLU (α = 0.01) | | |
| x = [2.0, 3.0]ᵀ | ReLU | | |
| x = [2.0, 3.0]ᵀ | Leaky ReLU (α = 0.01) | | |

**(b)** For the sigmoid case with x = [1.0, 1.0]ᵀ, compute the derivative dσ/dz using the formula dσ/dz = σ(z)(1 − σ(z)). Show your working.

**(c)** For the tanh case with x = [1.0, 1.0]ᵀ, compute the derivative d(tanh)/dz = 1 − tanh²(z). Show your working.

**(d)** Suppose this neuron is the only hidden neuron in a 1-hidden-layer network. The upstream gradient arriving from the output layer is δ = 0.6. Using the chain rule, compute the gradient that flows back through this neuron for each activation function (sigmoid, tanh, ReLU, Leaky ReLU) at input x = [1.0, 1.0]ᵀ. Use your results from parts (a)–(c).

---

## Section D — Critical Thinking

**Q11.** ReLU has a non-zero gradient only for positive inputs (gradient = 1) and zero gradient for negative inputs (gradient = 0). A student argues: *"This means ReLU also suffers from a vanishing gradient problem — half the neurons have zero gradient at any given input."* Is this argument correct? Explain precisely why ReLU's zero gradient for negative inputs is fundamentally different from sigmoid's vanishing gradient problem, and under what specific condition ReLU's zero gradient does become a serious problem.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** The temperature parameter T in softmax controls the sharpness of the probability distribution. Consider a language model that has been trained with standard softmax (T = 1) and is now used for text generation.

- **(a)** What happens to the generated text as T → 0? What happens as T → ∞? Explain in terms of what the probability distribution looks like in each case.
- **(b)** A teacher network is used to train a smaller student network via knowledge distillation. The teacher's softmax outputs are computed at T = 4 rather than T = 1. Why might a high temperature be useful here — what information is preserved at T = 4 that is lost at T = 1?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
