# Exercise 02 — Biological Neuron → Artificial Neuron → Perceptron

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** The bias term `b` in the perceptron formula `z = wᵀx + b` has which geometric effect on the decision boundary?

- a) It rotates the decision boundary around the origin
- b) It scales the distance between classes
- c) It allows the decision boundary to shift freely away from the origin
- d) It prevents the decision boundary from becoming vertical

---

**Q2.** A perceptron computes `ŷ = 1` and the true label is `y = 1`. According to the perceptron learning rule, what happens?

- a) Weights are increased to reinforce the correct prediction
- b) Weights are decreased to prevent overconfidence
- c) No update is made
- d) Only the bias is updated, not the weights

---

**Q3.** Which of the following functions can a single perceptron with a step function learn?

- a) XOR
- b) XNOR
- c) OR
- d) Both XOR and OR

---

**Q4.** A perceptron's step function is replaced with a linear activation `f(z) = z`, and 10 such neurons are stacked into layers. Compared to a single neuron with no activation, this network is:

- a) More powerful because it has more parameters
- b) More powerful because linear functions can approximate any curve
- c) Equivalent — the entire network still computes a single linear transformation
- d) Less powerful because linear activations reduce the output range

---

**Q5.** A perceptron is trained on a dataset where the correct decision boundary does not pass through the origin. The perceptron has no bias term. What will happen during training?

- a) Training will converge to the correct boundary because weights compensate for the missing bias
- b) Training will converge but to a suboptimal boundary forced through the origin
- c) Training will not converge because the perceptron learning rule requires a bias
- d) Training will converge correctly if the learning rate is small enough

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** The perceptron learning rule only updates weights when the prediction is wrong — it does nothing on correct predictions. Explain mathematically why updating on correct predictions would be harmful, not helpful.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** A perceptron is trained on the OR function and converges with weights `w₁ = 0.2, w₂ = 0.2, b = 0.3` and threshold `0.5`. A new point `(0.5, 0.5)` with label `1` is added to the training set. Without running the learning rule, determine whether the perceptron will need to update its weights for this point. Show your reasoning.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** Explain why XOR is not linearly separable using geometry only — no algebra, no inequalities. Your answer must reference the positions of the four XOR points in the plane and what any straight line would have to do to separate them.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** The linearity collapse argument shows that stacking linear layers produces a network equivalent to a single linear transformation. A student concludes: "So a single neuron is just as powerful as any deep linear network — depth adds nothing." This conclusion is technically correct but misses something important. What does it miss?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** Train a perceptron on the AND function from scratch using the following setup:

```
AND Truth Table:
  x₁  x₂  y
   0   0   0
   0   1   0
   1   0   0
   1   1   1

Initial weights:  w₁ = 0.0,  w₂ = 0.0,  b = 0.0
Learning rate:    η = 0.1
Activation:       ŷ = 1 if z ≥ 0.5, else 0
```

**(a)** Trace the perceptron learning rule sample by sample through each epoch. Show `z`, `ŷ`, `e`, and the updated weights after every sample. Stop when a full epoch passes with no updates.

**(b)** Write the equation of the final decision boundary in the form `x₂ = mx₁ + c`. Identify the slope and intercept.

**(c)** Draw the decision boundary on the x₁-x₂ plane and verify that all four AND points are correctly classified by the boundary.

---

## Section D — Critical Thinking

**Q11.** Minsky and Papert's 1969 book *Perceptrons* proved mathematically that a single perceptron cannot solve XOR, and this finding nearly halted neural network research for over a decade. In hindsight, was their criticism valid? What did they get right, what did they miss, and what does this episode reveal about how mathematical proofs can be misread in science?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** A perceptron is trained on a linearly separable dataset and the learning rule converges — a full epoch passes with zero updates. A careful student says: "Convergence does not mean the boundary is correct." What does this student mean? Describe a concrete scenario where a perceptron converges to a boundary that correctly classifies all training points but would fail on new data, and explain what this reveals about what the perceptron learning rule actually optimises for.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
