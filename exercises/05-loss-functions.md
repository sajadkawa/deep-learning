# Exercise 05 — Loss Functions

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** Which of the following is a required property of any loss function used with gradient descent?

- a) It must always produce a value between 0 and 1
- b) It must be differentiable with respect to the predicted output ŷ
- c) It must be symmetric — overestimates and underestimates must cost the same
- d) It must be convex so that gradient descent always finds the global minimum

---

**Q2.** For a single sample, the MSE gradient with respect to the prediction ŷ is:

- a) −2(y − ŷ)²
- b) 2(y − ŷ)
- c) −(y − ŷ)
- d) 2(ŷ − y)

---

**Q3.** A binary classification network outputs ŷ = 0.3 for a sample with true label y = 1. What is the binary cross-entropy loss for this sample?

- a) 0.357
- b) 0.700
- c) 1.204
- d) 2.303

---

**Q4.** A 3-class network outputs softmax probabilities ŷ = [0.2, 0.5, 0.3] for a sample whose true class is class 2 (0-indexed: class 1). What is the categorical cross-entropy loss?

- a) 0.916
- b) 0.693
- c) 0.357
- d) 1.204

---

**Q5.** A binary classification network with sigmoid output predicts ŷ = 0.02 when the true label is y = 1. Which statement correctly describes the gradient ∂L/∂z at this point for MSE versus BCE?

- a) MSE gives a larger gradient than BCE because the squared error is larger
- b) Both give the same gradient because the prediction error is the same
- c) BCE gives a gradient of approximately −0.98; MSE gives a gradient of approximately −0.04
- d) BCE gives a gradient of approximately −0.98; MSE gives a gradient of approximately −0.98

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** A loss function must be differentiable with respect to ŷ. Explain precisely why this requirement exists — what breaks in the training process if the loss function is not differentiable at some point?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** In the derivation of binary cross-entropy from maximum likelihood, the log is applied to convert the product of probabilities into a sum. Explain why this step is necessary — give both the numerical reason and the mathematical reason.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** The gradient of BCE with sigmoid simplifies to ∂L/∂z = ŷ − y. Walk through the key step that makes this cancellation happen and explain what it means for how the network learns when it is confidently wrong.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** The loss landscape for a deep network is described as having local minima, saddle points, and plateaus. Explain what a saddle point is and why it is considered more problematic in practice than a local minimum for high-dimensional networks.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** A binary classification network produces the following predictions on three training samples:

| Sample | True label y | Prediction ŷ |
|---|---|---|
| 1 | 1 | 0.8 |
| 2 | 0 | 0.3 |
| 3 | 1 | 0.1 |

**(a)** Compute the MSE loss for each sample and the mean MSE over all three samples. Show all arithmetic.

**(b)** Compute the BCE loss for each sample and the mean BCE over all three samples. Use ln (natural log). Show all arithmetic. You may use the following values: ln(0.8) = −0.223, ln(0.2) = −1.609, ln(0.7) = −0.357, ln(0.3) = −1.204, ln(0.1) = −2.303, ln(0.9) = −0.105.

**(c)** For sample 3 (y = 1, ŷ = 0.1), compute the gradient ∂L/∂z for both MSE and BCE, where z is the pre-sigmoid input. Assume σ(z) = 0.1, so z ≈ −2.197.

For MSE: use ∂L/∂z = ∂L/∂ŷ × dσ/dz, where ∂L/∂ŷ = 2(ŷ − y) and dσ/dz = σ(z)(1 − σ(z)).

For BCE: use ∂L/∂z = ŷ − y directly.

**(d)** Sample 3 is the hardest sample — the network is confidently wrong. Based on your results from part (c), which loss function sends a stronger gradient signal for this sample? What does this mean for how quickly the network corrects this mistake?

---

## Section D — Critical Thinking

**Q11.** A student trains a binary classification network using MSE loss instead of BCE. The network converges — the loss decreases and eventually stabilises. The student concludes that MSE works fine for classification. Construct a precise argument for why this conclusion is wrong even if the final accuracy is acceptable. Your argument must address: (a) what the network is actually optimising under MSE versus BCE, and (b) what happens to the gradient signal during the early stages of training when the network is making confident wrong predictions.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** The categorical cross-entropy loss L = −log(ŷ_c) depends only on the probability assigned to the correct class c. It does not directly penalise the probabilities assigned to wrong classes.

- **(a)** Does this mean the network never adjusts the scores for wrong classes? Explain your answer by considering how softmax connects all class scores to the correct class probability.
- **(b)** A network is trained on a 10-class problem. For a given sample, the correct class has ŷ_c = 0.4. Two wrong classes have ŷ = 0.3 each, and the remaining seven have ŷ ≈ 0.0. Another sample has ŷ_c = 0.4 but the probability is spread evenly across all 9 wrong classes (ŷ ≈ 0.067 each). Both samples have the same CCE loss. Are these two situations equally good from a learning perspective? Explain what the gradient signal looks like in each case.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
