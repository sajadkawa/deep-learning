# Exercise 07 — Overfitting, Underfitting & Bias-Variance Tradeoff

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** A model achieves 99% accuracy on the training set and 61% accuracy on the validation set. Which diagnosis is most consistent with this observation?

- a) Underfitting — the model is too simple
- b) Overfitting — the model has memorised the training data
- c) Good fit — high training accuracy is always desirable
- d) Data leakage — the validation set has contaminated the training set

---

**Q2.** You apply L2 regularisation with λ = 0.1 and learning rate η = 0.01 to a weight w = 2.0. Ignoring the original loss gradient for a moment, what is the weight decay factor applied to w at each update step?

- a) 0.998
- b) 0.980
- c) 0.900
- d) 1.002

---

**Q3.** A model has high bias and high variance simultaneously. Which of the following scenarios could produce this?

- a) A very deep network trained on a large, clean dataset
- b) A linear model trained on a large dataset with a non-linear true function
- c) A model with insufficient capacity trained on noisy data with irreducible noise
- d) A well-regularised network trained on a representative dataset

---

**Q4.** You normalise your entire dataset (compute mean and standard deviation, then standardise) before splitting into training, validation, and test sets. What problem does this introduce?

- a) The model will underfit because the data has been altered
- b) The test set statistics leak into the training set, producing optimistically biased performance estimates
- c) The validation set becomes identical to the test set
- d) No problem — normalisation before splitting is standard practice

---

**Q5.** A neural network is trained with dropout rate p = 0.4 using inverted dropout. At test time, which of the following correctly describes what happens?

- a) Each neuron is active with probability 0.6 and its output is unchanged
- b) All neurons are active and outputs are scaled down by 0.6
- c) All neurons are active and no rescaling is needed — the training-time scaling already compensated
- d) Dropout is applied at test time with a reduced rate of 0.2

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** A student argues: *"My model achieves 95% training accuracy, so it has clearly learned the task well."* Explain precisely why this reasoning is flawed. What would you need to see to conclude the model has actually learned the task?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** Why are three separate datasets (training, validation, test) required rather than just two (training and test)? Be precise about what goes wrong if you use the test set to make decisions during development.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** L1 and L2 regularisation both add a penalty to the loss function, yet L1 drives weights to exactly zero while L2 does not. Explain the mathematical reason for this difference. Your answer must refer to the gradient of each penalty term.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** Early stopping is described as "computationally free." What does this mean precisely? Compare it to L2 regularisation in terms of what each technique modifies and what it costs.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** Consider a single-weight network with sigmoid output and MSE loss:

```
z = w × x + b,   ŷ = σ(z),   L_orig = (y − ŷ)²
```

Parameters: w = 0.8, b = 0.0 (bias not regularised), x = 1.0, y = 1.0

You may use: σ(0.8) = 0.6900, σ(z)(1 − σ(z))|_{z=0.8} = 0.2139

Apply L2 regularisation with λ = 0.5 and learning rate η = 0.5.

**(a)** Compute the forward pass: z, ŷ, L_orig, L_reg = λw², and L_total = L_orig + L_reg.

**(b)** Compute the gradients:
- ∂L_orig/∂ŷ = −2(y − ŷ)
- ∂L_orig/∂z = ∂L_orig/∂ŷ × σ(z)(1 − σ(z))
- ∂L_orig/∂w = ∂L_orig/∂z × x
- ∂L_reg/∂w = 2λw
- ∂L_total/∂w = ∂L_orig/∂w + ∂L_reg/∂w

**(c)** Apply the update: w_new = w − η × ∂L_total/∂w. State the new weight.

**(d)** Now compute the unregularised update: w_new = w − η × ∂L_orig/∂w. Compare the two updated weights. By how much did L2 regularisation reduce the weight, and why does this make sense given the weight decay formula w(1 − 2ηλ)?

---

## Section D — Critical Thinking

**Q11.** The bias-variance tradeoff is often presented as a fundamental constraint — you cannot reduce both bias and variance simultaneously. Modern deep learning appears to violate this: very large neural networks achieve both low bias (they fit complex functions) and low variance (they generalise well), a phenomenon sometimes called the "double descent" curve.

Given what you know about bias, variance, and model complexity from this note, construct a hypothesis for why very large networks might escape the classical tradeoff. You are not expected to know the answer — you are expected to reason carefully from first principles.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** A colleague proposes the following regularisation strategy: *"Instead of L2 or dropout, I will simply stop training after 5 epochs regardless of the validation loss. This is simpler than early stopping because it requires no monitoring."*

- **(a)** Identify the flaw in this strategy compared to proper early stopping. Under what conditions would it overfit, and under what conditions would it underfit?
- **(b)** Early stopping saves the model at the epoch with the lowest validation loss, then continues training for a patience window. Why continue training after the best point rather than stopping immediately?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
