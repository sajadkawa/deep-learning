# Exercise 11 — Batch Normalization

---

## Section A — Multiple Choice

**1.** What is internal covariate shift?

- a) The shift in the loss surface caused by changing the learning rate
- b) The continuous change in each layer's input distribution caused by weight updates in preceding layers
- c) The change in gradient magnitude across layers during backpropagation
- d) The shift in output distribution caused by changing the batch size

&nbsp;

**2.** In batch normalization, what is the purpose of the learnable parameters γ and β?

- a) To control the learning rate for the batch norm layer
- b) To replace the weights and biases of the layer
- c) To restore the network's ability to represent any mean and variance after normalization
- d) To prevent the normalized values from becoming negative

&nbsp;

**3.** Why does batch normalization behave differently at inference compared to training?

- a) The activation function is disabled at inference
- b) At inference there is no mini-batch, so batch statistics cannot be computed — running statistics are used instead
- c) The learnable parameters γ and β are frozen at inference
- d) Batch normalization is always disabled at inference to speed up prediction

&nbsp;

**4.** A network is evaluated with `model.train()` still active instead of `model.eval()`. What is the consequence for batch normalization?

- a) The running statistics are reset to zero
- b) γ and β are ignored and the output is always zero-mean unit-variance
- c) Batch statistics are computed from single examples, producing incorrect and non-deterministic outputs
- d) The network falls back to using running statistics automatically

&nbsp;

**5.** After the normalization step in batch norm, every feature has mean 0 and variance 1. Why is this insufficient on its own without γ and β?

- a) Mean 0 and variance 1 causes vanishing gradients in all activation functions
- b) It removes the network's freedom to represent any other distribution — the optimal distribution for a layer may not be zero-mean unit-variance
- c) It makes the batch norm layer non-differentiable
- d) It causes the running statistics to diverge during training

&nbsp;

---

## Section B — Short Answer

**1.** Explain why batch normalization acts as a mild regularizer, even though it was not designed for that purpose.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A network is trained with batch normalization and achieves good training performance. At deployment, predictions are inconsistent — the same input produces different outputs on different calls. What is the most likely cause and how is it fixed?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Explain why batch normalization reduces sensitivity to weight initialization. Use what you know about how bad initialization causes vanishing or exploding activations (Note 10).

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** The running mean is updated as: μ_run ← α · μ_run + (1 − α) · μ_B, with α = 0.9. What happens to the running mean if α is set to 1.0? What if it is set to 0.0?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A single batch norm layer receives the following mini-batch of pre-activations (m = 4):

```
z = [1.0,  3.0,  5.0,  7.0]
```

The learnable parameters are γ = 3.0 and β = −1.0. Use ε = 0.

**(a)** Compute the batch mean μ_B.

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute the batch variance σ²_B.

&nbsp;

&nbsp;

&nbsp;

**(c)** Compute the four normalized values ẑ₁, ẑ₂, ẑ₃, ẑ₄. Verify that their mean is 0.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** Apply the scale and shift to obtain y₁, y₂, y₃, y₄.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** What are the mean and standard deviation of the output y? Confirm they match what γ and β specify.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** Batch normalization is described as solving internal covariate shift by stabilising each layer's input distribution. A student argues: "If we use a very small learning rate, weights change slowly, so the distributions barely shift — batch norm is unnecessary." Identify the flaw in this argument and explain why batch normalization provides value beyond what a small learning rate achieves.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Batch normalization computes statistics over a mini-batch during training. Consider what happens when the batch size is 1. Explain why batch normalization breaks down completely in this case, and what this implies about the relationship between batch size and the effectiveness of batch norm.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
