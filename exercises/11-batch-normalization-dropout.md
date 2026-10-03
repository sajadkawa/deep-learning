# Exercise 11 — Batch Normalization and Dropout

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

**4.** A network uses batch normalization throughout and large fully connected layers. You add dropout with rate 0.5 after the batch norm layers. What problem is most likely to occur?

- a) The network will underfit because too many neurons are removed
- b) Gradient descent will diverge because dropout interferes with backpropagation
- c) Variance shift — the running statistics accumulated during training will not match the inference distribution
- d) The learnable parameters γ and β will be driven to zero

&nbsp;

**5.** Where should dropout never be applied?

- a) After the first hidden layer
- b) In fully connected layers with more than 512 neurons
- c) In the output layer
- d) In layers that use ReLU activation

&nbsp;

---

## Section B — Short Answer

**1.** Explain why batch normalization acts as a mild regularizer, even though it was not designed for that purpose.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The original batch normalization paper places batch norm before the activation function. Explain the reasoning behind this placement for sigmoid and tanh activations.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Dropout was described in Note 07 as equivalent to training an ensemble of 2ⁿ sub-networks. Why does this ensemble interpretation break down when dropout is combined with batch normalization?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** A network is trained with batch normalization and evaluated with `model.train()` still active instead of `model.eval()`. Describe what goes wrong and why the outputs are non-deterministic.

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

**1.** Batch normalization is said to reduce the network's sensitivity to weight initialization. Using what you know about how bad initialization causes vanishing or exploding activations (Note 10), explain the mechanism by which batch normalization provides this protection — and identify one situation where it would not be sufficient.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A colleague argues: "Batch normalization already regularizes the network through mini-batch noise, so dropout is redundant and should never be used alongside it." Construct a specific scenario where this argument fails — where using dropout alongside batch normalization is justified despite the variance shift problem.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
