# Exercise 03 — Multilayer Perceptron (MLP)

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** In an MLP with 2 inputs, 2 hidden neurons, and 1 output neuron (as used in the XOR example), how many total learnable parameters (weights plus biases) does the network contain?

- a) 6 parameters
- b) 7 parameters
- c) 9 parameters
- d) 11 parameters

---

**Q2.** An MLP has 10 input nodes, one hidden layer with 50 neurons, a second hidden layer with 20 neurons, and an output layer with 2 neurons. Under the standard deep learning layer-counting convention, how many computational layers does this network have, and what is the total number of weight matrices?

- a) 4 layers, 4 weight matrices
- b) 3 layers, 3 weight matrices
- c) 4 layers, 3 weight matrices
- d) 3 layers, 4 weight matrices

---

**Q3.** An MLP has an input dimension of n = 4 features, a hidden layer of m = 8 neurons, and an output layer of k = 1 neuron. Using the weight convention wᵢⱼ (source i → destination j), what are the shapes of the transposed weight matrix W⁽¹⁾ᵀ and pre-activation vector z⁽¹⁾ in the equation z⁽¹⁾ = W⁽¹⁾ᵀx + b⁽¹⁾?

- a) W⁽¹⁾ᵀ shape (4×8),   z⁽¹⁾ shape (4×1)
- b) W⁽¹⁾ᵀ shape (8×4),   z⁽¹⁾ shape (8×1)
- c) W⁽¹⁾ᵀ shape (8×4),   z⁽¹⁾ shape (4×1)
- d) W⁽¹⁾ᵀ shape (4×8),   z⁽¹⁾ shape (8×1)

---

**Q4.** What does the Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991) guarantee?

- a) Gradient descent is guaranteed to find the globally optimal weights for any 1-hidden-layer network
- b) Any continuous function on a compact domain can be approximated to arbitrary precision by a 1-hidden-layer network with a finite number of neurons
- c) A 1-hidden-layer network is always more parameter-efficient than a deep network
- d) Any discontinuous function can be approximated without overfitting

---

**Q5.** Why does the classical Perceptron Learning Rule (wᵢ ← wᵢ + η(y - ŷ)xᵢ) fail when applied to train the hidden layers of an MLP?

- a) Hidden neurons do not have bias terms
- b) The step function cannot be computed by hidden neurons
- c) Hidden layers have no direct output error signal, leading to the credit assignment problem
- d) The learning rate η must be zero for hidden layers

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** In Note 03, we observed that an MLP solves XOR by transforming the four input points into a new coordinate space (h₁, h₂). Explain how this feature space warping makes an originally non-linearly separable problem linearly separable. In your explanation, specify what coordinates the points (0, 1) and (1, 0) are mapped to in hidden space.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** An MLP is configured with n = 784 input features (e.g. 28×28 grayscale images), a first hidden layer of m₁ = 128 neurons, a second hidden layer of m₂ = 64 neurons, and an output layer of k = 10 classes.
- **(a)** State the dimensions of the weight matrix and bias vector for each of the three computational layers (using the source i → destination j convention).
- **(b)** Calculate the exact number of learnable parameters (weights and biases) in each layer, and determine the total parameter count for the entire network. Show your arithmetic.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** A student reads the Universal Approximation Theorem and argues: *"Since a single hidden layer can approximate any continuous function, developing deep neural networks with 50 or 100 layers is an unnecessary complication."* Refute this argument. Give two distinct reasons why deep architectures are preferred over arbitrarily wide shallow ones.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** Describe the **Credit Assignment Problem** in the context of multilayer neural networks. Why did this problem contribute to the historical "AI Winter" of the 1970s–1980s, and what three mathematical components were required to resolve it? For each component, state what breaks if it is absent.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** Consider a 2-layer MLP with 2 inputs, 2 hidden neurons, and 1 output neuron configured as follows:

```
Weight Matrices and Biases (using source i → destination j indexing):

Layer 1 (Input → Hidden):
  W⁽¹⁾ = ⎡  2   -1 ⎤      b⁽¹⁾ = ⎡ -1 ⎤
         ⎣ -1    2 ⎦             ⎣ -1 ⎦

Layer 2 (Hidden → Output):
  W⁽²⁾ = ⎡ 1.5 ⎤          b⁽²⁾ = -1.0
         ⎣ 1.5 ⎦

Activation function for all neurons:
  ReLU: f(z) = max(0, z)
```

**(a)** For the input vector x = [1, 0]ᵀ (where x₁ = 1, x₂ = 0), compute:
1. The pre-activation vector z⁽¹⁾ = W⁽¹⁾ᵀx + b⁽¹⁾
2. The hidden activation vector h = f(z⁽¹⁾)
3. The output pre-activation z⁽²⁾ = W⁽²⁾ᵀh + b⁽²⁾
4. The final prediction ŷ = f(z⁽²⁾)

**(b)** Repeat the calculation for the input vector x = [1, 1]ᵀ (where x₁ = 1, x₂ = 1).

**(c)** Based on your calculations in (a) and (b), determine whether hidden neuron h₁ was active (fired) or inactive (dead) for each input. Explain why ReLU's output of zero for negative inputs produces sparse representations.

**(d)** Using the network above and the input x = [1, 0]ᵀ with true label y = 1 and MSE loss L = (y − ŷ)², write out the full chain rule expansion for ∂L/∂w⁽¹⁾₁₁ (the gradient of the loss with respect to the weight from x₁ to h₁). Substitute the numerical values you computed in part (a) into each factor. Identify which factors the Perceptron Learning Rule has no access to and explain why.

---

## Section D — Critical Thinking

**Q11.** In modern deep learning research, theoretical papers frequently study **Deep Linear Networks** (networks with multiple layers but no non-linear activation functions). If a deep linear network can compute nothing beyond what a single linear model computes, why would theorists find value in analyzing them? What properties do deep linear networks share with real non-linear neural networks that shallow linear models do not have? *(Hint: Consider the loss function landscape with respect to weights, and how gradient descent dynamics behave across factorized matrices).*

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** Consider a 2-layer autoencoder with n = 100 inputs, a hidden layer of m = 2 neurons, and k = 100 output neurons, using linear activation functions.
- **(a)** Geometrically, what does this network do to the 100-dimensional input data?
- **(b)** Can this network achieve zero reconstruction error for an arbitrary set of 100-dimensional input points? Why or why not?
- **(c)** How does this relate to the concept of **representation learning** and classical dimensionality reduction techniques such as Principal Component Analysis (PCA)?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q13.** The three pillars of modern neural network training — a continuous loss function, differentiable activations, and backpropagation — must all be present simultaneously. For each of the following broken configurations, explain precisely what fails and why training cannot proceed:

- **(a)** A network with a smooth differentiable loss and ReLU activations, but no backpropagation algorithm — the engineer instead tries to estimate gradients by randomly perturbing each weight one at a time.
- **(b)** A network with backpropagation and a smooth loss, but with Heaviside step activations in all hidden layers.
- **(c)** A network with backpropagation and ReLU activations, but where the output layer uses a hard threshold (0 or 1) instead of a continuous prediction.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
