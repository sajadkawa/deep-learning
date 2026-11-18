# Exercise 29 — Autoencoders

## Section A — Multiple Choice

**1.** An autoencoder is trained by minimising the difference between:

- a) The input and the latent code
- b) The input and the reconstructed output
- c) The latent code and a target label
- d) The encoder output and the decoder input

&nbsp;

**2.** What is the primary purpose of the bottleneck in an undercomplete autoencoder?

- a) To speed up training by reducing the number of parameters
- b) To prevent the network from learning the identity function by forcing compression
- c) To apply dropout regularisation to the latent layer
- d) To ensure the decoder has fewer parameters than the encoder

&nbsp;

**3.** A sparse autoencoder with an overcomplete bottleneck ($k > d$) adds which penalty to prevent a trivial solution?

- a) L2 penalty on the weights
- b) L1 penalty on the latent activations
- c) Dropout on the encoder output
- d) KL divergence between input and output distributions

&nbsp;

**4.** In a denoising autoencoder, the reconstruction loss is computed between:

- a) The corrupted input $\tilde{x}$ and the reconstruction $\hat{x}$
- b) The clean input $x$ and the corrupted input $\tilde{x}$
- c) The clean input $x$ and the reconstruction $\hat{x} = g_\theta(f_\phi(\tilde{x}))$
- d) The latent code $z$ and a zero vector

&nbsp;

**5.** An autoencoder trained only on normal data is used for anomaly detection. Which quantity serves as the anomaly score?

- a) The L2 norm of the latent code
- b) The reconstruction error $\| x - \hat{x} \|^2$
- c) The entropy of the latent activations
- d) The gradient of the loss with respect to the input

&nbsp;

---

## Section B — Short Answer

**1.** Explain why an overcomplete autoencoder (bottleneck size $k \geq d$) can learn the identity function without learning any useful representation. What constraint prevents this?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** State the MSE reconstruction loss formula and explain what it measures geometrically.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Describe the difference between L1 sparsity and KL divergence sparsity penalties in a sparse autoencoder. What does each one control?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** Why is a denoising autoencoder considered a form of self-supervised learning? What plays the role of the supervision signal?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A single-layer autoencoder has input dimension $d = 3$ and bottleneck dimension $k = 2$. The encoder uses tanh activation; the decoder uses a linear output.

**Encoder weights and bias:**

$$W^{(e)} = \begin{bmatrix} 0.5 & -0.3 & 0.8 \\ -0.2 & 0.6 & 0.4 \end{bmatrix}, \quad b^{(e)} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$

**Decoder weights and bias:**

$$W^{(d)} = \begin{bmatrix} 0.7 & -0.4 \\ 0.3 & 0.9 \\ -0.5 & 0.2 \end{bmatrix}, \quad b^{(d)} = \begin{bmatrix} 0.0 \\ 0.1 \\ -0.1 \end{bmatrix}$$

**Input:** $x = [1.0,\ 0.5,\ -1.0]^T$

Use $e^{0.35} \approx 1.4191$, $e^{0.40} \approx 1.4918$.

**(a)** Compute the encoder pre-activations $z^{(e)}_1$ and $z^{(e)}_2$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Apply tanh to obtain the latent code $z = [z_1, z_2]^T$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Compute the reconstruction $\hat{x} = W^{(d)} z + b^{(d)}$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** Compute the MSE reconstruction loss $\mathcal{L}(x, \hat{x})$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** Now suppose a sparse autoencoder adds an L1 penalty with $\lambda = 0.5$. Compute the total loss $\mathcal{L}_{\text{sparse}}$ using the latent code from part (b).

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A colleague proposes training an autoencoder with a bottleneck of size $k = d$ (same as the input dimension) and no regularisation, arguing that the network will still learn a useful representation because it must pass through a single layer. Identify the flaw in this argument and explain what the network is likely to learn instead.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Autoencoders and PCA both perform dimensionality reduction. A practitioner claims that a linear autoencoder (no nonlinear activations) with MSE loss is equivalent to PCA. Is this claim correct? If so, under what conditions? If not, what does the linear autoencoder learn instead, and how does it differ from PCA?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
