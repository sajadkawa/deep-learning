# Exercise 30 — Variational Autoencoders

## Section A — Multiple Choice

**1.** Why can a standard autoencoder not generate new samples by sampling from its latent space?

- a) Its decoder is not differentiable
- b) Its latent space is unstructured — random points do not correspond to valid inputs
- c) It uses MSE loss, which prevents generation
- d) Its encoder outputs are always zero-centred

&nbsp;

**2.** In a VAE, the encoder outputs:

- a) A single latent vector $z$
- b) The mean $\mu$ and log-variance $\log\sigma^2$ of a Gaussian distribution over $z$
- c) A probability distribution over the input space
- d) The mean $\mu$ and the reconstruction $\hat{x}$

&nbsp;

**3.** The reparameterisation trick rewrites the latent sample as $z = \mu + \sigma \odot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$. What problem does this solve?

- a) It prevents the encoder from collapsing to zero variance
- b) It makes the sampling step differentiable with respect to $\mu$ and $\sigma$
- c) It ensures the decoder output is always in $[0, 1]$
- d) It replaces the KL divergence term with a simpler penalty

&nbsp;

**4.** The KL divergence term in the VAE loss penalises the encoder for:

- a) Producing reconstructions that differ from the input
- b) Using too many latent dimensions
- c) Producing posterior distributions that deviate from the standard Gaussian prior
- d) Assigning high variance to all latent dimensions

&nbsp;

**5.** After training a VAE, how is a new sample generated?

- a) Pass a training input through the encoder and decoder
- b) Sample $z \sim \mathcal{N}(0, I)$ and pass it through the decoder
- c) Sample $z$ from the approximate posterior $q_\phi(z \mid x)$ for a random $x$
- d) Interpolate between two training inputs in input space

&nbsp;

---

## Section B — Short Answer

**1.** Explain why the VAE encoder outputs $\log\sigma^2$ rather than $\sigma^2$ directly. What property does this choice ensure?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** State the VAE loss function and identify what each term measures. Why is the full loss called the negative ELBO?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Write the closed-form KL divergence between $\mathcal{N}(\mu_j, \sigma^2_j)$ and $\mathcal{N}(0, 1)$ for a single latent dimension $j$. What value does it take when $\mu_j = 0$ and $\sigma^2_j = 1$?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** What is the $\beta$-VAE and what effect does increasing $\beta$ have on the learned latent representation?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A VAE has input dimension $d = 3$ and latent dimension $k = 2$. The encoder has two linear heads: one for $\mu$ and one for $\log\sigma^2$.

**Weights for $\mu$:**

$$W_\mu = \begin{bmatrix} 0.4 & -0.2 & 0.6 \\ -0.3 & 0.5 & 0.1 \end{bmatrix}, \quad b_\mu = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}$$

**Weights for $\log\sigma^2$:**

$$W_v = \begin{bmatrix} 0.2 & 0.3 & -0.4 \\ 0.1 & -0.2 & 0.3 \end{bmatrix}, \quad b_v = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

**Input:** $x = [1.0,\ 0.5,\ -1.0]^T$. Use $e^{0.75} \approx 2.1170$, $e^{-0.30} \approx 0.7408$.

**(a)** Compute $\mu = W_\mu x + b_\mu$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute $\log\sigma^2 = W_v x + b_v$, then recover $\sigma^2$ and $\sigma$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Apply the reparameterisation trick with $\varepsilon = [0.5,\ -1.0]^T$ to obtain the latent sample $z$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** Compute the KL divergence $D_{\text{KL}}(q_\phi(z \mid x) \| p(z))$ using the closed-form formula. Sum over both latent dimensions.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** Suppose the reconstruction loss for this input is $\mathcal{L}_{\text{recon}} = 0.85$. Write the total VAE loss.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A student trains a VAE and finds that the reconstruction quality is excellent but the model cannot generate realistic new samples. They suspect the KL term is the problem. Explain what has likely happened — what failure mode causes this, and what does it imply about the latent space?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A standard autoencoder and a VAE are both trained on the same dataset with the same encoder and decoder architecture. The standard autoencoder achieves lower reconstruction loss. Does this mean the standard autoencoder is a better model? Justify your answer by comparing what each model has learned and what each can be used for.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
