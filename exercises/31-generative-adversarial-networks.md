# Exercise 31 — Generative Adversarial Networks

## Section A — Multiple Choice

**1.** In the GAN minimax objective, what does the discriminator try to maximise?

- a) The probability that generated samples are classified as real
- b) $\mathbb{E}[\log D_\phi(x)] + \mathbb{E}[\log(1 - D_\phi(G_\theta(z)))]$
- c) The reconstruction error between real and generated samples
- d) The KL divergence between the generator distribution and the data distribution

&nbsp;

**2.** Why is the non-saturating generator loss $-\log D_\phi(G_\theta(z))$ preferred over the original $\log(1 - D_\phi(G_\theta(z)))$?

- a) It is easier to compute
- b) It provides stronger gradients early in training when the generator is weak
- c) It prevents the discriminator from becoming too strong
- d) It ensures the generator loss is always positive

&nbsp;

**3.** At the Nash equilibrium of GAN training, what does the discriminator output for all inputs?

- a) 1 for real inputs and 0 for generated inputs
- b) 0 for all inputs
- c) $\frac{1}{2}$ for all inputs
- d) The probability under the data distribution

&nbsp;

**4.** Mode collapse in GAN training refers to:

- a) The discriminator loss collapsing to zero
- b) The generator producing only a small subset of the data distribution
- c) The generator and discriminator reaching equilibrium too quickly
- d) The training loss oscillating without converging

&nbsp;

**5.** Which technique directly addresses the vanishing gradient problem for the generator when the discriminator is too strong?

- a) Label smoothing on the discriminator
- b) Spectral normalisation of discriminator weights
- c) Using the non-saturating generator loss $-\log D_\phi(G_\theta(z))$
- d) Updating the discriminator more frequently than the generator

&nbsp;

---

## Section B — Short Answer

**1.** Describe the roles of the generator and discriminator in a GAN. What does each network receive as input and what does it output?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Explain why GAN training alternates between updating the discriminator and the generator, rather than updating both simultaneously with a single combined loss.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** What is label smoothing and why does it help stabilise GAN training?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** Compare GANs and VAEs as generative models. What does each model optimise, and what are the practical differences in the quality and structure of their generated outputs?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A scalar GAN has discriminator $D_\phi(x) = \sigma(w_D x + b_D)$ with $w_D = 0.8$, $b_D = 0.1$, and generator $G_\theta(z) = w_G z + b_G$ with $w_G = 0.5$, $b_G = 0.2$.

Use $e^{-1.06} \approx 0.3465$, $e^{-0.50} \approx 0.6065$, $\ln(0.7427) \approx -0.2975$, $\ln(0.3775) \approx -0.9746$, $\ln(0.6225) \approx -0.4748$.

**Real sample:** $x_{\text{real}} = 1.2$. **Noise:** $z = 0.6$.

**(a)** Compute $D_\phi(x_{\text{real}})$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute $G_\theta(z)$ and then $D_\phi(G_\theta(z))$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Compute the discriminator loss $\mathcal{L}_D = -[\log D_\phi(x_{\text{real}}) + \log(1 - D_\phi(G_\theta(z)))]$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** Compute the non-saturating generator loss $\mathcal{L}_G = -\log D_\phi(G_\theta(z))$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** The discriminator is updated to increase $D_\phi(x_{\text{real}})$ and decrease $D_\phi(G_\theta(z))$. After one update step, suppose $D_\phi(x_{\text{real}}) = 0.85$ and $D_\phi(G_\theta(z)) = 0.45$. Recompute $\mathcal{L}_D$ and $\mathcal{L}_G$. Use $\ln(0.85) \approx -0.1625$, $\ln(0.55) \approx -0.5978$, $\ln(0.45) \approx -0.7985$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A GAN is trained on a dataset of handwritten digits (10 classes). After training, the generator produces only digits that look like the digit "1", regardless of the noise input $z$. Identify the failure mode, explain why it occurs in terms of the adversarial objective, and suggest one concrete technique to mitigate it.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The theoretical Nash equilibrium of GAN training requires $p_{G_\theta} = p_{\text{data}}$. In practice, this equilibrium is rarely reached. Explain two reasons why gradient-based optimisation of the minimax objective fails to converge to the Nash equilibrium in practice.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
