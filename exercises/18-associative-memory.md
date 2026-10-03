# Exercise 18 — Associative Memory

---

## Section A — Multiple Choice

**1.** In an auto-associative memory storing P patterns of length N using the Hebbian outer product rule, the weight matrix entry wᵢⱼ is:

- A) The average of xᵢᵐ across all stored patterns
- B) (1/N) Σₘ xᵢᵐ xⱼᵐ, with wᵢᵢ = 0
- C) (1/N) Σₘ yᵢᵐ xⱼᵐ, with no diagonal constraint
- D) The dot product of pattern i and pattern j

**2.** A hetero-associative memory stores pairs (xᵃ, yᵃ) and (xᵇ, yᵇ). Retrieval from input xᵃ is performed by:

- A) Running asynchronous updates until the network converges
- B) Computing y = sign(Wᵀ xᵃ)
- C) Computing y = sign(W xᵃ) in a single forward pass
- D) Minimising the energy function E = −½ sᵀ W s

**3.** Two patterns x¹ and x² are stored in an auto-associative memory. The crosstalk term when retrieving x¹ is proportional to:

- A) x¹ · x¹ = N
- B) x² · x¹
- C) ||x¹ − x²||²
- D) The energy of x²

**4.** The storage capacity of an associative memory built with the Hebbian outer product rule is approximately:

- A) N patterns
- B) N² patterns
- C) 0.14N patterns
- D) sqrt(N) patterns

**5.** Which of the following best describes the relationship between scaled dot-product attention and hetero-associative memory?

- A) They are unrelated — attention uses gradient descent, associative memory does not
- B) Attention is a continuous generalisation: keys are stored input patterns, values are stored output patterns, and softmax replaces the sign threshold
- C) Attention is a special case of auto-associative memory with symmetric weights
- D) Attention stores patterns using the Hebbian rule; associative memory uses backpropagation

---

## Section B — Short Answer

**1.** Explain why the diagonal of the weight matrix is zeroed in auto-associative memory but not in hetero-associative memory.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A network stores P = 20 patterns of length N = 100. Estimate the signal-to-noise ratio for retrieval and state whether reliable retrieval is expected.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Describe what happens to the error correction radius as the number of stored patterns P approaches the capacity limit P_max ≈ 0.14N.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** The Hopfield network is described as a special case of auto-associative memory. What property of the hetero-associative weight matrix is lost in the Hopfield network, and what does losing it enable?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A hetero-associative memory stores two input-output pairs with N = 5:

```
p¹ = (+1, −1, +1, −1, +1)   →   q¹ = (+1, +1, −1, +1, −1)
p² = (−1, +1, +1, −1, −1)   →   q² = (+1, −1, +1, +1, −1)
```

**(a)** Compute the weight matrix W = (1/5)(q¹(p¹)ᵀ + q²(p²)ᵀ). Show the two outer products and their sum.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Retrieve the output for input p¹. Compute net = W p¹ component by component and apply the sign threshold. Verify that the result equals q¹.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** A noisy probe p̃ = (−1, −1, +1, −1, +1) is presented (p¹ with the first component flipped). Compute net = W p̃ and show that the output still equals q¹. What does this demonstrate about the error tolerance of hetero-associative memory?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** An auto-associative memory stores 15 patterns of length N = 100. A student claims that storing the reversed patterns −x¹, −x², ..., −x¹⁵ alongside the originals would double the effective storage capacity. Evaluate this claim. What actually happens to the energy landscape when reversed patterns are added, and why does the capacity argument fail?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The attention mechanism in transformers is described as a soft, differentiable associative memory. A colleague argues that this connection is superficial — attention is trained by backpropagation while associative memory uses a one-shot Hebbian rule, so they are fundamentally different. Construct a response that identifies what is genuinely shared between the two mechanisms and what is genuinely different, and explain why the distinction matters for understanding what attention is doing.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
