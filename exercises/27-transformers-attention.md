# Exercise 27 — Transformers and Attention Mechanism

---

## Section A — Multiple Choice

**1.** In scaled dot-product attention, why is the dot product $QK^T$ divided by $\sqrt{d_k}$ before the softmax?

- A) To normalise the output values to the range $[0, 1]$
- B) To prevent large dot products from pushing the softmax into saturation, where gradients become very small
- C) To ensure the attention weights sum to $d_k$ rather than 1
- D) To reduce the computational cost of the matrix multiplication

&nbsp;

**2.** In the Transformer encoder, the multi-head self-attention sub-layer uses queries, keys, and values that all come from the same sequence. What does this allow each position to do?

- A) Predict the next token in the sequence
- B) Attend to all other positions in the same sequence, capturing relationships between any two tokens regardless of distance
- C) Copy information from the encoder output into the decoder
- D) Apply a different weight matrix to each position independently

&nbsp;

**3.** Why does the Transformer require positional encoding, while an RNN does not?

- A) The Transformer uses larger weight matrices that cannot represent position implicitly
- B) The RNN processes tokens sequentially and encodes position through the hidden state; the Transformer processes all positions in parallel and treats the input as a set with no inherent order
- C) Positional encoding is needed to prevent the softmax from producing uniform attention weights
- D) The Transformer's feed-forward network removes positional information, which must be re-added

&nbsp;

**4.** In the Transformer decoder, the first sub-layer uses masked self-attention. What does the mask prevent, and why is it necessary?

- A) It prevents the decoder from attending to the encoder output, forcing it to rely only on its own state
- B) It prevents position $t$ from attending to positions after $t$, ensuring the decoder cannot see future tokens during training
- C) It prevents the attention weights from becoming uniform, forcing the decoder to focus on one position
- D) It prevents the positional encoding from interfering with the attention computation

&nbsp;

**5.** An RNN processes a sequence of length $T = 100$. A Transformer processes the same sequence. Comparing the two architectures, which statement is correct?

- A) The RNN requires $O(T^2)$ operations; the Transformer requires $O(T)$
- B) The Transformer requires $O(T^2)$ memory for attention; the RNN requires $O(T)$ — the Transformer's quadratic cost is its main disadvantage for long sequences
- C) Both require $O(T)$ memory and $O(T)$ sequential operations
- D) The Transformer requires more sequential operations than the RNN because it has more layers

&nbsp;

---

## Section B — Short Answer

**1.** Explain the query-key-value abstraction in attention. What does each of the three matrices represent conceptually, and how do they interact to produce the output?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Multi-head attention uses $H$ parallel attention heads, each operating in a $d_k = d_{\text{model}}/H$ dimensional subspace. Explain why this is preferable to a single attention head operating in the full $d_{\text{model}}$ dimensional space, even though the total computation is similar.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** The Transformer encoder uses self-attention while the Transformer decoder uses both masked self-attention and cross-attention. Explain the distinct role of each of these three attention operations.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** The note states that in the Transformer, the gradient path length between any two positions is $O(1)$, compared to $O(T)$ in an RNN. Explain what this means structurally — what is the actual path a gradient takes from position $T$ back to position $1$ in each architecture?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

Scaled dot-product attention with $d_k = d_v = 2$, two key-value positions, one query.

$$q = \begin{bmatrix} 1.0 & 0.0 \end{bmatrix}, \quad K = V = \begin{bmatrix} 0.8 & 0.2 \\ 0.2 & 0.8 \end{bmatrix}$$

Use $e^{0.566} \approx 1.761$, $e^{0.141} \approx 1.151$. Round to 3 decimal places throughout.

**(i)** Compute the raw scores $s = qK^T$. Show the dot product with each key explicitly.

&nbsp;

&nbsp;

&nbsp;

**(ii)** Scale the scores: $s_{\text{scaled}} = s / \sqrt{d_k}$. Show the value of $\sqrt{d_k}$ and the scaled scores.

&nbsp;

&nbsp;

&nbsp;

**(iii)** Compute the attention weights $\alpha = \text{softmax}(s_{\text{scaled}})$. Show the exponentials, their sum, and the final weights.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(iv)** Compute the attended output $\text{out} = \alpha_1 v_1 + \alpha_2 v_2$. Show each weighted value vector before summing.

&nbsp;

&nbsp;

&nbsp;

**(v)** The query $q = [1.0, 0.0]$ is more similar to the first key $[0.8, 0.2]$ than to the second key $[0.2, 0.8]$. Confirm this by comparing the two raw scores from part (i). Then explain what the output vector from part (iv) represents — why is it not simply equal to the first value vector?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** The Transformer's self-attention allows every position to attend to every other position in $O(1)$ steps. An RNN achieves the same by processing sequentially — position $T$ has seen all previous positions through the hidden state. A student argues: "Both architectures can in principle capture any dependency, so the Transformer has no fundamental advantage over an LSTM for long-range dependencies." Identify the flaw in this argument.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Positional encoding adds a fixed signal to the input embeddings. A colleague proposes learning the positional encodings as parameters (a learned embedding table indexed by position) rather than using the fixed sinusoidal formula. Identify one advantage and one disadvantage of learned positional encodings compared to sinusoidal encodings.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
