# Exercise 26 — Encoder-Decoder Architectures

---

## Section A — Multiple Choice

**1.** Why can a single many-to-many RNN not directly solve machine translation?

- A) RNNs cannot process text — they only handle numerical sequences
- B) A single RNN produces one output per input step, forcing the output length to equal the input length
- C) RNNs do not support different vocabularies for input and output
- D) The hidden state is too small to store information about more than one language

&nbsp;

**2.** In an encoder-decoder architecture, what is the context vector $c$?

- A) The average of all encoder hidden states across the input sequence
- B) The encoder's final hidden state after processing the entire input sequence
- C) A learned embedding of the start-of-sequence token
- D) The concatenation of the encoder's first and last hidden states

&nbsp;

**3.** During training with teacher forcing, what does the decoder receive as input at each step?

- A) The encoder's hidden state at the corresponding input position
- B) Its own prediction from the previous step
- C) The ground truth output token from the previous step
- D) A zero vector at every step except the first

&nbsp;

**4.** An encoder-decoder model is trained on sentences of up to 20 tokens and then evaluated on sentences of 80 tokens. Performance degrades significantly on the longer sentences. What is the most direct cause?

- A) The decoder RNN has too few parameters to generate long sequences
- B) The fixed-size context vector cannot faithfully compress 80 tokens — information is lost at the bottleneck
- C) Teacher forcing was used during training, so the model cannot handle long sequences at inference
- D) The encoder was not trained on sequences longer than 20 tokens, so its weights are incorrect

&nbsp;

**5.** In image captioning, a CNN replaces the encoder RNN. What does the CNN's output play the role of?

- A) The decoder's hidden state at every step
- B) The input sequence fed to the decoder one token at a time
- C) The context vector — a fixed-size summary of the image passed to the decoder as its initial state
- D) The output vocabulary distribution over image regions

&nbsp;

---

## Section B — Short Answer

**1.** Explain the train-test mismatch introduced by teacher forcing. Why does teacher forcing help during training, and what problem does it create at inference time?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The encoder and decoder are described as "separate networks with separate weights." Why is it important that they have separate weight matrices rather than sharing the same RNN weights?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** In a standard encoder-decoder, the decoder has no direct access to the encoder's intermediate hidden states $h_1^{(e)}, \ldots, h_{T-1}^{(e)}$ — only the final state $c = h_T^{(e)}$ is passed. Explain precisely why this is a problem for long input sequences, using an information-theoretic argument.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** The decoder generates output tokens until it produces an $\langle\text{EOS}\rangle$ token. What would happen if the model never learned to produce $\langle\text{EOS}\rangle$? How is the model trained to produce it at the right time?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

An encoder-decoder has hidden size $d = 2$, scalar inputs ($n = 1$), and scalar outputs. The encoder uses the weights from Note 24 (already computed). The decoder has the following weights:

$$W_h^{(d)} = \begin{bmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{bmatrix}, \quad W_y^{(d)} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}, \quad b_h^{(d)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad W_o = \begin{bmatrix} 1.0 & 1.0 \end{bmatrix}$$

The encoder processes input sequence $x_1 = 1.0,\ x_2 = 0.5,\ x_3 = -1.0$ and produces:

$$c = h_3^{(e)} = \begin{bmatrix} -0.3685 \\ -0.0137 \end{bmatrix}$$

The decoder start token is $y_0 = 0.0$.

**(i)** The decoder is initialised with $h_0^{(d)} = c$. Compute $z_1^{(d)} = W_h^{(d)} h_0^{(d)} + W_y^{(d)} y_0 + b_h^{(d)}$. Since $W_y^{(d)} = \mathbf{0}$, the input term vanishes — show the matrix multiplication explicitly.

&nbsp;

&nbsp;

&nbsp;

**(ii)** Compute $h_1^{(d)} = \tanh(z_1^{(d)})$ and $\hat{y}_1 = W_o h_1^{(d)}$. Use $\tanh(-0.1843) \approx -0.1829$ and $\tanh(-0.0069) \approx -0.0069$.

&nbsp;

&nbsp;

&nbsp;

**(iii)** Compute $z_2^{(d)} = W_h^{(d)} h_1^{(d)} + W_y^{(d)} \hat{y}_1 + b_h^{(d)}$. Again the input term vanishes. Show the matrix multiplication.

&nbsp;

&nbsp;

&nbsp;

**(iv)** Compute $h_2^{(d)} = \tanh(z_2^{(d)})$ and $\hat{y}_2 = W_o h_2^{(d)}$ (round to 4 decimal places).

&nbsp;

&nbsp;

&nbsp;

**(v)** The decoder weight $W_h^{(d)} = 0.5 \cdot I$ (half the identity matrix). What does this mean for how the decoder's hidden state evolves over time? What would happen to $h_t^{(d)}$ as $t \to \infty$ if the input term remained zero?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A student proposes a "shortcut" encoder-decoder: instead of using the encoder's final hidden state as the context vector, use the **average** of all encoder hidden states: $c = \frac{1}{T} \sum_{t=1}^{T} h_t^{(e)}$. Analyse this proposal — what information does it preserve that the final hidden state loses, and what information does it lose that the final hidden state preserves?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The bottleneck problem is described as a fundamental limitation of the encoder-decoder architecture. A colleague argues: "The solution is simple — just increase the hidden size $d$. A larger context vector can store more information." Evaluate this argument. Is it correct in principle? Why does it fail in practice?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
