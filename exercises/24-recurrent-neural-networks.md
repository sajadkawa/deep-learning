# Exercise 24 — Recurrent Neural Networks

---

## Section A — Multiple Choice

**1.** Why can a feedforward network not directly process a variable-length sequence?

- A) It cannot use tanh as an activation function
- B) Its input layer has a fixed size, so sequences of different lengths cannot be fed in without padding or truncation
- C) It has too many parameters to process sequential data efficiently
- D) It applies the same weights at every position, which prevents it from learning order

&nbsp;

**2.** In the RNN update equation $h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)$, what is the role of $W_h$?

- A) It maps the current input into hidden space
- B) It maps the previous hidden state into hidden space, allowing information to persist across time steps
- C) It maps the hidden state to the output vocabulary
- D) It initialises the hidden state before the sequence begins

&nbsp;

**3.** An RNN processes sequences of length 10 and sequences of length 100. How does the number of trainable parameters change between the two cases?

- A) The 100-step RNN has 10× more parameters because it processes 10× more steps
- B) The number of parameters is the same — weights are shared across all time steps
- C) The 100-step RNN has fewer parameters because weight sharing is more aggressive
- D) The number of parameters depends on the hidden size, not the sequence length

&nbsp;

**4.** In a character-level RNN with vocabulary size $V$ and hidden size $d$, the input at each step is a one-hot vector. What does multiplying this one-hot vector by $W_x \in \mathbb{R}^{d \times V}$ compute?

- A) The dot product between the input and all hidden states seen so far
- B) A weighted average of all character embeddings
- C) The column of $W_x$ corresponding to the active character — a learned character embedding
- D) The softmax probability distribution over the next character

&nbsp;

**5.** A vanilla RNN is trained on sentences of average length 50 tokens. It performs well on short-range dependencies (e.g. subject–verb agreement within 3 tokens) but fails on long-range dependencies (e.g. pronoun resolution across 30 tokens). What is the most direct cause of this failure?

- A) The hidden size is too small to store information about 50 tokens
- B) The learning rate is too high, causing the weights to oscillate
- C) Gradients vanish as they are backpropagated through 30 tanh multiplications, so the loss signal from the distant token reaches near zero
- D) The output weight matrix $W_y$ is not updated during backpropagation

&nbsp;

---

## Section B — Short Answer

**1.** An RNN has three weight matrices: $W_x$, $W_h$, and $W_y$. State the shape of each matrix in terms of $n$ (input size), $d$ (hidden size), and $K$ (output classes), and explain in one sentence what each matrix does.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Explain what truncated BPTT is and why it is used in practice. What does it trade off?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** The vanishing gradient problem exists in both deep feedforward networks and RNNs, but the cause is more severe in RNNs. Explain why sequence length makes the problem worse in RNNs compared to depth in feedforward networks.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** A many-to-many (synchronised) RNN and a many-to-many encoder-decoder RNN both take a sequence as input and produce a sequence as output. What is the key structural difference between them, and what does that difference allow the encoder-decoder to do that the synchronised model cannot?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A minimal RNN has scalar inputs ($n = 1$), hidden size $d = 2$, and no output layer. The weights and initial hidden state are:

$$W_x = \begin{bmatrix} 0.6 \\ -0.4 \end{bmatrix}, \quad W_h = \begin{bmatrix} 0.3 & 0.2 \\ -0.1 & 0.5 \end{bmatrix}, \quad b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad h_0 = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Input sequence: $x_1 = 1.0,\ x_2 = -1.0$.

Use $\tanh(0.6) \approx 0.5370$, $\tanh(-0.4) \approx -0.3799$, and compute subsequent values from scratch.

**(i)** Compute $z_1 = W_h h_0 + W_x x_1 + b_h$. Show the full matrix arithmetic.

&nbsp;

&nbsp;

&nbsp;

**(ii)** Compute $h_1 = \tanh(z_1)$.

&nbsp;

&nbsp;

&nbsp;

**(iii)** Compute $z_2 = W_h h_1 + W_x x_2 + b_h$. Show each term separately before summing.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(iv)** Compute $h_2 = \tanh(z_2)$ (round to 4 decimal places).

&nbsp;

&nbsp;

&nbsp;

**(v)** The hidden state $h_2$ encodes the entire sequence $(1.0,\ -1.0)$. Compare $h_2$ to $h_1$. What has the second input $x_2 = -1.0$ done to the hidden state, and what does this illustrate about how the RNN accumulates information?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A student proposes replacing the hidden-to-hidden weight matrix $W_h$ with a fixed identity matrix $I$ (not learned) to reduce the number of parameters. The input-to-hidden matrix $W_x$ and output matrix $W_y$ would still be learned. Analyse this proposal: what would the network still be able to do, what would it lose, and is the result still meaningfully recurrent?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Weight sharing across time steps is presented as a strength of RNNs — it keeps the parameter count fixed regardless of sequence length. But weight sharing also imposes a constraint. Describe a type of sequence task where this constraint is a limitation, and explain precisely why the shared weights cannot capture what the task requires.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
