# Exercise 25 — LSTM and GRU

---

## Section A — Multiple Choice

**1.** What is the fundamental structural reason why a vanilla RNN cannot learn long-range dependencies, even with sufficient hidden size?

- A) The hidden state vector is too small to store information about distant tokens
- B) The hidden state is overwritten at every step through a tanh nonlinearity, so information from distant steps is compressed exponentially
- C) The weight matrix $W_h$ is not updated during backpropagation for early time steps
- D) The softmax output layer discards information from the hidden state

&nbsp;

**2.** In an LSTM, the cell state update is $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. Why does this allow gradients to flow back through many time steps without vanishing?

- A) The sigmoid gates produce values greater than 1, amplifying the gradient
- B) The cell state is reset to zero at each step, preventing gradient accumulation
- C) The update involves only elementwise multiplications of $c_{t-1}$ — no matrix multiplication or tanh is applied directly to it, so gradients are not repeatedly squashed
- D) The forget gate always outputs 1, ensuring the full gradient passes through

&nbsp;

**3.** An LSTM forget gate outputs $f_t = [0.02, 0.95]^T$ at time step $t$. What does this tell you about what the network has learned?

- A) The network is writing new information into both dimensions of the cell state
- B) The network is erasing almost all of dimension 1 of the cell state and preserving almost all of dimension 2
- C) The network is blocking the input gate from writing to the cell state
- D) The network is preventing the hidden state from being exposed as output

&nbsp;

**4.** In a GRU, the hidden state update is $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. What happens when $z_t \approx 0$ for all dimensions?

- A) The hidden state is replaced entirely by the new candidate $\tilde{h}_t$
- B) The hidden state is set to zero
- C) The hidden state is copied forward unchanged from $h_{t-1}$ — long-range information is preserved
- D) The reset gate takes over and recomputes the hidden state from scratch

&nbsp;

**5.** An LSTM with hidden size $d = 128$ and input size $n = 64$ is compared to a GRU with the same $d$ and $n$. Approximately how many fewer parameters does the GRU have in its recurrent cell?

- A) About 25% fewer — the GRU has $3/4$ the parameters of the LSTM
- B) About 50% fewer — the GRU has half the parameters of the LSTM
- C) The same number — both models use the same weight matrices
- D) About 10% fewer — the GRU only removes the output gate

&nbsp;

---

## Section B — Short Answer

**1.** The LSTM has two state vectors: $h_t$ and $c_t$. Explain the distinct role of each. Why is the cell state described as a "highway" and what property of the cell state update makes this analogy accurate?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Explain what the reset gate $r_t$ does in a GRU. Give a concrete example of a sequence processing scenario where $r_t \approx 0$ would be the correct behaviour.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** The LSTM forget gate bias $b_f$ is often initialised to a positive value (e.g. 1.0) rather than zero. Explain why this initialisation strategy makes sense given the role of the forget gate.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** Both the LSTM and GRU solve the vanishing gradient problem, but through slightly different mechanisms. Identify the specific path in each architecture through which gradients can flow without vanishing, and explain what structural property of that path prevents the gradient from shrinking.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A minimal LSTM has scalar inputs ($n = 1$), hidden size $d = 2$, and initial states $h_0 = c_0 = \mathbf{0}$. The input is $x_1 = 1.0$. Since $h_0 = \mathbf{0}$, all $W \cdot h_0$ terms vanish — each gate reduces to $U \cdot x_1 + b$.

The weight vectors and biases are:

$$U_f = \begin{bmatrix} 0.4 \\ 0.3 \end{bmatrix},\ b_f = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}, \quad U_i = \begin{bmatrix} 0.8 \\ 0.6 \end{bmatrix},\ b_i = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$U_c = \begin{bmatrix} 0.5 \\ 0.7 \end{bmatrix},\ b_c = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad U_o = \begin{bmatrix} 0.6 \\ 0.5 \end{bmatrix},\ b_o = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Use: $\sigma(-0.1) \approx 0.4750$, $\sigma(-0.2) \approx 0.4502$, $\sigma(0.6) \approx 0.6457$, $\sigma(0.8) \approx 0.6900$, $\tanh(0.5) \approx 0.4621$, $\tanh(0.7) \approx 0.6044$. Compute remaining values from scratch (round to 4 decimal places).

**(i)** Compute the forget gate $f_1 = \sigma(U_f x_1 + b_f)$. Show the pre-activation vector before applying $\sigma$.

&nbsp;

&nbsp;

&nbsp;

**(ii)** Compute the input gate $i_1 = \sigma(U_i x_1 + b_i)$ and the candidate $\tilde{c}_1 = \tanh(U_c x_1 + b_c)$.

&nbsp;

&nbsp;

&nbsp;

**(iii)** Compute the cell state $c_1 = f_1 \odot c_0 + i_1 \odot \tilde{c}_1$. Show each elementwise product explicitly.

&nbsp;

&nbsp;

&nbsp;

**(iv)** Compute the output gate $o_1 = \sigma(U_o x_1 + b_o)$, then compute $h_1 = o_1 \odot \tanh(c_1)$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(v)** The forget gate values from part (i) are both close to 0.5. What would happen to $c_1$ if the forget gate had instead output $f_1 = [1.0, 1.0]^T$? What would happen if it had output $f_1 = [0.0, 0.0]^T$? What does this tell you about the role of the forget gate bias?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A student claims: "The LSTM is strictly better than the GRU because it has more parameters and a separate cell state, so it can always learn what the GRU can learn and more." Evaluate this claim carefully. Under what conditions is it true, and under what conditions does it break down?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The GRU update gate $z_t$ controls both how much of the old hidden state to keep and how much of the new candidate to write — it plays the combined role of the LSTM's forget gate and input gate. Identify a specific type of sequence where having independent forget and input gates (as in the LSTM) would be advantageous over the GRU's single update gate, and explain precisely why the coupling in the GRU is a limitation for that case.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
