# Exercise 28 — Attention and Memory Models

---

## Section A — Multiple Choice

**1.** Why is soft attention trainable with standard backpropagation while hard attention is not?

- A) Soft attention uses sigmoid activations; hard attention uses ReLU, which has zero gradient for negative inputs
- B) Soft attention computes a weighted average — a smooth, differentiable function of the scores; hard attention uses $\arg\max$, which has zero gradient almost everywhere
- C) Soft attention is applied only at the output layer; hard attention is applied at every layer, causing gradient explosion
- D) Hard attention requires more memory than backpropagation can handle

&nbsp;

**2.** In self-attention, queries, keys, and values all come from the same sequence. What does this allow the model to compute that cross-attention cannot?

- A) The probability distribution over the output vocabulary
- B) Context-dependent representations — each position's output encodes its relationship to every other position in the same sequence
- C) The gradient of the loss with respect to the input embeddings
- D) A fixed positional encoding for each token in the sequence

&nbsp;

**3.** A memory-augmented network has a controller with hidden size $d = 128$ and an external memory $M \in \mathbb{R}^{50 \times 128}$. How many numbers can the external memory store compared to the hidden state alone?

- A) The same — both store 128 numbers
- B) 50 times more — the memory stores 50 independent 128-dimensional vectors
- C) 128 times more — each memory slot stores 128 copies of the hidden state
- D) Fewer — the memory must share capacity with the hidden state

&nbsp;

**4.** In the Neural Turing Machine, content-based addressing attends to memory slots based on similarity to a query key. What does location-based addressing add that content-based addressing alone cannot provide?

- A) The ability to retrieve multiple memory slots simultaneously
- B) The ability to move sequentially through memory — accessing the next or previous slot regardless of content
- C) The ability to erase memory slots that are no longer needed
- D) The ability to increase the sharpening parameter $\beta_t$ during training

&nbsp;

**5.** The modern Hopfield network (Ramsauer et al., 2020) showed that one step of attention is equivalent to one Hopfield retrieval update. In this correspondence, what do the keys $K$ represent?

- A) The query vectors used to retrieve patterns from memory
- B) The stored patterns in the Hopfield network
- C) The energy function that the network minimises during retrieval
- D) The output layer weights that map hidden states to predictions

&nbsp;

---

## Section B — Short Answer

**1.** Explain the tradeoff between soft and hard attention in terms of interpretability and trainability. Give one application scenario where hard attention would be preferred despite its training complexity.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A standard LSTM has hidden size $d = 256$. A memory-augmented network has the same controller but adds an external memory $M \in \mathbb{R}^{100 \times 256}$. Explain two specific advantages the memory-augmented network has over the plain LSTM for a task that requires storing and retrieving many independent facts.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** The NTM write operation is $M_i \leftarrow M_i(1 - \alpha_i e_t) + \alpha_i a_t$. Explain what happens to memory slot $M_i$ when (a) $\alpha_i \approx 1$ and $e_t = \mathbf{1}$, and (b) $\alpha_i \approx 0$. What does this tell you about the role of the attention weights in the write operation?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** The note states that attention is "associative memory made differentiable." Explain this claim by mapping the components of soft attention (query, keys, values, weights, output) onto the components of a hetero-associative memory system from Note 18.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A memory-augmented network has external memory with $N = 3$ slots and slot dimension $d = 2$:

$$M = \begin{bmatrix} 1.0 & 0.0 \\ 0.5 & 0.5 \\ 0.0 & 1.0 \end{bmatrix}$$

The controller produces query $q = [0.8,\ 0.2]$ at time step $t$.

Use $e^{0.566} \approx 1.761$, $e^{0.354} \approx 1.425$, $e^{0.141} \approx 1.151$. Round to 3 decimal places.

**(i)** Compute the raw attention score for each memory slot: $s_i = q \cdot M_i / \sqrt{d}$. Show the dot product with each slot explicitly.

&nbsp;

&nbsp;

&nbsp;

**(ii)** Compute the attention weights $\alpha = \text{softmax}(s)$. Show the exponentials, their sum, and the final weights.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(iii)** Compute the read vector $r_t = \sum_i \alpha_i M_i$. Show each weighted slot before summing.

&nbsp;

&nbsp;

&nbsp;

**(iv)** The controller now performs a write operation with erase vector $e_t = [1.0,\ 1.0]^T$ and add vector $a_t = [0.0,\ 1.0]^T$. Apply the write update $M_i \leftarrow M_i(1 - \alpha_i e_t) + \alpha_i a_t$ to slot $M_1$ only. Show the calculation explicitly.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(v)** After the write in part (iv), slot $M_1$ has changed. Explain qualitatively what the write operation did to $M_1$ — what was erased and what was written — and why the attention weight $\alpha_1$ from part (ii) determines how strongly the slot was modified.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** Self-attention has $O(T^2)$ complexity because every position attends to every other position. A student proposes "sparse attention" — each position attends only to its $k$ nearest neighbours rather than all $T$ positions, reducing complexity to $O(Tk)$. Identify one type of linguistic dependency that sparse attention would handle well and one that it would fail to capture. Explain why.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The classical Hopfield network from Note 15 has a storage capacity of approximately $0.14N$ patterns, where $N$ is the number of neurons. The modern Hopfield network (equivalent to attention) has exponential capacity in the dimension $d$. Explain what structural difference between the two models accounts for this dramatic improvement in capacity.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
