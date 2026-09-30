# Exercise 09 — Deep Feedforward Network Architectures

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** A network has an input layer (784 units), two hidden layers (512 and 256 units), and an output layer (10 units). What is the depth L of this network?

- a) 2
- b) 3
- c) 4
- d) 5

---

**Q2.** How many trainable parameters does a fully connected layer have if it connects 128 input neurons to 64 output neurons?

- a) 128 × 64 = 8,192
- b) 128 × 64 + 128 = 8,320
- c) 128 × 64 + 64 = 8,256
- d) 128 + 64 = 192

---

**Q3.** The Universal Approximation Theorem guarantees that a single hidden layer network can approximate any continuous function. What does it NOT guarantee?

- a) That the approximation can be made arbitrarily precise
- b) That the required width is practical for complex tasks
- c) That the activation function must be non-linear
- d) That the network can represent any continuous function on a compact domain

---

**Q4.** In the computational graph of a feedforward network, why must the graph be acyclic (a DAG)?

- a) To ensure the network has fewer parameters
- b) To guarantee a valid topological evaluation order exists for the forward pass
- c) To prevent vanishing gradients during backpropagation
- d) To allow the same weight to be reused across layers

---

**Q5.** You are designing a network for a 5-class classification problem. Which output layer configuration is correct?

- a) 1 output neuron, sigmoid activation
- b) 5 output neurons, ReLU activation
- c) 5 output neurons, softmax activation
- d) 5 output neurons, no activation

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** The note states that depth provides exponential representational efficiency over width (Montufar et al., 2014: O(nᴸ) linear regions for deep vs O(n × L) for shallow). Explain in concrete terms what a "linear region" is and why more linear regions means a more expressive network.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** The forward pass stores all intermediate values z⁽ˡ⁾ and a⁽ˡ⁾ at every layer. Explain precisely why these values must be stored — what would fail during training if they were discarded after the forward pass?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** The note recommends a funnel width profile (wider early layers, narrower later layers) as a common architecture pattern. Give the geometric/representational reason why this shape makes sense — what is each stage of the funnel doing to the data?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** A student argues: *"The UAT proves a 1-hidden-layer network can approximate any function, so there is no theoretical reason to use deep networks — depth is just an engineering convenience."* Identify the precise flaw in this argument.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** Consider the following two architectures, both taking a 10-dimensional input and producing a 3-dimensional output.

**Architecture X:** n₀=10, n₁=32, n₂=32, n₃=32, n₄=3 (depth L=4)

**Architecture Y:** n₀=10, n₁=128, n₂=3 (depth L=2)

**(a)** Compute the total parameter count for Architecture X. Show the calculation layer by layer.

**(b)** Compute the total parameter count for Architecture Y. Show the calculation layer by layer.

**(c)** Architecture X is deeper but has fewer parameters than Architecture Y. Using the Montufar et al. result, compute the number of linear regions each can represent (use n=32, L=4 for X and n=128, L=2 for Y). What does this tell you about the relationship between parameter count and representational power?

**(d)** You have 500 training examples. Which architecture would you choose and why? Consider both the parameter counts from (a)/(b) and the representational power from (c).

---

## Section D — Critical Thinking

**Q11.** The note states that the output layer design is "fixed by the task" and is not a design choice. A student disagrees: *"I could use a sigmoid output for a 3-class problem by thresholding the output at 0.5 — it would still produce a prediction."*

Explain precisely why this approach is wrong — not just suboptimal, but structurally incorrect. Your answer should address what sigmoid outputs represent, what softmax outputs represent, and what is lost when you use the wrong one.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** The reuse argument says that a feature learned in an early layer of a deep network is available to all subsequent layers at no additional parameter cost. This is presented as the core reason depth is efficient.

- **(a)** Identify a type of task or data where this reuse argument is strong — where compositional structure genuinely exists in the data. Explain why depth maps naturally onto that structure.
- **(b)** Identify a type of task or data where the reuse argument is weak — where depth provides little advantage over a wide shallow network. Explain why.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
