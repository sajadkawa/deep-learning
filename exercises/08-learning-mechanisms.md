# Exercise 08 — Learning Mechanisms: Hebbian, Competitive, Boltzmann

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** Two input neurons have activations x₁ = 1 and x₂ = 0. The output neuron fires (y = 1). Using the Hebbian rule Δwᵢⱼ = η × xᵢ × yⱼ with η = 0.3, what are Δw₁ and Δw₂?

- a) Δw₁ = 0.3, Δw₂ = 0.3
- b) Δw₁ = 0.3, Δw₂ = 0
- c) Δw₁ = 0, Δw₂ = 0.3
- d) Δw₁ = 0, Δw₂ = 0

---

**Q2.** In competitive learning with three output neurons, neuron 2 has the smallest Euclidean distance to the current input. Which neurons update their weights?

- a) All three neurons update equally
- b) Only neuron 2 updates
- c) Neurons 1 and 3 update; neuron 2 does not
- d) All three neurons update, but neuron 2 updates most

---

**Q3.** In a Boltzmann machine, the probability of a state (v, h) is P(v, h) = e^(−E(v,h)) / Z. What happens to the probability of a state as its energy decreases?

- a) The probability decreases
- b) The probability is unchanged — only Z matters
- c) The probability increases
- d) The probability becomes negative

---

**Q4.** Oja's rule modifies the Hebbian rule by adding a decay term −yⱼ × wᵢⱼ. What is the primary purpose of this term?

- a) To increase the learning rate for large weights
- b) To prevent weights from growing without bound
- c) To introduce competition between output neurons
- d) To make the rule equivalent to the perceptron learning rule

---

**Q5.** The perceptron learning rule is Δwᵢ = η × (y − ŷ) × xᵢ. Which of the following correctly identifies what makes Hebbian, competitive, and Boltzmann learning fundamentally different from this rule?

- a) They use a different learning rate
- b) They require more training data
- c) They do not use a labeled target — the update signal comes from the data structure itself
- d) They only work for binary inputs

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** Pure Hebbian learning is described as unstable. Explain precisely what instability means here — what happens to the weights over time, and why does this make the rule unusable in practice without modification?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** Competitive learning is mathematically equivalent to k-means clustering. Explain the correspondence precisely: what plays the role of the centroid, what plays the role of the assignment step, and what plays the role of the update step?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** The Boltzmann learning rule is Δwᵢⱼ = η × (⟨vᵢhⱼ⟩_data − ⟨vᵢhⱼ⟩_model). Explain in plain terms what each of the two expectation terms measures and what the rule is trying to achieve. Why is computing ⟨vᵢhⱼ⟩_model the computational bottleneck?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** The note states that the Boltzmann learning rule is "Hebbian in structure." In what sense is this true? Identify the Hebbian component and explain what makes the Boltzmann rule different from pure Hebbian learning despite this structural similarity.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** A Hebbian network has 2 input neurons and 1 output neuron. Initial weights: w₁ = 0, w₂ = 0. Learning rate η = 0.2. The output neuron fires (y = 1) for every pattern presented.

Three patterns are presented in order:
- Pattern 1: x = [1, 0]
- Pattern 2: x = [1, 1]
- Pattern 3: x = [1, 0]

**(a)** Apply the Hebbian rule Δwᵢ = η × xᵢ × y after each pattern. Show the weight update and the new weights after each step.

**(b)** After all three patterns, state the final weights. Which input does the neuron respond to more strongly, and why does this reflect the training data?

**(c)** Now apply one step of Oja's rule to Pattern 1 (x = [1, 0]) starting from w = [0.40, 0.20]. The output is y = w·x.

Compute:
- y = w₁ × x₁ + w₂ × x₂
- Δw₁ = η × y × (x₁ − y × w₁)
- Δw₂ = η × y × (x₂ − y × w₂)
- New weights w₁_new and w₂_new

You may use: 0.4 × 0.4 = 0.16, 0.2 × 0.4 × 0.4 = 0.032, 0.2 × 0.4 × 0.2 = 0.016

**(d)** Compare the Oja update to the pure Hebbian update for the same pattern and starting weights. What is the key difference in how w₂ changes?

---

## Section D — Critical Thinking

**Q11.** The note states that Hebbian learning detects co-occurrence — inputs that are frequently active together develop strong connections. A student argues: *"This means Hebbian learning is just memorising which inputs appear together — it is not really learning anything useful."*

Construct a counter-argument. Give a concrete example of a task where detecting co-occurrence is genuinely useful, and explain how the weight structure that emerges from Hebbian learning encodes something meaningful about the data.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** Competitive learning suffers from the dead neuron problem — some neurons never win and their weights never update. This is described as analogous to the dying ReLU problem from Note 04.

- **(a)** Explain the analogy precisely. What is the structural similarity between a dead competitive neuron and a dead ReLU neuron?
- **(b)** Soft competition (the basis of SOMs) addresses the dead neuron problem by allowing neighbours of the winner to also update. Explain why this helps, and identify a potential downside of allowing too large a neighbourhood.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
