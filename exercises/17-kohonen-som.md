# Exercise 17 — Kohonen's Self-Organising Map

---

## Section A — Multiple Choice

**1.** What distinguishes the SOM from standard competitive learning (winner-takes-all)?

- a) The SOM uses a different distance metric to find the winning neuron
- b) The SOM updates the winning neuron and its neighbours, preserving topology
- c) The SOM uses supervised learning to assign inputs to neurons
- d) The SOM updates all neurons equally regardless of distance from the winner

&nbsp;

**2.** In a trained SOM, what does it mean for two neurons to be adjacent on the map grid?

- a) They were initialised with similar random weight vectors
- b) They respond to inputs that are similar in the original input space
- c) They have the same number of training examples assigned to them
- d) They were updated in the same training step

&nbsp;

**3.** The Gaussian neighbourhood function h(j, j*) = exp(−d² / 2σ²) is evaluated at d=2, σ=1.0. What is h?

- a) exp(−1)
- b) exp(−2)
- c) exp(−4)
- d) exp(−0.5)

&nbsp;

**4.** During the ordering phase of SOM training, σ is large. What is the primary effect of this?

- a) Only the BMU updates — all other neurons are unaffected
- b) Most of the map updates together, allowing global topology to form
- c) The learning rate η is automatically increased to compensate
- d) The weight vectors are reset to their initial random values

&nbsp;

**5.** A U-matrix entry between two adjacent neurons has a high value. What does this indicate?

- a) The two neurons respond to the same input
- b) The two neurons have similar weight vectors
- c) There is a cluster boundary between the two neurons
- d) The two neurons were both BMUs for the same training example

&nbsp;

---

## Section B — Short Answer

**1.** Explain why the SOM requires both an ordering phase and a convergence phase. What goes wrong if training uses only a small, fixed σ throughout?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A SOM is trained on 3D colour vectors (R, G, B) using a 10×10 map grid. After training, what would you expect to observe about the spatial arrangement of neurons that respond to red inputs versus neurons that respond to blue inputs?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Compare the SOM to k-means clustering. Both partition the input space into regions. What does the SOM provide that k-means does not?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** In the SOM weight update rule Δwⱼ = η · h(j, j*) · (x − wⱼ), what happens to a neuron whose weight vector is already equal to x? What does this imply about convergence?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A 1D SOM has 3 neurons (j=1, j=2, j=3) arranged in a line. Each neuron has a 2D weight vector. Current weights, input, and parameters:

```
w₁ = (0.1, 0.9)
w₂ = (0.7, 0.5)
w₃ = (0.4, 0.2)

Input:           x = (0.8, 0.2)
Learning rate:   η = 0.4
Neighbourhood:   σ = 1.0
```

**(a)** Compute the Euclidean distance from x to each weight vector. Identify the BMU.

**(b)** Compute the grid distance d(j, j*) from each neuron to the BMU. Then compute the neighbourhood strength h(j, j*) = exp(−d² / 2σ²) for each neuron.

**(c)** Compute the weight update Δwⱼ = η · h(j, j*) · (x − wⱼ) for each neuron and the new weight vectors after one update step.

&nbsp;

---

## Section D — Critical Thinking

**1.** A researcher trains a SOM on a dataset where the data lies on a 1D curve embedded in 3D space (like a spiral). They use a 10×10 2D map grid. After training, they find that only a narrow diagonal strip of neurons is active — most neurons never win on any input. Explain why this happens and what it reveals about the relationship between the map topology and the data topology.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** The SOM is described as topology-preserving, but this guarantee is approximate — it can fail when the data has a complex topology that cannot be unfolded onto a 2D grid without tearing. Give a concrete example of a data distribution where the SOM topology-preservation property would break down, and explain what the resulting map would look like.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
