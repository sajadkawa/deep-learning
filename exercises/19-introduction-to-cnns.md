# Exercise 19 — Introduction to Convolutional Neural Networks

---

## Section A — Multiple Choice

**1.** A fully connected layer receives a 64×64 grayscale image (flattened to 4,096 inputs) and has 256 output neurons. How many weights does this layer have (excluding biases)?

- A) 256
- B) 4,096
- C) 1,048,576
- D) 65,536

**2.** A CNN neuron has a 5×5 receptive field applied to a single-channel input. How many weights does this neuron use (excluding bias)?

- A) 5
- B) 10
- C) 25
- D) 50

**3.** Parameter sharing in a convolutional layer means:

- A) All neurons in the layer share the same bias value
- B) The same filter weights are applied at every spatial position in the input
- C) Weights are shared between different layers of the network
- D) Each filter is applied only once, at the centre of the image

**4.** A 3×3 filter is applied to a 7×7 input with stride 1 and no padding. What is the size of the output feature map?

- A) 7×7
- B) 5×5
- C) 3×3
- D) 9×9

**5.** Which of the following best describes translation equivariance in a convolutional layer?

- A) The network produces the same output regardless of where the object appears
- B) If the input shifts by (Δr, Δc), the feature map shifts by the same amount
- C) The filter weights are updated to match the position of the detected feature
- D) The network applies a different filter at each spatial position

---

## Section B — Short Answer

**1.** Explain why flattening an image before passing it to a fully connected layer destroys spatial structure. What information is lost, and why does this matter for visual tasks?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A CNN is trained to detect cats in photographs. After training, it correctly identifies a cat in the top-left corner of an image. Explain, in terms of parameter sharing, why the same network will also detect a cat in the bottom-right corner without any additional training.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Describe the hierarchical feature learning that occurs in a deep CNN trained on natural images. What types of features are typically detected at early, middle, and late layers, and why does this hierarchy emerge from training rather than being explicitly designed?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** Distinguish between image classification, object detection, and semantic segmentation. For each task, state what the output of the network looks like.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A 5×5 input feature map and a 3×3 filter are given below. The filter slides over the input with stride 1 and no padding.

**Input (5×5):**

```
1  2  3  0  1
0  1  2  3  0
1  0  1  2  1
2  1  0  1  2
0  1  2  1  0
```

**Filter (3×3):**

```
1  1  0
0  1  1
1  0  1
```

**(a)** What is the size of the output feature map? Show the formula used.

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute the output value at position (1,1) — the top-left corner of the output. Show the patch extracted from the input, the element-wise products with the filter, and the sum.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Compute the output value at position (1,2) — one step to the right. Show the same steps.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** Complete the full 3×3 output feature map. You may state the remaining values without showing full working.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A student argues: "Parameter sharing is a constraint that limits what the network can learn. A fully connected layer is strictly more powerful because it can learn different weights for every position. CNNs are just a restricted version of FC networks, so they should always perform worse." Evaluate this argument. Under what conditions is the student correct, and under what conditions is the argument wrong? What does the concept of inductive bias have to do with this?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** CNNs were originally designed for 2D images. Consider applying a 1D convolutional layer to a time series (a sequence of scalar values over time). What would the "receptive field" correspond to in this context? What assumption about the structure of time series data does the 1D CNN encode, and is this assumption reasonable? Give one example where it holds and one where it might not.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
