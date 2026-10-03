# Exercise 20 — CNN Architecture: Convolutional, Pooling, and Fully Connected Layers

---

## Section A — Multiple Choice

**1.** A convolutional layer has input size 10×10, filter size 3×3, stride=2, and padding=1. What is the output size?

- A) 4×4
- B) 5×5
- C) 6×6
- D) 8×8

**2.** A convolutional layer has F=3, C_in=16, K=32, and bias. How many learnable parameters does it have?

- A) 144
- B) 4,608
- C) 4,640
- D) 9,248

**3.** Which of the following correctly describes "same" padding?

- A) Padding that makes the output the same size as the filter
- B) Zero padding chosen so that the output spatial size equals the input spatial size (with stride=1)
- C) Padding that ensures all filter positions overlap by the same amount
- D) Padding that sets border pixels to the mean of their neighbours

**4.** A 2×2 max pool with stride=2 is applied to a 6×6 feature map. What is the output size?

- A) 2×2
- B) 3×3
- C) 4×4
- D) 5×5

**5.** Three 3×3 convolutional layers are stacked with stride=1 and no padding. What is the effective receptive field of a neuron in the third layer's output?

- A) 3×3
- B) 5×5
- C) 7×7
- D) 9×9

---

## Section B — Short Answer

**1.** Explain why the parameter count of a convolutional layer does not depend on the spatial size of the input (H and W). What does it depend on, and why is this property important?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A student applies a 3×3 filter with stride=1 and no padding to a 5×5 input, then applies a 3×3 filter with stride=1 and no padding to the result. What is the final output size? What is the effective receptive field of a neuron in the final output? Show your working.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Max pooling has zero learnable parameters. Explain what it does, why it is used, and what property of the feature map it provides that makes it useful for object recognition.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** In the LeNet-style architecture in Note 20, the two convolutional layers have 2,572 parameters and the three FC layers have 41,854 parameters. Why do the FC layers dominate, and what architectural change do modern networks use to reduce this?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A small CNN processes a 6×6 single-channel input through the following layers:

**Layer 1 — Convolution:** 3×3 filter, stride=1, padding=0, bias=0

```
Filter:
  1  0  1
  0  1  0
  1  0  1
```

**Layer 2 — ReLU**

**Layer 3 — Max Pooling:** 2×2, stride=2

**Layer 4 — Flatten**

**Layer 5 — Fully Connected:** 4 inputs → 2 outputs

```
W = [ 1   0 ]     b = [0, 0]
    [ 0   1 ]
    [ 1   0 ]
    [ 0   1 ]
```

**Input (6×6):**

```
1  0  2  1  0  1
0  1  1  0  2  0
2  1  0  1  1  2
1  0  1  2  0  1
0  2  1  0  1  0
1  1  0  1  2  1
```

**(a)** What is the output size after Layer 1? Show the formula.

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute the convolution output at positions (0,0) and (0,3). Show the patch, element-wise products, and sum for each.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** The full convolution output is:

```
  6  4  3  7
  4  3  5  4
  3  5  5  3
  5  5  3  6
```

Apply ReLU. What changes?

&nbsp;

&nbsp;

**(d)** Apply 2×2 max pooling with stride=2. Show the four regions and their maximum values. What is the output?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** Flatten the pooled output and apply the FC layer. Compute z = W^T × flat + b. Show the computation for each output neuron.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(f)** How many total learnable parameters does this network have? Count the conv layer (filter + bias) and the FC layer (weights + biases) separately.

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A colleague proposes replacing all convolutional layers in a CNN with fully connected layers, arguing that FC layers are strictly more general — they can learn any mapping, including the convolution operation. The colleague claims this would give a more powerful network. Evaluate this argument. Under what conditions is the colleague technically correct? Why does this argument fail in practice for image data, and what does the concept of inductive bias have to do with it?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Consider two architectures for processing a 32×32 image:
- Architecture A: one 7×7 conv layer (K=64 filters)
- Architecture B: three 3×3 conv layers (K=64 filters each)

Both have the same effective receptive field. Compare them on: (i) parameter count, (ii) number of non-linearities, (iii) representational capacity. Which would you choose and why?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
