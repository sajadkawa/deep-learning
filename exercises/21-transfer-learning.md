# Exercise 21 — Transfer Learning and Pre-trained Models

---

## Section A — Multiple Choice

**1.** A ResNet-18 pre-trained on ImageNet is used for feature extraction on a new 5-class dataset. The final FC layer (512→1000) is replaced with a new FC layer (512→5). Which parameters are updated during training?

- A) All parameters in the network
- B) Only the new FC layer (512→5)
- C) The last convolutional block and the new FC layer
- D) The batch normalisation layers and the new FC layer

**2.** VGG16 has 138 million parameters. Approximately what fraction of these are in the three fully connected layers?

- A) 11%
- B) 45%
- C) 89%
- D) 99%

**3.** A plain 56-layer network achieves higher training error than a plain 20-layer network on the same dataset. This is an example of:

- A) Overfitting — the deeper network memorises the training data
- B) The degradation problem — optimisation fails in very deep plain networks
- C) Underfitting — the deeper network has too many parameters
- D) The vanishing gradient problem in the forward pass

**4.** In a ResNet basic block, the output is computed as F(x) + x. What is the role of the +x term during backpropagation?

- A) It doubles the gradient magnitude at every layer
- B) It provides a direct gradient path that prevents the gradient from vanishing
- C) It regularises the weights by penalising large activations
- D) It ensures the output is always positive

**5.** You have 500 labelled images of a new type of industrial defect. You want to classify them using a ResNet-18 pre-trained on ImageNet. Which strategy is most appropriate?

- A) Train ResNet-18 from scratch on the 500 images
- B) Feature extraction — freeze the backbone, train a new head
- C) Fine-tune all layers at the standard learning rate
- D) Use only the first two convolutional blocks and discard the rest

---

## Section B — Short Answer

**1.** Explain why the features learned by a CNN trained on ImageNet transfer to tasks like medical image classification, even though ImageNet contains no medical images. Which layers transfer best and why?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** In fine-tuning, the learning rate for the backbone is set much lower than for the new head (e.g. 1e-5 vs 1e-3). Explain why. What would happen if the same high learning rate were applied to the backbone?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** VGG16 uses only 3×3 filters throughout all convolutional layers. Explain the two advantages of this design choice over using larger filters (e.g. 7×7 or 11×11). Your answer should reference both parameter count and representational capacity.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** A ResNet basic block learns the residual F(x) = H(x) − x rather than the full mapping H(x) directly. Explain why learning a residual is easier to optimise than learning the full mapping, particularly when the optimal transformation is close to the identity.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A ResNet-18 backbone (pre-trained on ImageNet) is used for feature extraction on a 10-class target dataset. The original FC layer (512→1000) is removed. A new two-layer head is attached:

```
Head FC1:   512 inputs  → 256 outputs   (+ bias, + ReLU)
Head FC2:   256 inputs  →  10 outputs   (+ bias, + Softmax)
```

The backbone (everything except the original FC layer) has **11,176,512 parameters**, all frozen.

**(a)** Compute the number of trainable parameters in Head FC1. Show the formula.

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute the number of trainable parameters in Head FC2. Show the formula.

&nbsp;

&nbsp;

&nbsp;

**(c)** What is the total number of trainable parameters? What fraction of the total network parameters (backbone + head) are trainable? Express as a percentage.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** During feature extraction, the backbone processes a batch of 32 images and produces a 32×512 feature matrix. Head FC1 applies W (512×256) and bias b (256). Write the matrix operation that produces the pre-activation output Z, and state its shape.

&nbsp;

&nbsp;

&nbsp;

**(e)** After training the head, you decide to fine-tune the last residual block (Layer 4) of ResNet-18. Layer 4 contains two basic blocks, each with: 2× Conv(3×3, 512, 512) with no bias + 2× BN(512). Compute the number of parameters in one basic block. How many additional parameters become trainable when Layer 4 is unfrozen?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A colleague argues: "Transfer learning is just a better form of weight initialisation — instead of random weights, you start from ImageNet weights. The network will converge to the same solution either way given enough training data and time." Evaluate this argument. Is the colleague correct that transfer learning is equivalent to initialisation? What does the argument miss about why transfer learning works, and under what conditions does the distinction matter most?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Consider applying transfer learning from ImageNet to two very different target tasks: (i) classifying galaxy morphologies from telescope images, and (ii) classifying audio spectrograms by music genre. For each task, discuss whether ImageNet features are likely to transfer, which layers are most reusable, and what strategy (feature extraction vs fine-tuning) you would choose. What does this reveal about the limits of transfer learning?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
