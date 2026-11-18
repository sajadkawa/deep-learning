# Exercise 32 — Applications in Computer Vision

## Section A — Multiple Choice

**1.** What is the key architectural difference between a classification CNN and a fully convolutional network (FCN)?

- a) FCNs use deeper encoders than classification CNNs
- b) FCNs replace fully connected layers with convolutional layers, preserving spatial structure
- c) FCNs use larger kernels to capture more context
- d) FCNs remove all pooling layers to maintain resolution

&nbsp;

**2.** In U-Net, what is the purpose of the skip connections between encoder and decoder?

- a) They allow gradients to bypass the bottleneck during backpropagation
- b) They concatenate high-resolution encoder feature maps into the decoder, recovering spatial detail
- c) They reduce the number of parameters by sharing weights between encoder and decoder
- d) They apply batch normalisation between encoder and decoder levels

&nbsp;

**3.** A predicted bounding box has area 500 and a ground truth box has area 600. Their intersection has area 300. What is the IoU?

- a) 0.375
- b) 0.500
- c) 0.600
- d) 0.375

&nbsp;

**4.** Why is pixel accuracy a misleading metric for semantic segmentation on datasets with class imbalance?

- a) It does not account for bounding box overlap
- b) A model predicting only the dominant class achieves high accuracy while failing on minority classes
- c) It requires knowing the number of object instances
- d) It penalises large objects more than small objects

&nbsp;

**5.** What distinguishes a one-stage detector (YOLO) from a two-stage detector (Faster R-CNN)?

- a) One-stage detectors use anchor boxes; two-stage detectors do not
- b) One-stage detectors predict boxes and classes in a single forward pass; two-stage detectors first propose regions then classify them
- c) One-stage detectors only detect one object per image
- d) Two-stage detectors use transposed convolutions for upsampling

&nbsp;

---

## Section B — Short Answer

**1.** Explain what a transposed convolution does and why it is preferred over bilinear interpolation for upsampling in segmentation networks.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Describe the role of anchor boxes in Faster R-CNN. What does the Region Proposal Network predict for each anchor?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Explain the difference between semantic segmentation and instance segmentation. Give one example task where instance segmentation is necessary but semantic segmentation is insufficient.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** What does mAP measure in object detection, and why is it preferred over a single precision or recall value?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Part 1 — IoU**

A predicted bounding box has corners $(x_1, y_1, x_2, y_2) = (10, 12, 50, 48)$.
A ground truth box has corners $(x_1, y_1, x_2, y_2) = (15, 10, 55, 45)$.

**(a)** Compute the area of the predicted box and the ground truth box.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute the intersection coordinates and the intersection area.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Compute the union area and the IoU. Does this detection count as a true positive at threshold $\theta = 0.5$?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**Part 2 — mIoU**

A $4 \times 4$ segmentation output (16 pixels total) has 2 classes: background (0) and foreground (1). The confusion is as follows:

| Pixels | True label | Predicted label |
|---|---|---|
| 8 | 0 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 0 |
| 3 | 1 | 1 |

**(d)** Identify TP, FP, FN for each class.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** Compute $\text{IoU}_0$, $\text{IoU}_1$, and $\text{mIoU}$.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(f)** Compute pixel accuracy. Explain why mIoU gives a more informative picture of model performance here.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** U-Net was originally designed for biomedical image segmentation where labelled data is scarce. Explain how the U-Net architecture is particularly well-suited to the low-data regime compared to a standard FCN without skip connections.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** YOLO divides the image into an $S \times S$ grid and each cell predicts a fixed number of bounding boxes. Identify one fundamental limitation of this design and explain what types of images or scenes it would fail on.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
