# Exercise 22 — Data Augmentation

---

## Section A — Multiple Choice

**1.** Which of the following best explains why data augmentation reduces overfitting?

- A) It increases the number of model parameters, giving the model more capacity
- B) It prevents the model from memorising specific training examples by ensuring inputs vary across epochs
- C) It reduces the learning rate, slowing down weight updates
- D) It removes noisy samples from the training set

**2.** A model is trained to classify handwritten digits (0–9). Which augmentation is most likely to hurt performance?

- A) Random horizontal flip
- B) Small random rotation (±10°)
- C) Brightness jitter
- D) Random crop with padding

**3.** Why is augmentation applied only during training and not during evaluation?

- A) Augmentation is too computationally expensive to apply at test time
- B) The test set must measure performance on the real data distribution, not on transformed versions
- C) Augmented images are lower quality and would reduce test accuracy
- D) The model's weights are frozen during evaluation, so augmentation has no effect

**4.** A training pipeline applies: RandomHorizontalFlip → RandomCrop(32, padding=4) → ColorJitter → Normalize. Which of these steps should also be applied at test time?

- A) All four steps
- B) RandomHorizontalFlip and Normalize only
- C) Normalize only
- D) None — the test set should receive raw images

**5.** Test-time augmentation (TTA) applies multiple transforms to each test image and averages the predictions. What is the main tradeoff?

- A) Higher accuracy at the cost of increased inference time
- B) Lower accuracy but faster inference
- C) Higher accuracy with no additional cost
- D) Reduced overfitting at the cost of higher training loss

---

## Section B — Short Answer

**1.** Explain the connection between data augmentation and the concept of inductive bias. What inductive bias does horizontal flipping encode, and for which tasks is this bias incorrect?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A student applies aggressive colour jitter (hue shift of ±0.5) to a dataset of traffic sign images. Explain why this is likely to hurt performance rather than help it. What does this reveal about the limits of augmentation?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Describe how random crop with padding works and explain why it is more effective than simply cropping a fixed region from the centre of the image.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** Augmentation and dropout are both described as regularisation techniques. Explain the key difference in where each operates and why using both together is beneficial.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A training set contains **500 images**. A deterministic augmentation pipeline is applied offline (before training), generating all augmented versions explicitly. The pipeline consists of:

- Horizontal flip: original + flipped = **2 versions**
- Vertical flip: original + flipped = **2 versions**
- 90° rotation: 0°, 90°, 180°, 270° = **4 versions**

All combinations are generated independently.

**(a)** How many unique versions does the pipeline produce per image? Show the calculation.

&nbsp;

&nbsp;

&nbsp;

**(b)** What is the total effective dataset size after applying the full pipeline to all 500 images?

&nbsp;

&nbsp;

&nbsp;

**(c)** The vertical flip is removed from the pipeline (not appropriate for upright objects). How many versions per image remain, and what is the new effective dataset size?

&nbsp;

&nbsp;

&nbsp;

**(d)** A colour augmentation step is added: 3 discrete brightness levels (dark, normal, bright). This is applied on top of the pipeline from part (c). How many versions per image does the combined pipeline produce? What is the new effective dataset size?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** The original dataset has 500 images and the model has 2,000,000 parameters. After the full augmentation from part (d), what is the ratio of training examples to parameters? What does this suggest about the risk of overfitting compared to training on the original 500 images?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A colleague argues: "More augmentation is always better — the more transforms you apply, the more the model generalises." Construct a concrete counterexample that shows this is false. Your example should identify a specific augmentation, a specific task, and explain precisely why applying that augmentation would hurt rather than help.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Consider training a CNN to detect pneumonia from chest X-rays. The dataset has 800 positive (pneumonia) and 800 negative (healthy) images. A colleague proposes applying the same augmentation pipeline used for ImageNet training: RandomResizedCrop, RandomHorizontalFlip, and ColorJitter. Evaluate each of these three transforms for this specific task — is each one appropriate, inappropriate, or requires modification? Justify each answer.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
