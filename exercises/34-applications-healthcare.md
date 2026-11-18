# Exercise 34 — Applications in Healthcare

## Section A — Multiple Choice

**1.** Why is accuracy a misleading metric for a disease detection model trained on a dataset that is 95% normal cases?

- a) Accuracy does not account for the number of parameters in the model
- b) A model that always predicts "normal" achieves 95% accuracy while detecting zero disease cases
- c) Accuracy is only valid for multi-class problems, not binary classification
- d) Accuracy penalises false positives more than false negatives

&nbsp;

**2.** In transfer learning for medical imaging, which layers of a pre-trained ImageNet model are most useful without modification?

- a) The final fully connected layers, which encode high-level semantic features
- b) The early convolutional layers, which encode universal low-level features like edges and textures
- c) The batch normalisation layers, which normalise to ImageNet statistics
- d) The output layer, which maps features to ImageNet class probabilities

&nbsp;

**3.** In Grad-CAM, the importance weight $\alpha^k_c$ for feature map $k$ and class $c$ is computed as:

- a) The maximum gradient value in feature map $k$
- b) The global average pooling of the gradient $\frac{\partial S_c}{\partial A^k}$ over all spatial positions
- c) The dot product of the feature map $A^k$ with the class weight vector
- d) The L2 norm of the feature map $A^k$

&nbsp;

**4.** Why is the ReLU applied in the final step of Grad-CAM?

- a) To normalise the heatmap values to the range $[0, 1]$
- b) To keep only the spatial regions that positively influence the class score
- c) To match the activation function used in the final conv layer
- d) To remove noise from the gradient computation

&nbsp;

**5.** Which of the following best describes the cost-sensitive loss weighting approach for class imbalance?

- a) Duplicate minority class samples until classes are balanced
- b) Assign higher loss weight to the minority class so it contributes equally to training
- c) Remove majority class samples until classes are balanced
- d) Apply dropout only to the majority class neurons

&nbsp;

---

## Section B — Short Answer

**1.** Explain why transfer learning from ImageNet to medical imaging works despite the large domain shift between natural photographs and medical images.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A chest X-ray classifier achieves 98% accuracy on a test set where 97% of images are normal. Explain why this result is not clinically useful and state two metrics that would give a more informative evaluation.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Describe the two steps of Grad-CAM: what is computed in each step and what does the final heatmap represent?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** How are transformer models applied to genomic sequences? What self-supervised pre-training task is used and why is it well-suited to DNA data?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A CNN classifying chest X-rays has a final conv layer with $K = 3$ feature maps, each of spatial size $2 \times 2$. The class is "pneumonia" ($c$).

**Feature maps:**

$$A^1 = \begin{bmatrix} 0.8 & 0.2 \\ 0.3 & 0.6 \end{bmatrix}, \quad A^2 = \begin{bmatrix} 0.1 & 0.9 \\ 0.4 & 0.5 \end{bmatrix}, \quad A^3 = \begin{bmatrix} 0.7 & 0.3 \\ 0.2 & 0.8 \end{bmatrix}$$

**Gradients** $\frac{\partial S_c}{\partial A^k_{ij}}$ (uniform within each map):

$$\frac{\partial S_c}{\partial A^1} = \begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}, \quad \frac{\partial S_c}{\partial A^2} = \begin{bmatrix} 0.3 & 0.3 \\ 0.3 & 0.3 \end{bmatrix}, \quad \frac{\partial S_c}{\partial A^3} = \begin{bmatrix} 0.2 & 0.2 \\ 0.2 & 0.2 \end{bmatrix}$$

**(a)** Compute the Grad-CAM importance weights $\alpha^1$, $\alpha^2$, $\alpha^3$ using global average pooling of the gradients.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** Compute the weighted sum $\sum_k \alpha^k A^k$ element-wise.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** Apply ReLU to obtain the Grad-CAM heatmap $L^c$. Which spatial position has the highest saliency?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** A dataset has 900 normal cases and 100 pneumonia cases. Compute the cost-sensitive class weights $w_c = \frac{N}{C \cdot N_c}$ for each class. By what factor is the pneumonia loss weighted more than the normal loss?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A deep learning model for diabetic retinopathy screening achieves AUC-ROC of 0.97 on a held-out test set from the same hospital where it was trained. When deployed at a different hospital with a different imaging device, AUC-ROC drops to 0.71. Identify the problem and suggest two concrete steps to address it before deployment.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A clinician argues that a black-box model with 95% sensitivity should not be used in clinical practice without interpretability, even if a less accurate but interpretable model exists. Evaluate this argument — under what conditions is interpretability essential, and under what conditions might it be acceptable to deploy a black-box model?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
