# Exercise 01 — ML vs Deep Learning

Answer all questions.

---

## Section A — Multiple Choice

*Circle the most correct answer.*

**Q1.** A traditional ML pipeline and a deep learning pipeline both receive the same raw image as input. The fundamental difference in how they process it is:

- a) Deep learning uses more training data
- b) Traditional ML uses a CPU while deep learning uses a GPU
- c) Traditional ML requires a human to design useful representations before the model learns; deep learning learns those representations as part of training
- d) Deep learning always achieves higher accuracy

---

**Q2.** A neural network with no activation functions, regardless of how many layers it has, is mathematically equivalent to:

- a) A decision tree
- b) A single linear transformation
- c) A support vector machine
- d) A shallow network with one hidden layer

---

**Q3.** Which of the following best explains why deep learning became practically viable around 2012 and not earlier?

- a) Neural network theory was only developed after 2010
- b) The convergence of large labeled datasets, GPU compute, and improved training algorithms
- c) Traditional ML algorithms were proven to be insufficient
- d) Deep learning requires less data than traditional ML

---

**Q4.** The bias term `b` in the neuron formula `z = wᵀx + b` serves which geometric purpose?

- a) It scales the output to be between 0 and 1
- b) It allows the decision boundary to shift away from the origin
- c) It prevents the weights from becoming too large
- d) It adds non-linearity to the model

---

**Q5.** A SHAP value of +0.85 for the feature "missed payments" in a loan default model means:

- a) The feature has an 85% probability of causing default
- b) The feature pushed the prediction strongly toward high risk
- c) The feature is 85% accurate
- d) The feature should be removed from the model

---

## Section B — Short Answer

*Answer precisely. One or two paragraphs maximum per question.*

**Q6.** The statement "deep learning does not require feature engineering" is commonly repeated but imprecise. Write a more technically accurate version of this statement and explain why the original phrasing is misleading.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q7.** A colleague argues: "If I stack 50 linear layers in a neural network, it must be more powerful than a single layer because it has more parameters." Prove mathematically why this argument is wrong.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q8.** Grad-CAM can show you which region of an image a CNN focused on when making a prediction. A student concludes: "This means CNNs are interpretable." What is wrong with this conclusion? What is the precise distinction that needs to be made?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q9.** You are given two problems:

- Problem A: Predict customer churn from a table of 15 features (age, income, usage, etc.) with 5,000 records
- Problem B: Classify spoken words from raw audio with 500,000 recordings

For each problem, argue whether traditional ML or deep learning is the more appropriate starting point and why. Your answer must reference feature engineering, data scale, and input complexity.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

**Q10.** A neural network has 3 layers stacked with no activation function. The weight matrices and biases are:

```
Layer 1: W₁ = [[0.5, 0.2], [0.3, 0.8]],  b₁ = [0.1, 0.1]
Layer 2: W₂ = [[0.4, 0.6], [0.2, 0.5]],  b₂ = [0.0, 0.0]
Layer 3: W₃ = [[1.0, 0.0], [0.0, 1.0]],  b₃ = [0.2, 0.2]
Input:   x  = [1, 0]
```

**(a)** Compute the output of each layer step by step.

**(b)** Now combine all three layers into a single equivalent transformation W = W₃W₂W₁ and b = W₃W₂b₁ + W₃b₂ + b₃. Compute the final output directly from the input.

**(c)** What do your results from (a) and (b) demonstrate about stacking linear layers?

---

## Section D — Critical Thinking

**Q11.** AlexNet (2012) is often cited as the moment deep learning took over. A student says: "AlexNet won because it was a better algorithm." A researcher says: "AlexNet won because of hardware." Who is more correct, and what does this tell us about how breakthroughs in AI actually happen?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

**Q12.** You are building a system to detect fraudulent transactions for a bank. The model must be explainable to regulators. Based on what you know about interpretability in ML vs deep learning, which approach would you recommend and why? What are the tradeoffs you are accepting?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
