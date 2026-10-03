# Exercise 14 — Model Evaluation: Metrics

---

## Section A — Multiple Choice

**1.** A dataset has 950 negative samples and 50 positive samples. A classifier that always predicts negative achieves what accuracy?

- a) 50%
- b) 5%
- c) 95%
- d) 100%

&nbsp;

**2.** A spam filter incorrectly marks a legitimate email as spam. Which cell of the confusion matrix does this correspond to?

- a) True Positive
- b) False Negative
- c) True Negative
- d) False Positive

&nbsp;

**3.** A classifier has precision = 0.9 and recall = 0.4. What does this tell you about its behaviour?

- a) It catches most positives but raises many false alarms
- b) It rarely raises false alarms but misses many actual positives
- c) It performs equally well on both classes
- d) It is better than a random classifier on both metrics

&nbsp;

**4.** Why is the harmonic mean used for F1 rather than the arithmetic mean?

- a) The harmonic mean is always larger, giving a more optimistic score
- b) The harmonic mean is dominated by the smaller value, penalising classifiers that sacrifice one metric for the other
- c) The harmonic mean is easier to compute from the confusion matrix
- d) The harmonic mean weights recall more heavily than precision

&nbsp;

**5.** A classifier is evaluated on a dataset where 2% of samples are positive. Its ROC AUC is 0.85. Why might this be misleading?

- a) AUC above 0.8 is always unreliable
- b) ROC AUC is diluted by the large number of true negatives, making the classifier appear better than it is for finding the rare positive class
- c) AUC cannot be computed when classes are imbalanced
- d) The ROC curve requires equal class sizes to be valid

&nbsp;

---

## Section B — Short Answer

**1.** A confusion matrix has TP = 45, FP = 15, FN = 5, TN = 135. Compute precision, recall, and F1. Show your working.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Explain what the AUC of the ROC curve represents as a probability statement. What does AUC = 0.5 mean in this interpretation?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** A medical imaging classifier for detecting tumours achieves F1 = 0.75 with precision = 0.95 and recall = 0.62. A colleague argues the F1 score is good enough and the model is ready for deployment. What concern should you raise, and which metric should drive the deployment decision?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** Describe what happens to precision and recall as the classification threshold is lowered from 1.0 to 0.0. Why does this tradeoff exist?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A binary classifier is evaluated on 200 test samples: 60 are positive, 140 are negative. The results are:

```
TP = 42,   FN = 18,   FP = 14,   TN = 126
```

**(a)** Verify the counts are consistent — confirm TP + FN = 60 and FP + TN = 140.

&nbsp;

&nbsp;

**(b)** Compute accuracy.

&nbsp;

&nbsp;

**(c)** Compute precision and recall.

&nbsp;

&nbsp;

&nbsp;

**(d)** Compute F1.

&nbsp;

&nbsp;

&nbsp;

**(e)** Now suppose the threshold is lowered so that 10 of the FN cases are now correctly predicted positive, but this also causes 20 additional FP cases. Recompute the new confusion matrix, precision, and recall. Has F1 improved or worsened?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** Two classifiers are compared on a fraud detection task (1% of transactions are fraudulent). Classifier A has AUC-ROC = 0.92. Classifier B has average precision (AUC-PR) = 0.61. A manager argues Classifier A is clearly better because 0.92 > 0.61. Construct a counter-argument — explain why average precision may be the more informative metric here, and what the two numbers actually tell you about each classifier's ability to find fraud.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** You are building a classifier to screen job applications for a shortlisting system. False positives (unqualified candidates shortlisted) waste interviewer time. False negatives (qualified candidates rejected) mean good candidates are lost. You cannot optimise both simultaneously. Describe how you would decide which metric to prioritise, what information you would need from the business to make that decision, and what threshold strategy you would use.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
