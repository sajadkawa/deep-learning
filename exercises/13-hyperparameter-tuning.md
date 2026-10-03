# Exercise 13 — Hyperparameter Tuning

---

## Section A — Multiple Choice

**1.** Which of the following is a hyperparameter, not a parameter?

- a) The weight connecting neuron 3 in layer 2 to neuron 1 in layer 3
- b) The bias of the output neuron
- c) The dropout rate applied after the first hidden layer
- d) The gradient of the loss with respect to a weight

&nbsp;

**2.** A model is trained with a very large learning rate. What is the most likely symptom?

- a) Training loss decreases very slowly over many epochs
- b) Training loss oscillates or diverges rather than decreasing steadily
- c) The model overfits the training data immediately
- d) Gradients vanish in the early layers

&nbsp;

**3.** You double the batch size from 64 to 128. According to the linear scaling rule, what should you do to the learning rate?

- a) Halve it
- b) Keep it the same
- c) Double it
- d) Square it

&nbsp;

**4.** Early stopping monitors validation loss and stops training when it stops improving. What must be done at the end of training to ensure the best model is used?

- a) Run one final epoch on the full dataset including validation data
- b) Restore the model weights from the epoch with the lowest validation loss
- c) Reduce the learning rate and continue training for 10 more epochs
- d) Retrain from scratch using the number of epochs that triggered early stopping

&nbsp;

**5.** Why does random search outperform grid search when only a few hyperparameters matter?

- a) Random search evaluates more total configurations than grid search
- b) Random search avoids evaluating the same configuration twice
- c) Random search explores more distinct values of the important dimensions with the same number of evaluations
- d) Random search uses a smarter stopping criterion than grid search

&nbsp;

---

## Section B — Short Answer

**1.** Explain why the number of epochs is not a hyperparameter that should be tuned by grid search or random search. What is the standard approach instead?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Small batch sizes are said to find flatter minima than large batch sizes. Explain the mechanism — what property of small batch training leads to this outcome?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** The learning rate range test starts with a very small η and increases it exponentially over a short run. Describe what you observe in the loss curve and how you use it to choose a starting learning rate.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** A practitioner changes the learning rate, dropout rate, and batch size simultaneously and observes that validation loss improves. What is wrong with this approach, and what should they do instead?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

A model is trained with early stopping (patience = 3). The validation loss at each epoch is:

```
Epoch:       1      2      3      4      5      6      7      8      9     10
Val loss:  1.42   1.18   0.97   0.89   0.91   0.88   0.93   0.95   0.99  1.02
```

**(a)** Identify the epoch with the lowest validation loss. This is the epoch whose weights would be saved.

&nbsp;

&nbsp;

**(b)** Trace the patience counter at each epoch. Show when it resets to 0 and when it increments.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** At which epoch does early stopping trigger? How many epochs of training were saved compared to running all 10 epochs?

&nbsp;

&nbsp;

&nbsp;

**(d)** If patience were set to 1 instead of 3, at which epoch would training stop? What is the risk of using a very low patience value?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A practitioner follows the tuning order from Note 13: learning rate first, then batch size, then regularization, then architecture. They find a learning rate that works well with batch size 32. They then increase batch size to 256 and apply the linear scaling rule, but training becomes unstable. Identify two reasons why the linear scaling rule might fail at large batch sizes, and suggest what they should do instead.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** Bayesian optimization is described as more sample-efficient than random search. Despite this, random search is more commonly used in practice. Give two reasons why a practitioner might prefer random search over Bayesian optimization even when sample efficiency matters.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
