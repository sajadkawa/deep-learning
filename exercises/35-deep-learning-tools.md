# Exercise 35 — Deep Learning Tools: PyTorch, TensorFlow, Keras

## Section A — Multiple Choice

**1.** What does automatic differentiation in a deep learning framework do?

- a) Automatically selects the best learning rate for training
- b) Builds a computational graph and computes gradients for all parameters via the chain rule
- c) Converts Python code to GPU-optimised CUDA kernels
- d) Automatically determines the optimal network architecture

&nbsp;

**2.** In PyTorch, what is the correct order of operations in a single training iteration?

- a) `backward()` → `zero_grad()` → `forward()` → `step()`
- b) `zero_grad()` → `forward()` → `backward()` → `step()`
- c) `forward()` → `step()` → `backward()` → `zero_grad()`
- d) `forward()` → `zero_grad()` → `step()` → `backward()`

&nbsp;

**3.** What is the purpose of calling `model.eval()` before running validation in PyTorch?

- a) It freezes all model parameters so they cannot be updated
- b) It switches layers like dropout and batch normalisation to inference mode
- c) It moves the model to CPU for evaluation
- d) It clears the gradient buffers before the validation pass

&nbsp;

**4.** In TensorFlow 2, what is the role of `tf.GradientTape`?

- a) It records operations performed on tensors so gradients can be computed afterwards
- b) It compiles a Python function to a static computation graph
- c) It moves tensors between CPU and GPU automatically
- d) It replaces the need for an optimiser by computing weight updates directly

&nbsp;

**5.** Which of the following best describes the difference between PyTorch and Keras?

- a) PyTorch uses static graphs; Keras uses dynamic graphs
- b) PyTorch requires a manual training loop; Keras abstracts training into `compile` and `fit`
- c) Keras supports GPU training; PyTorch does not
- d) PyTorch is only for research; Keras cannot be used in production

&nbsp;

---

## Section B — Short Answer

**1.** Explain what a dynamic computation graph is and why it makes debugging easier compared to a static graph.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** What does `optimiser.zero_grad()` do and why must it be called before `loss.backward()` in every training iteration?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**3.** Describe the `DataLoader` pattern in PyTorch. What does `num_workers > 0` do and why does it matter for GPU training?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**4.** What does the `@tf.function` decorator do in TensorFlow and what is the trade-off between using it and not using it?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section C — Calculation and Trace

The following PyTorch code defines a two-layer network and performs one training step. Trace through it and answer the questions.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model     = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[3.0]])

logits = model(x)
loss   = criterion(logits, y)
optimiser.zero_grad()
loss.backward()
optimiser.step()
```

**(a)** What is the shape of the output of the first `nn.Linear(2, 3)` layer for this input?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(b)** What is the shape of the final model output `logits`?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(c)** What does `criterion(logits, y)` compute? Write the formula for this specific case.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(d)** After `loss.backward()`, which tensors have their `.grad` attribute populated? Which do not?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**(e)** What would happen if `optimiser.zero_grad()` were removed and the training loop ran for a second iteration?

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

---

## Section D — Critical Thinking

**1.** A student implements a custom training loop in PyTorch and notices that the validation loss is lower than the training loss throughout training. They suspect a bug. Identify the most likely cause and explain how to fix it.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

**2.** A team is building a deep learning system for a hospital. One engineer argues for PyTorch because it is easier to debug and iterate on. Another argues for TensorFlow/Keras because it integrates with the hospital's existing Google Cloud infrastructure. Evaluate both arguments and describe the factors that should drive the final decision.

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;
