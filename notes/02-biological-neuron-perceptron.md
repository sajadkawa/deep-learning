# 02 — Biological Neuron → Artificial Neuron → Perceptron

---

## Part 1: Biological Neuron

The artificial neuron is directly inspired by how a biological neuron works. Understanding the biology gives the math a reason to exist.

### Structure of a Biological Neuron

The diagram below shows the key anatomical parts of a biological neuron and the direction of signal flow.

![Structure of a Biological Neuron](images/biological_neuron.png)

| Biological Part | Role |
|---|---|
| Dendrites | Receive input signals from other neurons |
| Cell Body (Soma) | Accumulates and processes signals |
| Axon | Transmits output signal |
| Axon Terminals | Pass signal to next neuron via synapse |
| Synapse | Connection point — controls signal strength |

### How it fires

A biological neuron does not fire for every signal it receives. It accumulates incoming signals. If the total exceeds a **threshold**, it fires an output signal down the axon.

```
Incoming signals accumulate
         ↓
Total signal < threshold  →  neuron stays silent
Total signal ≥ threshold  →  neuron FIRES
```

This is called the **all-or-nothing principle**.

---

## Part 2: The McCulloch-Pitts Neuron

Before Rosenblatt built the perceptron in 1958, Warren McCulloch and Walter Pitts proposed the first mathematical model of a neuron in 1943. It is simpler than the perceptron — and understanding it makes the perceptron's contribution clear.

### The Model

The McCulloch-Pitts (MP) neuron takes binary inputs and produces a binary output. The diagram below shows the structure: inputs weighted by fixed integers, summed, then compared to a fixed threshold.

![McCulloch-Pitts Neuron](images/mp_neuron.png)

| Component | Description |
|---|---|
| Inputs xᵢ | Binary only — 0 or 1 |
| Weights wᵢ | Fixed integers — set by the designer, not learned |
| Weighted sum z | z = Σ wᵢxᵢ (no bias term) |
| Threshold θ | Fixed — fires if z ≥ θ, silent otherwise |
| Output ŷ | Binary — 0 or 1 |

### The Rule

```
         ┌ 1   if z ≥ θ
ŷ  =     │
         └ 0   if z < θ
```

### Worked Example — AND Gate

Set weights w₁ = 1, w₂ = 1 and threshold θ = 2:

```
x₁  x₂  z = x₁ + x₂   z ≥ 2?   ŷ
 0   0       0            No      0
 0   1       1            No      0
 1   0       1            No      0
 1   1       2            Yes     1
```

The MP neuron computes AND correctly — with manually chosen weights and threshold.

### Worked Example — OR Gate

Set weights w₁ = 1, w₂ = 1 and threshold θ = 1:

```
x₁  x₂  z = x₁ + x₂   z ≥ 1?   ŷ
 0   0       0            No      0
 0   1       1            Yes     1
 1   0       1            Yes     1
 1   1       2            Yes     1
```

### Worked Example — NOT Gate

Single input, set weight w₁ = -1 and threshold θ = 0:

```
x₁   z = -x₁   z ≥ 0?   ŷ
 0      0         Yes     1
 1     -1         No      0
```

NOT is computed by using a negative (inhibitory) weight.

### What the MP Neuron Got Right

```
✓ Binary threshold logic mirrors the all-or-nothing principle of biological neurons
✓ Showed that logical functions (AND, OR, NOT) can be computed by neuron-like units
✓ Proved that networks of such units are computationally universal
✓ Established the mathematical framework that all later work builds on
```

### The Critical Limitation

```
Weights and threshold are fixed — set by hand
              ↓
The designer must already know the answer
              ↓
The system cannot learn from data
              ↓
Not a learning machine — a logic gate
```

This is the gap Rosenblatt closed. The perceptron keeps the same structure but makes the weights **adjustable** and introduces a **learning rule** that updates them from examples.

```
McCulloch-Pitts (1943):   fixed weights, no learning, binary inputs only
         ↓
Perceptron (1958):         learnable weights, learning rule, real-valued inputs
```

---

## Part 3: Artificial Neuron and the Perceptron

The perceptron (Rosenblatt, 1958) is the direct mathematical model of the biological neuron. There is no meaningful separation between "artificial neuron" and "perceptron" at this level — the perceptron IS the artificial neuron. We build it step by step.

### Mapping Biology to Math

| Biological | Artificial / Perceptron |
|---|---|
| Dendrites | Inputs x₁, x₂, ..., xₙ |
| Synapse strength | Weights w₁, w₂, ..., wₙ |
| Cell body accumulation | Weighted sum z = Σ wᵢxᵢ + b |
| Threshold / firing | Activation function f(z) |
| Axon output | Output ŷ = f(z) |
| Resting state | Bias term b |

The diagram below shows the perceptron structure: real-valued inputs, learnable weights, a bias, a weighted sum, and a differentiable activation function.

![Perceptron (Artificial Neuron)](images/perceptron_diagram.png)

### The Math

**Step 1 — Weighted sum (pre-activation):**

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

or in vector form:

z = wᵀx + b
```

**Step 2 — Activation (post-activation):**

```
ŷ = f(z)
```

---

### The Neuron Formula is a Line — y = mx + c

This is a connection most textbooks skip. Look at the weighted sum for two inputs:

```
z = w₁x₁ + w₂x₂ + b
```

Now compare to the equation of a straight line:

```
y = mx + c
```

They are the same structure:

```
  z  = w₁x₁  +  w₂x₂  +  b
  │     │           │       │
  │     └── slope ──┘       └── intercept
  │
 output (before activation)
```

For two inputs, the decision boundary `z = 0` becomes:

```
w₁x₁ + w₂x₂ + b = 0

Rearranging:
x₂ = -(w₁/w₂)x₁ - (b/w₂)
      └── slope ──┘  └── intercept
```

This is literally a straight line in the x₁-x₂ plane. One side gives z > 0 (neuron fires, class 1); the other gives z < 0 (neuron silent, class 0). A single perceptron can only ever draw one straight line to separate classes — exactly why it fails on XOR and why we need multiple layers.

### What is the Bias?

The bias b is the intercept of that line:

```
Without bias:   x₂ = -(w₁/w₂)x₁         (line always passes through origin)
With bias:      x₂ = -(w₁/w₂)x₁ - b/w₂  (line can shift freely)
```

The diagram below shows the geometric effect: without bias every decision boundary passes through the origin; with bias it can shift freely.

![Decision Boundary: Effect of the Bias Term](images/decision_boundary_bias.png)

Without bias, the decision boundary is forced through the origin — severely limiting what the neuron can learn.

---

### Activation: The Step Function

The original perceptron uses a **step function** (Heaviside function):

```
         ┌ 1   if z ≥ 0
f(z)  =  │
         └ 0   if z < 0
```

The diagram below shows the step function: output jumps from 0 to 1 at z = 0.

![Step Function (Heaviside)](images/step_function.png)

The neuron outputs 1 (fires) if the weighted sum crosses zero, otherwise 0.

---

### Why the Step Function is Not Enough — and Why Activation Functions Exist

The step function has a critical problem for training:

```
Step function:
  - Output is only 0 or 1
  - Not differentiable at z = 0
  - Derivative is 0 everywhere else
         ↓
  Cannot use gradient descent
  Cannot express probabilities
```

But there is a deeper problem that goes beyond the step function — it applies to **any linear activation**, including no activation at all.

**The linearity collapse problem:**

Suppose we stack two neurons with no activation function (or a linear one):

```
Layer 1 output:   z₁ = W₁x + b₁
Layer 2 output:   z₂ = W₂z₁ + b₂
                     = W₂(W₁x + b₁) + b₂
                     = (W₂W₁)x + (W₂b₁ + b₂)
                     = Wx + b        ← still a single line
```

Now stack 10 layers:

```
Layer 10 output = W₁₀(W₉(...(W₁x + b₁)...)) + b₁₀
                = Wx + b             ← still a single line
```

**No matter how many layers you stack, without non-linear activation, the entire network collapses to a single linear transformation — equivalent to one neuron.**

```
10 layers, no activation:          1 neuron:

x → L1 → L2 → ... → L10 → ŷ  ≡  x → [Wx + b] → ŷ
```

Depth becomes meaningless. The network cannot learn anything a single neuron cannot.

This is the real reason activation functions exist:

```
✓ Non-linearity        →  breaks the collapse, allows complex boundaries
✓ Differentiability    →  allows gradient descent to work
✓ Expressiveness       →  output can represent probabilities, ranges, etc.
```

Examples: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax — covered in full detail in **Note 04**.  
For now, we use the step function to understand the perceptron fully.

---

## Part 5: Forward Pass — How a Perceptron Computes

Given inputs, weights, and bias — the perceptron computes a prediction in two steps:

```
FORWARD PASS

  Inputs + Weights
        ↓
  z = wᵀx + b        ← weighted sum
        ↓
  ŷ = f(z)           ← apply activation (step function)
        ↓
  Output ŷ
```

### Example

```
x₁ = 1,  x₂ = 0
w₁ = 0.5, w₂ = 0.5, b = -0.5

z = (0.5)(1) + (0.5)(0) + (-0.5)
z = 0.5 + 0 - 0.5
z = 0

ŷ = f(0) = 1     (since z ≥ 0)
```

---

## Part 6: Perceptron Learning Rule

The perceptron learns by adjusting weights when it makes a wrong prediction.

### The Rule

```
For each training example (x, y):

  1. Compute prediction:   ŷ = f(wᵀx + b)
  2. Compute error:        e = y - ŷ
  3. Update weights:       wᵢ ← wᵢ + η × e × xᵢ
  4. Update bias:          b  ← b  + η × e
```

Where:
- y  = true label
- ŷ  = predicted label
- e  = error (0 if correct, ±1 if wrong)
- η  = learning rate (small positive number, e.g. 0.1)

### Intuition

```
Prediction correct  →  e = 0  →  no update
Prediction too low  →  e = +1 →  weights increase (push output up)
Prediction too high →  e = -1 →  weights decrease (push output down)
```

---

## Part 7: Full Training Example — OR Function

### OR Truth Table

| x₁ | x₂ | y (OR) |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

### Geometric View

```
x₂
 1 │  ●(0,1)    ●(1,1)
   │
 0 │  ○(0,0)    ●(1,0)
   └──────────────────── x₁
      0          1

● = class 1 (OR = 1)
○ = class 0 (OR = 0)
```

A single straight line can separate ○ from ● — OR is **linearly separable**.

![OR Function — Data Points and Decision Boundary](images/or_function.png)

### Initial Setup

```
w₁ = 0.0,  w₂ = 0.0,  b = 0.0
η  = 0.1
Step function: ŷ = 1 if z ≥ 0.5, else 0
  (we use threshold 0.5 for cleaner hand computation)
```

---

### Epoch 1

**Sample 1: x = [0, 0], y = 0**

```
z  = (0.0)(0) + (0.0)(0) + 0.0 = 0.0
ŷ  = 0     (0.0 < 0.5)
e  = 0 - 0 = 0
→  No update
w₁ = 0.0,  w₂ = 0.0,  b = 0.0
```

**Sample 2: x = [0, 1], y = 1**

```
z  = (0.0)(0) + (0.0)(1) + 0.0 = 0.0
ŷ  = 0     (0.0 < 0.5)
e  = 1 - 0 = 1
w₁ ← 0.0 + 0.1 × 1 × 0 = 0.0
w₂ ← 0.0 + 0.1 × 1 × 1 = 0.1
b  ← 0.0 + 0.1 × 1     = 0.1
w₁ = 0.0,  w₂ = 0.1,  b = 0.1
```

**Sample 3: x = [1, 0], y = 1**

```
z  = (0.0)(1) + (0.1)(0) + 0.1 = 0.1
ŷ  = 0     (0.1 < 0.5)
e  = 1 - 0 = 1
w₁ ← 0.0 + 0.1 × 1 × 1 = 0.1
w₂ ← 0.1 + 0.1 × 1 × 0 = 0.1
b  ← 0.1 + 0.1 × 1     = 0.2
w₁ = 0.1,  w₂ = 0.1,  b = 0.2
```

**Sample 4: x = [1, 1], y = 1**

```
z  = (0.1)(1) + (0.1)(1) + 0.2 = 0.4
ŷ  = 0     (0.4 < 0.5)
e  = 1 - 0 = 1
w₁ ← 0.1 + 0.1 × 1 × 1 = 0.2
w₂ ← 0.1 + 0.1 × 1 × 1 = 0.2
b  ← 0.2 + 0.1 × 1     = 0.3
w₁ = 0.2,  w₂ = 0.2,  b = 0.3
```

**End of Epoch 1:** w₁ = 0.2, w₂ = 0.2, b = 0.3

---

### Epoch 2

**Sample 1: x = [0, 0], y = 0**

```
z  = (0.2)(0) + (0.2)(0) + 0.3 = 0.3
ŷ  = 0     (0.3 < 0.5)
e  = 0 - 0 = 0
→  No update
```

**Sample 2: x = [0, 1], y = 1**

```
z  = (0.2)(0) + (0.2)(1) + 0.3 = 0.5
ŷ  = 1     (0.5 ≥ 0.5)
e  = 1 - 1 = 0
→  No update
```

**Sample 3: x = [1, 0], y = 1**

```
z  = (0.2)(1) + (0.2)(0) + 0.3 = 0.5
ŷ  = 1     (0.5 ≥ 0.5)
e  = 1 - 1 = 0
→  No update
```

**Sample 4: x = [1, 1], y = 1**

```
z  = (0.2)(1) + (0.2)(1) + 0.3 = 0.7
ŷ  = 1     (0.7 ≥ 0.5)
e  = 1 - 1 = 0
→  No update
```

**End of Epoch 2:** No updates — perceptron has converged.

---

### Final Verification

```
Final weights: w₁ = 0.2,  w₂ = 0.2,  b = 0.3

x=[0,0]: z = 0.3          → ŷ = 0  ✓  (y=0)
x=[0,1]: z = 0.5          → ŷ = 1  ✓  (y=1)
x=[1,0]: z = 0.5          → ŷ = 1  ✓  (y=1)
x=[1,1]: z = 0.7          → ŷ = 1  ✓  (y=1)
```

All correct. The perceptron learned the OR function in 2 epochs.

### Decision Boundary

The learned decision boundary is:

```
w₁x₁ + w₂x₂ + b = 0.5   (our threshold)
0.2x₁ + 0.2x₂ + 0.3 = 0.5
0.2x₁ + 0.2x₂ = 0.2
x₁ + x₂ = 1
```

The diagram below shows the four OR points and the learned decision boundary x₁ + x₂ = 1 that correctly separates them.

![OR — Learned Decision Boundary](images/or_decision_boundary.png)

---

## Part 8: Why XOR Fails — The Limitation of a Single Perceptron

### XOR Truth Table

| x₁ | x₂ | y (XOR) |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### Geometric Intuition

The coordinate plane below shows the four XOR points. The two class-1 points sit on opposite corners — no single straight line can separate them from the class-0 points.

![XOR — Coordinate Plane](images/xor_coordinate_plane.png)
The three panels below show three different line attempts — every one misclassifies at least one point.

![XOR is Not Linearly Separable — No Single Line Works](images/xor_attempts.png)

**It is impossible.** No single straight line can separate the two classes of XOR. XOR is **not linearly separable**.

### Mathematical Proof

A perceptron computes:

```
ŷ = 1  if  w₁x₁ + w₂x₂ + b ≥ 0
ŷ = 0  otherwise
```

For XOR to be solved, we need all four conditions simultaneously:

```
(0,0) → 0:   b < 0                    ... (i)
(0,1) → 1:   w₂ + b ≥ 0              ... (ii)
(1,0) → 1:   w₁ + b ≥ 0              ... (iii)
(1,1) → 0:   w₁ + w₂ + b < 0         ... (iv)
```

From (ii):  w₂ ≥ -b > 0  (using i)
From (iii): w₁ ≥ -b > 0

Adding (ii) and (iii):

```
w₁ + w₂ + 2b ≥ 0
w₁ + w₂ + b  ≥ -b > 0
```

But condition (iv) requires:

```
w₁ + w₂ + b < 0
```

**Contradiction.** No values of w₁, w₂, b can satisfy all four conditions.

### The Conclusion

```
Single Perceptron
       ↓
Can only learn LINEARLY SEPARABLE functions
       ↓
OR  ✓   AND  ✓   NAND  ✓
XOR ✗   XNOR ✗
```

This was the core criticism in Minsky & Papert's 1969 book "Perceptrons" — it nearly killed neural network research for a decade.

The solution: **stack multiple layers** → Multilayer Perceptron (Note 03).

---

## Summary

```
BIOLOGICAL NEURON          ARTIFICIAL NEURON / PERCEPTRON
──────────────────         ───────────────────────────────
Dendrites                  Inputs x
Synapse strength           Weights w
Cell body                  Weighted sum z = wᵀx + b  ≡  y = mx + c
Threshold / firing         Activation function f(z)
Axon output                Output ŷ = f(z)
```

Key takeaways:

- The McCulloch-Pitts neuron (1943) was the first mathematical model of a neuron — binary inputs, fixed weights, hard threshold, no learning
- The perceptron (1958) extended MP by making weights learnable from data — that is the key advance
- Artificial neuron and perceptron are the same thing — the perceptron is the artificial neuron with a step function
- The neuron formula `z = wᵀx + b` is a straight line — the decision boundary is literally `y = mx + c`
- Without non-linear activation, stacking any number of layers collapses to a single line — depth is meaningless
- A perceptron learns using the perceptron learning rule: `w ← w + η(y - ŷ)x`
- It can solve linearly separable problems (OR, AND)
- It **cannot** solve non-linearly separable problems (XOR) — proven geometrically and mathematically
- Stacking multiple perceptrons into layers solves the XOR problem → **Note 03**
