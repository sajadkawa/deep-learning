# Deep Learning — Syllabus

Postgraduate level course.

---

## Unit I — Foundations of Neural Networks

### 01 — AI Foundations, ML Paradigms & Deep Learning
- What is AI — rational agents, Norvig & Russell's definition
- Symbolic AI and expert systems — the knowledge bottleneck
- Search and planning — state spaces, where it works, state space explosion
- Traditional programming vs machine learning — rules vs learned patterns
- Types of learning: supervised, unsupervised, reinforcement
- Types of tasks: classification, regression, clustering
- Data fundamentals — sample, feature, label, variable types
- ML vs Deep Learning — feature engineering vs feature learning
- Why DL became viable around 2012 — data, GPU compute, algorithms

### 02 — Biological Neuron → McCulloch-Pitts Unit → Perceptron
- Biological neuron — structure and all-or-nothing firing principle
- McCulloch-Pitts neuron — binary logic, fixed weights, hard threshold, no learning
- McCulloch-Pitts as a logic gate — AND, OR, NOT with worked examples
- Perceptron — learnable weights, bias, step function, learning rule
- Decision boundary as a straight line — geometry of the neuron formula
- Bias shifts the decision boundary freely away from the origin
- Why non-linear activation is necessary — the linearity collapse problem
- Full OR training trace — convergence demonstrated with real numbers
- XOR failure — a single perceptron can only solve linearly separable problems

### 03 — Multilayer Perceptron (MLP)
- Why a single perceptron fails on XOR — linear boundary insufficient
- MLP architecture: input layer, hidden layers, output layer
- Forward pass: z = Wᵀx + b, a = f(z), layer by layer
- Representation power — each layer learns increasingly abstract features
- Universal approximation theorem — existence vs. practice, and depth vs. width
- XOR solved with a 2-layer MLP — worked example and feature space warping
- The credit assignment problem — why the perceptron rule fails on hidden layers

### 04 — Activation Functions
- Why activation must be non-linear — without it depth is meaningless
- Sigmoid — range (0,1), vanishing gradient problem
- Tanh — zero-centred, still saturates
- ReLU — sparse activation, dying ReLU problem
- Leaky ReLU — fixing dying ReLU
- Softmax — probability distribution for multiclass output
- How to choose activation function per layer

### 05 — Loss Functions
- What a loss function measures — distance between prediction and truth
- MSE — regression, penalises large errors heavily
- Binary Cross-Entropy — binary classification
- Categorical Cross-Entropy — multiclass classification
- Why cross-entropy outperforms MSE for classification — gradient behaviour
- Loss as the signal that drives all learning

### 06 — Gradient Descent and Backpropagation
- Loss surface — parameters form a landscape, training finds the minimum
- Gradient descent: w ← w − η∇L
- Learning rate — too large overshoots, too small converges slowly
- Batch GD vs Stochastic GD vs Mini-batch GD
- Backpropagation — chain rule applied layer by layer to compute ∂L/∂w
- Forward pass computes predictions; backward pass computes gradients
- Vanishing gradients — early layers learn slowly
- Exploding gradients — gradient clipping as fix

### 07 — Overfitting, Underfitting & Bias-Variance Tradeoff
- Overfitting — memorises training data, fails on unseen data
- Underfitting — model too simple, high bias
- Bias-variance tradeoff — complexity reduces bias but increases variance
- Training, validation, test sets — roles and why all three are needed
- Learning curves — diagnosing fit from training vs validation loss
- Regularization: L1, L2, dropout, early stopping

### 08 — Learning Mechanisms: Hebbian, Competitive, Boltzmann
- Hebbian learning — neurons that fire together wire together
- Competitive learning — winner-takes-all, basis of clustering and SOM
- Boltzmann learning — probabilistic energy-based network
- How these differ from the perceptron rule — no error signal, no labeled data
- Where each appears in modern networks

