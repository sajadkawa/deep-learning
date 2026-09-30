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

---

## Unit II — Training Deep Networks

### 09 — Deep Feedforward Network Architectures
- What makes a network "deep" — depth vs width, representational hierarchy
- Feedforward architecture — layers, connections, information flow
- Depth and width tradeoffs — when to go deeper vs wider
- Parameter count — how depth and width affect model size
- Computational graph — nodes, edges, forward evaluation order
- Universal approximation revisited — depth gives exponential efficiency over width
- Practical architecture design — input/output layer sizing, hidden layer choices

### 10 — Weight Initialization Methods
- Why initialization matters — symmetry breaking, gradient flow at the start
- Zero initialization failure — all neurons learn the same thing
- Random initialization — breaks symmetry but scale matters
- Xavier/Glorot initialization — variance scaled to fan-in and fan-out
- He initialization — variance scaled for ReLU networks
- Effect of bad initialization on vanishing and exploding gradients

### 11 — Batch Normalization and Dropout
- Internal covariate shift — why layer inputs shift during training
- Batch normalization — normalize, scale, shift per mini-batch
- Where to place batch norm — before or after activation
- Batch norm at inference — running mean and variance
- Dropout revisited — inverted dropout, effect on generalization
- Batch norm vs dropout — what each regularizes and when to combine them

### 12 — Optimization Algorithms: SGD, Adam, RMSProp
- Limitations of vanilla gradient descent — saddle points, ravines, slow convergence
- Momentum — accumulating gradient history to accelerate learning
- RMSProp — adaptive learning rate per parameter using gradient magnitude
- Adam — combines momentum and RMSProp, bias correction
- Learning rate schedules — step decay, cosine annealing
- Choosing an optimizer in practice

### 13 — Hyperparameter Tuning
- What counts as a hyperparameter — learning rate, batch size, epochs, architecture choices
- Learning rate — most critical hyperparameter, effect on convergence
- Batch size — effect on gradient noise, generalization, and compute
- Epochs and early stopping — when to stop training
- Grid search vs random search vs manual tuning
- Practical tuning workflow — what to tune first and why

### 14 — Model Evaluation: Metrics
- Why accuracy alone is insufficient — class imbalance problem
- Confusion matrix — TP, TN, FP, FN
- Precision and recall — tradeoff and when each matters
- F1 score — harmonic mean of precision and recall
- ROC curve and AUC — threshold-independent performance measure
- Choosing the right metric for the task

### 15 — Hopfield Network
- Associative memory — retrieving a pattern from a partial or noisy input
- Hopfield network architecture — fully connected, symmetric weights, no self-connections
- Energy function — how the network settles into stable states
- Hebbian weight rule for storing patterns
- Synchronous vs asynchronous update
- Storage capacity — why the network degrades beyond ~0.14N patterns

### 16 — Boltzmann Machine
- Stochastic neurons — probabilistic firing, temperature parameter
- Boltzmann distribution — probability of a state proportional to e^(−E/T)
- Visible and hidden units — learning a generative model of data
- Contrastive Hebbian learning — positive and negative phase
- Restricted Boltzmann Machine (RBM) — tractable training, no hidden-to-hidden connections
- RBMs as building blocks of deep belief networks

### 17 — Kohonen's Self-Organizing Feature Maps
- Unsupervised topology-preserving mapping — neighbourhood structure preserved
- SOM architecture — input layer, competitive output layer (map grid)
- Winner-takes-all selection — best matching unit (BMU)
- Neighbourhood function — Gaussian decay, updating BMU and neighbours
- Learning rate and neighbourhood decay over time
- What a trained SOM represents — topographic map of input space

### 18 — Associative Memory
- Auto-associative memory — pattern completes itself from partial input
- Hetero-associative memory — one pattern retrieves a different stored pattern
- Hebbian storage rule for associative memories
- Retrieval dynamics — convergence to stored attractor
- Performance measures — storage capacity, error correction ability, crosstalk
- Relationship to Hopfield networks and modern attention mechanisms

