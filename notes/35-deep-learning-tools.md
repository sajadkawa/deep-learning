# Note 35 — Deep Learning Tools: PyTorch, TensorFlow, Keras

## Part 1: Why Frameworks Exist

Writing a deep learning model from scratch requires implementing forward passes, computing gradients by hand, managing memory, and writing GPU kernels. This is feasible for a single perceptron — it is not feasible for a 50-layer ResNet.

Deep learning frameworks solve three problems:

**Automatic differentiation.** The framework builds a computational graph of every operation. When you call `.backward()`, it traverses the graph in reverse and computes gradients for every parameter automatically using the chain rule. You write the forward pass; the framework handles the backward pass.

**GPU acceleration.** Moving a tensor to a GPU and running operations on it is a single line of code. The framework handles memory allocation, kernel dispatch, and synchronisation.

**Modular layers.** Common building blocks — linear layers, convolutions, batch normalisation, optimisers, loss functions — are pre-implemented, tested, and optimised. You compose them rather than reimplement them.

> Frameworks do not change what deep learning is — they change how much of it you have to implement yourself. The concepts in Notes 01–34 are the same regardless of which framework you use.

---

## Part 2: PyTorch

PyTorch (Meta AI, 2016) uses a **dynamic computation graph** (define-by-run): the graph is built as operations execute, not before. This makes debugging natural — you can inspect tensors, use Python control flow, and set breakpoints anywhere in the forward pass.

### Core Abstractions

**`torch.Tensor`** — the fundamental data structure. A multi-dimensional array that lives on CPU or GPU, tracks gradients, and supports all arithmetic operations.

**`torch.nn.Module`** — the base class for all models. Subclass it, define `__init__` (layers) and `forward` (computation). PyTorch handles parameter registration, `.to(device)`, `.train()`/`.eval()` mode switching, and state dict serialisation.

**`torch.optim`** — optimisers (SGD, Adam, RMSProp). Call `.zero_grad()`, `.backward()`, `.step()` in every training iteration.

**`torch.utils.data.DataLoader`** — wraps a `Dataset` and handles batching, shuffling, and parallel data loading.

### Standard Training Loop

```python
model     = MyModel().to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss   = criterion(logits, y_batch)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        # validation loop
```

The four lines inside the training loop are always the same: forward pass, compute loss, zero gradients, backward pass, update weights.

### GPU Training

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = model.to(device)
x      = x.to(device)
```

Moving a model or tensor to the GPU copies its data to GPU memory. All subsequent operations on that tensor run on the GPU. Results must be moved back to CPU (`.cpu()`) before converting to NumPy.

### DataLoader Pattern

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(MyDataset(X_train, y_train),
                    batch_size=32, shuffle=True, num_workers=2)
```

`num_workers > 0` loads batches in parallel background processes, preventing the GPU from waiting for data.

---

## Part 3: TensorFlow

TensorFlow (Google, 2015) originally used a **static computation graph** (define-then-run): you first define the entire graph symbolically, then execute it in a session. TensorFlow 2 (2019) switched to eager execution by default — operations execute immediately, like PyTorch.

### Core Abstractions

**`tf.Tensor`** — equivalent to `torch.Tensor`. Immutable by default in eager mode.

**`tf.Variable`** — a mutable tensor that holds model parameters and is tracked by the automatic differentiation engine.

**`tf.GradientTape`** — records operations for automatic differentiation. Gradients are computed by calling `.gradient()` on the tape.

### Standard Training Step in TensorFlow

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn   = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function   # compiles to a static graph for speed
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss   = loss_fn(y_batch, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

The `@tf.function` decorator traces the function once and compiles it to a static graph, giving performance close to TF1 while keeping the readable Python syntax.

---

## Part 4: Keras

Keras is a high-level API that runs on top of TensorFlow (and historically on Theano and CNTX). Since TensorFlow 2, Keras is the official high-level interface: `tf.keras`.

### Sequential API

For simple feedforward models:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,  activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_val, y_val))
```

### Functional API

For models with branching, skip connections, or multiple inputs/outputs:

```python
inputs  = tf.keras.Input(shape=(784,))
x       = tf.keras.layers.Dense(128, activation='relu')(inputs)
x       = tf.keras.layers.Dense(64,  activation='relu')(x)
outputs = tf.keras.layers.Dense(10,  activation='softmax')(x)
model   = tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Keras vs PyTorch

| Property | Keras (tf.keras) | PyTorch |
|---|---|---|
| API level | High-level — `compile`, `fit`, `evaluate` | Lower-level — manual training loop |
| Flexibility | Less flexible for custom training | Full control over every step |
| Debugging | Harder inside `@tf.function` | Natural — Python debugger works |
| Deployment | TF Serving, TFLite, TF.js | TorchScript, ONNX |
| Research use | Less common | Dominant in research |
| Production use | Common (Google ecosystem) | Growing |

---

## Part 5: Common Workflow and GPU Training

Regardless of framework, the workflow is the same:

1. **Define the model** — architecture, layers, activations
2. **Compile / configure** — loss function, optimiser, metrics
3. **Prepare data** — `DataLoader` (PyTorch) or `tf.data.Dataset` (TF)
4. **Train** — iterate over batches, forward pass, loss, backward pass, update
5. **Evaluate** — run on held-out test set, report metrics
6. **Deploy** — serialise model, serve predictions

### Saving and Loading Models

**PyTorch:**
```python
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))
```

**Keras:**
```python
model.save('model.keras')
model = tf.keras.models.load_model('model.keras')
```

### Choosing a Framework

| Situation | Recommendation |
|---|---|
| Research, custom architectures, debugging | PyTorch |
| Production deployment in Google ecosystem | TensorFlow / Keras |
| Rapid prototyping, teaching | Keras |
| This course | PyTorch — used throughout Notes 09–34 |

---

## Summary

- Deep learning frameworks provide automatic differentiation, GPU acceleration, and modular pre-implemented layers — eliminating the need to implement backpropagation and GPU kernels manually.
- PyTorch uses a dynamic computation graph (define-by-run): the graph is built during execution, making debugging natural; the standard training loop is: forward pass → loss → `zero_grad` → `backward` → `step`.
- TensorFlow uses eager execution by default in TF2; `tf.GradientTape` records operations for differentiation; `@tf.function` compiles a function to a static graph for performance.
- Keras is the high-level API on top of TensorFlow: `Sequential` for simple models, `Functional` for branching architectures; `compile` + `fit` abstracts the training loop.
- The `DataLoader` pattern (PyTorch) and `tf.data.Dataset` (TF) handle batching, shuffling, and parallel data loading; `num_workers > 0` prevents GPU starvation.
- PyTorch dominates research; TensorFlow/Keras is common in production; both implement the same underlying concepts covered in Notes 01–34.
