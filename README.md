---
# MNIST Classification using Neural Network from Scratch (NumPy)

## 📌 Overview

This project implements a **fully connected neural network from scratch using NumPy** to classify handwritten digits from the **MNIST dataset**.
No deep learning frameworks (TensorFlow / PyTorch) are used — all computations including **forward propagation, backpropagation, and gradient descent** are implemented manually.

The model achieves **~90%+ test accuracy** on MNIST using a CPU-only environment.
---

## Model Architecture

- **Input Layer**: 784 neurons (28×28 flattened image)
- **Hidden Layer 1**: 128 neurons (ReLU)
- **Hidden Layer 2**: 64 neurons (ReLU)
- **Output Layer**: 10 neurons (Softmax)

```
784 → 128 → 64 → 10
```

---

## Features Implemented

- Fully connected (dense) neural network
- He weight initialization
- ReLU and Softmax activations
- Forward propagation
- Backpropagation using chain rule
- **Vanilla Batch Gradient Descent**
- Categorical Cross-Entropy loss
- Unit tests for forward and backward passes
- Training and evaluation on MNIST
- Loss and accuracy visualization

---

## Project Structure

```
MNIST_NN_FROM_SCRATCH/
│
├── model.py        # Neural network implementation
├── train.py        # Training & evaluation script
├── tests.py        # Unit tests
├── README.md       # Project documentation
├── requirements.txt
└── .gitignore
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Required libraries:**

- Python 3.x
- NumPy
- scikit-learn
- pandas
- matplotlib

---

## How to Run

### Run Unit Tests (Sanity Check)

```bash
python tests.py
```

Expected output:

```
All tests passed!
```

---

### Train the Model

```bash
python train.py
```

During training, you will see:

- Epoch-wise loss and accuracy
- Final test accuracy
- Loss and accuracy plots

---

## Results

- **Training Accuracy**: ~90–92%
- **Test Accuracy**: ~90–93%
- **Epochs**: 100-120
- **Optimizer**: Vanilla Batch Gradient Descent

---

## Optimization Details

The network uses **vanilla gradient descent**, updating parameters as:

[
W = W - \alpha \frac{\partial L}{\partial W}
]

For the output layer:

- **Softmax + Categorical Cross-Entropy**
- Gradient simplifies to:
  [
  \frac{\partial L}{\partial Z} = \hat{y} - y
  ]

Hidden layers use ReLU activation with standard backpropagation.

---

## Testing

Unit tests verify:

- Forward pass output shapes
- Backpropagation gradient dimensions
- Parameter-gradient consistency

This ensures mathematical correctness before training.

---

## Key Learnings

- Understanding neural networks at a mathematical level
- Importance of correct loss-activation pairing
- Backpropagation mechanics
- Gradient descent behavior on real data
- Debugging training instability

---

## Future Improvements

- Mini-batch gradient descent
- Adam optimizer
- Regularization (L2, Dropout)
- CNN implementation for higher accuracy

---

## Author

**Mukhesh Kumar Reddy**

---
