# MNIST Classification – Neural Network from Scratch

## Overview
This project implements a fully connected neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset.

## Architecture
- Input: 784 neurons
- Hidden Layer 1: 128 (ReLU)
- Hidden Layer 2: 64 (ReLU)
- Output: 10 (Sigmoid)

## Features
- Manual forward pass and backpropagation
- Gradient descent optimization
- Configurable layers, activations, and learning rate
- Unit tests for correctness

## Results
- Training accuracy: ~92%
- Test accuracy: ≥90%

## Usage
```bash
pip install -r requirements.txt
python train.py
python tests.py
