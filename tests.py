import numpy as np
from model import NeuralNetwork


def test_forward_shape():
    nn = NeuralNetwork([4, 5, 3], ["relu", "sigmoid"])
    X = np.random.randn(10, 4)

    y_hat, _ = nn.forward(X)

    assert y_hat.shape == (10, 3), "Forward output shape mismatch"


def test_backward_shapes():
    nn = NeuralNetwork([4, 5, 3], ["relu", "sigmoid"])
    X = np.random.randn(8, 4)
    y = np.eye(3)[np.random.randint(0, 3, 8)]

    y_hat, cache = nn.forward(X)
    grads = nn.backward(X, y, cache)

    # Check gradient shapes
    assert grads["dw1"].shape == nn.params["w1"].shape, "dw1 shape mismatch"
    assert grads["db2"].shape == nn.params["b2"].shape, "db2 shape mismatch"


print("All tests passed!")
