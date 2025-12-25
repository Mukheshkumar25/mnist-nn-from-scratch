import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from model import NeuralNetwork

X,y = fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False)
X = X / 255.0
y = y.astype(int)

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

X_train,X_test,y_train,y_test = train_test_split(X,y_encoded,test_size=0.2,random_state=42)

nn = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    activations=["relu", "relu", "softmax"],
    learning_rate=0.1
)


epochs = 30
batch_size = 128
n_samples = X_train.shape[0]

losses = []
accuracies = []
for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(n_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0
    correct = 0

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        y_hat, cache = nn.forward(X_batch)

        loss = -np.mean(
            np.sum(y_batch * np.log(y_hat + 1e-8), axis=1)
        )

        grads = nn.backward(X_batch, y_batch, cache)
        nn.update(grads)

        epoch_loss += loss * len(X_batch)
        preds = np.argmax(y_hat, axis=1)
        labels = np.argmax(y_batch, axis=1)
        correct += np.sum(preds == labels)

    epoch_loss /= n_samples
    epoch_acc = correct / n_samples

    losses.append(epoch_loss)
    accuracies.append(epoch_acc)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}"
    )




y_test_hat, _ = nn.forward(X_test)
test_preds = np.argmax(y_test_hat, axis=1)
test_labels = np.argmax(y_test, axis=1)
test_acc = np.mean(test_preds == test_labels)

print(f"\nTest Accuracy: {test_acc:.4f}")

#print("Losses:",losses)
#print("Accuracies:",accuracies)
# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title("Training Accuracy")

plt.show()
