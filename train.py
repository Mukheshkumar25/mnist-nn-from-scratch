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
    learning_rate=0.05
)


epochs = 200
losses = []
accuracies = []
for epoch in range(epochs):
    y_hat,cache = nn.forward(X_train)
    loss = -np.mean(np.sum(y_train * np.log(y_hat + 1e-8), axis=1))
    grads = nn.backward(X_train, y_train, cache)
    nn.update(grads)

    preds = np.argmax(y_hat,axis=1)
    labels = np.argmax(y_train,axis=1)
    acc = np.mean(preds == labels)

    losses.append(loss)
    accuracies.append(acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")



y_test_hat, _ = nn.forward(X_test)
test_preds = np.argmax(y_test_hat, axis=1)
test_labels = np.argmax(y_test, axis=1)
test_acc = np.mean(test_preds == test_labels)

print(f"\nTest Accuracy: {test_acc:.4f}")

print("Losses:",losses)
print("Accuracies:",accuracies)
# Plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.title("Training Accuracy")

plt.show()
