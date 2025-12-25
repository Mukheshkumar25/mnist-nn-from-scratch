import numpy as np

class NeuralNetwork:
    """
    Docstring for NeuralNetwork
    I am Creating  a Fully connected Neural Network(Dense)
    layer_sizes: list of layers [784, 128, 64, 10]
    activations: list of activations ['relu', 'relu', 'sigmoid']
    """
    def __init__(self,layer_sizes,activations,learning_rate = 0.001):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activations = activations
        self.params = self.__init__params()
    
    def __init__params(self):
        params = {}
        for i in range(len(self.layer_sizes) - 1):
            params[f"w{i+1}"] = np.random.randn(self.layer_sizes[i],self.layer_sizes[i+1])*np.sqrt(2/self.layer_sizes[i])
            params[f"b{i+1}"] = np.zeros((1, self.layer_sizes[i+1]))

            
        return params
        
    

    def _softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _activate(self,function,z):
        if function == "relu":
            return np.maximum(0,z)
        if function == "sigmoid":
            return 1 / (1 + np.exp(-z))
        if function == "softmax":
            return self._softmax(z)
        raise ValueError("Unsupported activation")
    
    def _activate_derivative(self,z,func):
        if func == "relu":
            return (z > 0).astype(float)
        if func == "sigmoid":
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        raise ValueError("Unsupported activation")
    
    

        
    def forward(self,x):
        cache = {"a0":x}
        A = x

        for i,act in enumerate(self.activations):
            z = A @ self.params[f"w{i+1}"] + self.params[f"b{i+1}"]
            A = self._activate(act,z)
            cache[f"z{i+1}"] = z
            cache[f"a{i+1}"] = A

        return A,cache
    
    def backward(self, x, y, cache):
        grads = {}
        m = x.shape[0]
        L = len(self.activations)

        # ---------- OUTPUT LAYER (Softmax + CE) ----------
        # dZ = y_hat - y  (NO activation derivative here)
        dz = cache[f"a{L}"] - y

        grads[f"dw{L}"] = cache[f"a{L-1}"].T @ dz / m
        grads[f"db{L}"] = np.sum(dz, axis=0, keepdims=True) / m

        da = dz @ self.params[f"w{L}"].T

        # ---------- HIDDEN LAYERS ----------
        for i in reversed(range(L - 1)):
            dz = da * self._activate_derivative(
                cache[f"z{i+1}"], self.activations[i]
            )

            grads[f"dw{i+1}"] = cache[f"a{i}"].T @ dz / m
            grads[f"db{i+1}"] = np.sum(dz, axis=0, keepdims=True) / m

            da = dz @ self.params[f"w{i+1}"].T

        return grads

    
    def update(self,grads):
        for i in range(len(self.activations)):
            self.params[f"w{i+1}"] = (
                self.params[f"w{i+1}"] - self.learning_rate * grads[f"dw{i+1}"]
            )
            self.params[f"b{i+1}"] = (
                self.params[f"b{i+1}"] - self.learning_rate * grads[f"db{i+1}"]
            )

