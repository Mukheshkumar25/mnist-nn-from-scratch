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
        
