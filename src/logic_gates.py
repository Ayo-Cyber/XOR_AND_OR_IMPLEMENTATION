import numpy as np
from src.neural_network import NeuralNetwork
from src.tensorflow_model import create_and_train_nn, create_and_train_xor_nn


def train_numpy_model(gate):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if gate == 'AND':
        y = np.array([[0], [0], [0], [1]])
        nn = NeuralNetwork([2, 1], alpha=0.5)
        nn.fit(X, y, epochs=1000, display_update=500)
        return nn
    elif gate == 'OR':
        y = np.array([[0], [1], [1], [1]])
        nn = NeuralNetwork([2, 1], alpha=0.5)
        nn.fit(X, y, epochs=1000, display_update=500)
        return nn
    elif gate == 'XOR':
        y = np.array([[0], [1], [1], [0]])
        nn = NeuralNetwork([2, 2, 1], alpha=0.5)
        nn.fit(X, y, epochs=5000, display_update=1000)
        return nn

def train_tensorflow_model(gate):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if gate == 'AND':
        y = np.array([[0], [0], [0], [1]])
        model = create_and_train_nn(2, 1, X, y, epochs=100)
        return model
    elif gate == 'OR':
        y = np.array([[0], [1], [1], [1]])
        model = create_and_train_nn(2, 1, X, y, epochs=100)
        return model
    elif gate == 'XOR':
        y = np.array([[0], [1], [1], [0]])
        model = create_and_train_xor_nn(X, y, epochs=100)
        return model
