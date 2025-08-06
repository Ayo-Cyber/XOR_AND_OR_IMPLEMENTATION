import numpy as np

class NeuralNetwork:
    """
    Attributes:
        layers (list of int): Number of neurons per layer, e.g. [2, 2, 1].
        alpha (float): Learning rate for gradient descent.
        W (list of np.ndarray): Weight matrices for each layer transition.
        b (list of np.ndarray): Bias vectors for each layer transition.
    """
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.W = []  # weights: list of matrices of shape (n_in, n_out)
        self.b = []  # biases: list of row-vectors of shape (1, n_out)

        for i in range(len(layers) - 1):
            n_in, n_out = layers[i], layers[i+1]
            # Xavier initialization: keeps variance of activations stable
            W = np.random.randn(n_in, n_out) / np.sqrt(n_in)
            b = np.zeros((1, n_out))
            self.W.append(W)
            self.b.append(b)

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(map(str, self.layers)))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deriv(self, sigmoid_x):
        return sigmoid_x * (1.0 - sigmoid_x)

    def fit(self, X, y, epochs=1000, display_update=100):
        for epoch in range(1, epochs + 1):
            for x_i, y_i in zip(X, y):
                self._update_sample(x_i, y_i)
            if epoch == 1 or epoch % display_update == 0:
                loss = self.calculate_loss(X, y)
                print(f"[INFO] epoch={epoch}, loss={loss:.7f}")

    def _update_sample(self, x, target):
        # --- FORWARD PASS ---
        activations = [np.atleast_2d(x)]
        for W, b in zip(self.W, self.b):
            z = activations[-1] @ W + b
            a = self.sigmoid(z)
            activations.append(a)

        # --- BACKPROPAGATION ---
        # Output-layer error and delta
        output = activations[-1]
        error = output - target
        deltas = [error * self.sigmoid_deriv(output)]

        # Hidden-layer deltas
        for i in range(len(self.W) - 1, 0, -1):
            delta_next = deltas[-1]
            z_hidden = activations[i]
            delta = (delta_next @ self.W[i].T) * self.sigmoid_deriv(z_hidden)
            deltas.append(delta)
        deltas.reverse()

        # --- WEIGHT & BIAS UPDATES ---
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a_prev = activations[i]
            delta = deltas[i]
            grad_W = a_prev.T @ delta
            grad_b = delta
            self.W[i] = W - self.alpha * grad_W
            self.b[i] = b - self.alpha * grad_b

    def predict(self, X):
        a = np.atleast_2d(X)
        for W, b in zip(self.W, self.b):
            a = self.sigmoid(a @ W + b)
        return a

    def calculate_loss(self, X, targets):
        predictions = self.predict(X)
        errors = predictions - targets
        return 0.5 * np.sum(errors**2)
