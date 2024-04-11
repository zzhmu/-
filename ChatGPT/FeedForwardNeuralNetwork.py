import numpy as np


class FeedForwardNeuralNetwork:
    def __init__(self, layer_sizes, activation="sigmoid"):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i - 1])
            for i in range(1, self.num_layers)
        ]
        self.biases = [
            np.random.randn(layer_sizes[i], 1) for i in range(1, self.num_layers)
        ]
        self.activation_function = self._get_activation_function(activation)

    def _get_activation_function(self, name):
        if name == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == "relu":
            return lambda x: np.maximum(0, x)
        elif name == "tanh":
            return lambda x: np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def _tanh_derivative(self, x):
        return 1 - np.square(x)

    def _calculate_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def _forward_propagation(self, X):
        activation = X
        activations = [X]
        weighted_inputs = []

        for w, b in zip(self.weights, self.biases):
            weighted_input = np.dot(w, activation) + b
            weighted_inputs.append(weighted_input)
            activation = self.activation_function(weighted_input)
            activations.append(activation)

        return activations, weighted_inputs

    def _backpropagation(self, X, y, activations, weighted_inputs):
        num_examples = X.shape[1]
        deltas = [None] * (self.num_layers - 1)

        # Compute error in output layer
        output_error = activations[-1] - y
        deltas[-1] = output_error * self.activation_function(
            weighted_inputs[-1], derivative=True
        )

        # Backpropagate the error
        for l in range(self.num_layers - 2, 0, -1):
            deltas[l - 1] = np.dot(
                self.weights[l].T, deltas[l]
            ) * self.activation_function(weighted_inputs[l - 1], derivative=True)

        # Compute gradients
        gradients_w = [
            np.dot(d, a.T) / num_examples for d, a in zip(deltas, activations[:-1])
        ]
        gradients_b = [np.mean(d, axis=1, keepdims=True) for d in deltas]

        return gradients_w, gradients_b

    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            activations, weighted_inputs = self._forward_propagation(X)
            loss = self._calculate_loss(y, activations[-1])
            gradients_w, gradients_b = self._backpropagation(
                X, y, activations, weighted_inputs
            )

            # Update weights and biases
            self.weights = [
                w - learning_rate * dw for w, dw in zip(self.weights, gradients_w)
            ]
            self.biases = [
                b - learning_rate * db for b, db in zip(self.biases, gradients_b)
            ]

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss}")

    def predict(self, X):
        activations, _ = self._forward_propagation(X)
        return activations[-1]
