import numpy as np
from keras.datasets import fashion_mnist
import keras
import matplotlib.pyplot as plt

fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class NeuralNetwork:
    def __init__(self, layers):
        self.activation = lambda x: np.maximum(x, 0)  # ReLU activation
        self.activation_prime = lambda x: (x > 0).astype(float)  # Derivative of ReLU
        self.layers = layers
        self.weights = []
        self.biases = []
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def feedforward(self, x):
        a = x
        z_store = []  # Store linear combinations
        a_store = [a]  # Store activations

        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation(z)
            z_store.append(z)
            a_store.append(a)

        return a, z_store, a_store

    def backpropagation(self, y_true, y_pred, z_store, a_store):
        delta = y_pred - y_true
        deltas = [delta]

        # Loop in reverse order starting from the second to last layer
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.activation_prime(z_store[i - 1])
            deltas.append(delta)

        # Reverse deltas to match weights' order
        deltas.reverse()

        # Gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= np.dot(a_store[i].T, deltas[i]) * learning_rate
            self.biases[i] -= np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def train(self, x, y_true, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred, z_store, a_store = self.feedforward(x)
            self.backpropagation(y_true, y_pred, z_store, a_store)
            loss = np.mean((y_pred - y_true) ** 2)
            print(f'Epoch {epoch + 1}, Loss: {loss}')

    def predict(self, x):
        a, _, _ = self.feedforward(x)
        return a


# Define network architecture
layers = [784, 128, 64, 10]  # Example: 784 input neurons, two hidden layers with 128 and 64 neurons, and 10 output neurons
learning_rate = 0.001
epochs = 10

# Initialize the network
nn = NeuralNetwork(layers)

# Prepare the data
train_images_reshaped = train_images.reshape(train_images.shape[0], -1) / 255.0  # Flatten and normalize the images
train_labels_one_hot = np.eye(10)[train_labels]  # Convert labels to one-hot encoding

# Train the network
nn.train(train_images_reshaped, train_labels_one_hot, epochs, learning_rate)

# Predict
predictions = nn.predict(test_images.reshape(test_images.shape[0], -1) / 255.0)
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate
accuracy = np.mean(predicted_labels == test_labels)
print(f'Accuracy: {accuracy}')
