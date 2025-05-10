import math
import random
import matplotlib.pyplot as plt

def read_csv(filepath):
    data = []
    with open(filepath, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            x1, x2, y = map(int, parts)
            data.append(([x1, x2], [y]))
    return data

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def loss(y_hat, y_pred):
    return -(y_hat * math.log(y_pred) + (1 - y_hat) * math.log(1 - y_pred))

class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.input = []
        self.output = 0.0
        self.z = 0.0
        self.grad_w = [0.0] * len(weight)
        self.grad_b = 0.0

    def activate(self, inputs):
        self.input = inputs
        self.z = sum(self.weight[i] * inputs[i] for i in range(len(self.weight))) + self.bias
        self.output = sigmoid(self.z)
        return self.output

class Layer:
    def __init__(self, weight_list, bias_list):
        self.neurons = [Neuron(w, b) for w, b in zip(weight_list, bias_list)]

    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]

class Network:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            weights = [[random.uniform(-1, 1) for _ in range(layer_sizes[i - 1])] for _ in range(layer_sizes[i])]
            biases = [random.uniform(-1, 1) for _ in range(layer_sizes[i])]
            self.layers.append(Layer(weights, biases))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, expected):
        last_layer = self.layers[-1]
        for i, neuron in enumerate(last_layer.neurons):
            y_pred = neuron.output
            y_true = expected[i]
            derivative_loss = (y_pred - y_true) * sigmoid_derivative(neuron.z)
            for j in range(len(neuron.weight)):
                neuron.grad_w[j] = derivative_loss * neuron.input[j]
            neuron.grad_b = derivative_loss

        for l in reversed(range(len(self.layers) - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            for i, neuron in enumerate(layer.neurons):
                downstream = sum(n.weight[i] * n.grad_b for n in next_layer.neurons)
                derivative_loss = downstream * sigmoid_derivative(neuron.z)
                for j in range(len(neuron.weight)):
                    neuron.grad_w[j] = derivative_loss * neuron.input[j]
                neuron.grad_b = derivative_loss

    def update_weights(self, lr):
        for layer in self.layers:
            for neuron in layer.neurons:
                for j in range(len(neuron.weight)):
                    neuron.weight[j] -= lr * neuron.grad_w[j]
                neuron.bias -= lr * neuron.grad_b

def train(network, data, lr, epochs):
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in data:
            output = network.forward(x)
            total_loss += loss(y[0], output[0])
            network.backward(y)
            network.update_weights(lr)
        loss_history.append(total_loss)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
    return loss_history

if __name__ == "__main__":
    xor_data = read_csv("data.csv")
    nn = Network([2, 2, 1])
    losses = train(nn, xor_data, lr=0.1, epochs=10000)

    for x, y in xor_data:
        pred = nn.forward(x)
        print(f"Input: {x}, Expected: {y[0]}, Predicted: {round(pred[0], 4)}")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.grid(True)
    plt.show()