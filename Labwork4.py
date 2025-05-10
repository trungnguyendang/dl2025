import math
import random

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def read_architecture(file_path):
    with open(file_path, 'r') as f:
        lines = [int(line.strip()) for line in f if line.strip()]
    if len(lines) < 2:
        raise ValueError("Architecture must have at least input and output layers")
    return lines

class Layer:
    def __init__(self, size, prev_size):
        self.weights = [[random.uniform(0, 1) for _ in range(size)] for _ in range(prev_size)] if prev_size > 0 else []
        self.biases = [random.uniform(0, 1) for _ in range(size)] if prev_size > 0 else []
        self.activations = [0.0] * size

    def forward(self, inputs):
        if not self.weights:
            self.activations = inputs[:]
        else:
            self.activations = [
                sigmoid(sum(inputs[j] * self.weights[j][i] for j in range(len(inputs))) + self.biases[i])
                for i in range(len(self.biases))
            ]
        return self.activations

class NeuralNetwork:
    def __init__(self, architecture_file):
        sizes = read_architecture(architecture_file)
        self.layers = [Layer(sizes[0], 0)] + [Layer(sizes[i], sizes[i - 1]) for i in range(1, len(sizes))]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

if __name__ == "__main__":
    nn = NeuralNetwork('Labwork4_file.txt')
    x = [0.5, 0.3, 0.1, 0.2]
    print(f"Input: {x}\nOutput: {nn.forward(x)}")
