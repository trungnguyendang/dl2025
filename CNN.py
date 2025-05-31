import math
import random
import matplotlib.pyplot as plt


def read_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        f.read(16)
        images = []
        while True:
            img = f.read(28 * 28)
            if not img:
                break
            img = [b / 255.0 for b in img]  # Normalize to [0, 1]
            images.append([[img[i * 28 + j] for j in range(28)] for i in range(28)])
        return images


def read_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        f.read(8)
        labels = list(f.read())
        return labels


def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def softmax(z):
    max_z = max(z)
    exp_z = [math.exp(zi - max_z) for zi in z]
    sum_exp_z = sum(exp_z)
    return [exp_zi / sum_exp_z for exp_zi in exp_z]


def cross_entropy_loss(y_pred, y_true):
    return -sum(t * math.log(p + 1e-10) for t, p in zip(y_true, y_pred))

def conv2d(image, kernel, stride=1, padding=0):
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    padded = [[0] * (w + 2 * padding) for _ in range(h + 2 * padding)]
    for i in range(h):
        for j in range(w):
            padded[i + padding][j + padding] = image[i][j]
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    output = [[0] * out_w for _ in range(out_h)]
    for i in range(0, h + 2 * padding - kh + 1, stride):
        for j in range(0, w + 2 * padding - kw + 1, stride):
            sum_val = 0
            for ki in range(kh):
                for kj in range(kw):
                    sum_val += padded[i + ki][j + kj] * kernel[ki][kj]
            output[i // stride][j // stride] = sum_val
    return output

def max_pooling(image, size=2, stride=2):
    h, w = len(image), len(image[0])
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    output = [[0] * out_w for _ in range(out_h)]
    for i in range(0, h - size + 1, stride):
        for j in range(0, w - size + 1, stride):
            max_val = float('-inf')
            for ki in range(size):
                for kj in range(size):
                    max_val = max(max_val, image[i + ki][j + kj])
            output[i // stride][j // stride] = max_val
    return output


def flatten(matrix):
    return [val for row in matrix for val in row]

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None, activation_derivative=None):
        self.weights = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.uniform(-1, 1) for _ in range(output_size)]
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.outputs = [0.0] * output_size
        self.z_values = [0.0] * output_size
        self.inputs = []
        self.grad_w = [[0.0 for _ in range(input_size)] for _ in range(output_size)]
        self.grad_b = [0.0 for _ in range(output_size)]

    def forward(self, inputs):
        self.inputs = inputs
        self.z_values = []
        for i in range(len(self.weights)):
            z = sum(w * x for w, x in zip(self.weights[i], inputs)) + self.biases[i]
            self.z_values.append(z)
        if self.activation:
            self.outputs = [self.activation(z) for z in self.z_values]
        else:
            self.outputs = self.z_values
        return self.outputs


class Network:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(DenseLayer(layer_sizes[i - 1], layer_sizes[i],
                                          activation=relu,
                                          activation_derivative=relu_derivative))
        self.layers.append(DenseLayer(layer_sizes[-2], layer_sizes[-1],
                                      activation=None,
                                      activation_derivative=None))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, expected):
        last_layer = self.layers[-1]
        y_pred = softmax(last_layer.outputs)
        for i in range(len(last_layer.outputs)):
            delta = y_pred[i] - expected[i]
            last_layer.grad_b[i] = delta
            for j in range(len(last_layer.inputs)):
                last_layer.grad_w[i][j] = delta * last_layer.inputs[j]

        for l in reversed(range(len(self.layers) - 1)):
            current = self.layers[l]
            next_layer = self.layers[l + 1]
            for i in range(len(current.outputs)):
                downstream = sum(
                    next_layer.weights[k][i] * next_layer.grad_b[k]
                    for k in range(len(next_layer.outputs))
                )
                if current.activation_derivative:
                    delta = downstream * current.activation_derivative(current.z_values[i])
                else:
                    delta = downstream
                current.grad_b[i] = delta
                for j in range(len(current.inputs)):
                    current.grad_w[i][j] = delta * current.inputs[j]

    def update_weights(self, lr):
        for layer in self.layers:
            for i in range(len(layer.weights)):
                for j in range(len(layer.weights[i])):
                    layer.weights[i][j] -= lr * layer.grad_w[i][j]
                layer.biases[i] -= lr * layer.grad_b[i]


# Training function
def train_mnist(images, labels, network, lr, epochs, batch_size):
    loss_history = []

    def one_hot_encode(label, size=10):
        encoded = [0] * size
        encoded[label] = 1
        return encoded

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            for image, label in zip(batch_images, batch_labels):
                conv_output = conv2d(image, kernel=[[1, 0, -1], [1, 0, -1], [1, 0, -1]], stride=1, padding=0)
                conv_relu = [[relu(val) for val in row] for row in conv_output]
                pooled = max_pooling(conv_relu, size=2, stride=2)
                flattened = flatten(pooled)
                z = network.forward(flattened)
                output = softmax(z)
                target = one_hot_encode(label, size=10)
                loss = cross_entropy_loss(output, target)
                total_loss += loss
                network.backward(target)
                network.update_weights(lr)
        avg_loss = total_loss / len(images)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")
    return loss_history


# Main function
def main():
    data_dir = "data"
    images = read_mnist_images(f"{data_dir}/train-images.idx3-ubyte")[:3000]
    labels = read_mnist_labels(f"{data_dir}/train-labels.idx1-ubyte")[:3000]

    kernel = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]

    conv_output = conv2d(images[0], kernel, stride=1, padding=0)
    conv_relu = [[relu(val) for val in row] for row in conv_output]
    pooled = max_pooling(conv_relu, size=2, stride=2)
    flattened_size = len(flatten(pooled))

    network = Network([flattened_size, 128, 64, 10])

    # Train the model and get loss history
    loss_history = train_mnist(images, labels, network, lr=0.01, epochs=5, batch_size=32)

    # Plot the loss over time
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Loss Over Time")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()