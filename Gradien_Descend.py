import matplotlib.pyplot as plt


def function(x):
    return x ** 2 + 5 * x + 6


def derivative(x):
    return 2 * x + 5


def gradient_descent(start_x, learning_rate, iterations):
    x = start_x
    x_history = [x]

    for _ in range(iterations):
        gradient = derivative(x)
        x = x - learning_rate * gradient
        x_history.append(x)

    return x, x_history


# Parameters
start_x = 10
learning_rate = 0.1
iterations = 50

# Run gradient descent
min_x, x_history = gradient_descent(start_x, learning_rate, iterations)

# Generate points for plotting the function
x_vals = [i / 10 for i in range(-150, 151)]
y_vals = [function(x) for x in x_vals]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, 'b-', label='f(x) = x² + 5x + 6')
plt.plot(x_history, [function(x) for x in x_history], 'ro-', label='Gradient Descent Path')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.savefig('gradient_descent.png')
plt.show()