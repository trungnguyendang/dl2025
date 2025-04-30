import matplotlib.pyplot as plt

def read_csv(filepath):
    X = []
    Y = []
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:  # Skip header
                values = line.strip().split(',')
                if len(values) == 2:
                    try:
                        x_val = float(values[0])
                        y_val = float(values[1])
                        X.append(x_val)
                        Y.append(y_val)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return [], []
    return X, Y

# Mean squared error loss function
def mean_squared_error(y_true, y_predicted):
    cost = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_predicted)) / len(y_true)
    return cost

# Gradient descent for linear regression
def gradient_descent(x, y, iterations=1000, learning_rate=0.0001, stopping_threshold=1e-6):
    current_weight = 0.1
    current_bias = 0.01
    n = float(len(x))
    costs = []
    weights = []
    biases = []
    previous_cost = None

    for i in range(iterations):
        # Compute predictions
        y_predicted = [current_weight * xi + current_bias for xi in x]
        current_cost = mean_squared_error(y, y_predicted)

        # Stopping condition
        if previous_cost is not None and abs(previous_cost - current_cost) <= stopping_threshold:
            print(f"Stopping at iteration {i}: Cost change below threshold")
            break

        previous_cost = current_cost
        costs.append(current_cost)
        weights.append(current_weight)
        biases.append(current_bias)

        # Compute gradients
        errors = [yi - ypi for yi, ypi in zip(y, y_predicted)]
        weight_derivative = -(2 / n) * sum(xi * ei for xi, ei in zip(x, errors))
        bias_derivative = -(2 / n) * sum(errors)

        # Check for numerical instability
        if abs(weight_derivative) > 1e10 or abs(bias_derivative) > 1e10:
            print(f"Stopping at iteration {i}: Gradients too large (weight: {weight_derivative}, bias: {bias_derivative})")
            break

        # Update parameters
        current_weight -= learning_rate * weight_derivative
        current_bias -= learning_rate * bias_derivative

        # Check for NaN or infinite values
        if not (isinstance(current_weight, float) and isinstance(current_bias, float) and
                abs(current_weight) < float('inf') and abs(current_bias) < float('inf')):
            print(f"Stopping at iteration {i}: Parameters became invalid (weight: {current_weight}, bias: {current_bias})")
            break

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot data and regression line
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, [current_weight * xi + current_bias for xi in x], 'r-',
             label=f'y = {current_weight:.2f}x + {current_bias:.2f}')
    plt.title('Linear Regression with Gradient Descent (Small Dataset)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()

    # Plot cost vs weights
    plt.subplot(1, 2, 2)
    plt.plot(weights, costs, 'g-')
    plt.scatter(weights, costs, marker='o', color='red', label='Cost')
    plt.title('Cost vs Weights')
    plt.xlabel('Weight')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('linear_regression_gradient_descent_small_dataset.png')
    plt.show()

    return current_weight, current_bias, weights, biases, costs

# Parameters
path = './lr.csv'
iterations = 1000
learning_rate = 0.0001
stopping_threshold = 1e-6

# Read data
x, y = read_csv(path)
if not x or not y or len(x) != 4 or len(y) != 4:
    print("Error: Expected exactly 4 data points in CSV.")
    exit(1)

# Run gradient descent
weight, bias, weights, biases, costs = gradient_descent(
    x, y, iterations, learning_rate, stopping_threshold
)

print(f"Final Weight: {weight:.4f}, Final Bias: {bias:.4f}")