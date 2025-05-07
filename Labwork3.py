import math

def read_csv(filepath):
    salary = []
    experience = []
    loan = []
    try:
        with open(filepath, 'r') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 3:
                    continue
                try:
                    salary.append(float(parts[0]))
                    experience.append(float(parts[1]))
                    loan.append(int(parts[2]))
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    return salary, experience, loan

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def compute_lost(w0, w1, w2, salary, experience, loan):
    m = len(loan)
    cost = 0.0
    for i in range(m):
        z = w0 + w1 * salary[i] + w2 * experience[i]
        h = sigmoid(z)
        cost += - (loan[i] * math.log(h) + (1 - loan[i]) * math.log(1 - h))
    return cost / m

def compute_focal_loss(w0, w1, w2, salary, experience, loan):
    gamma = 2.0
    alpha = 1.0
    m = len(loan)
    total_loss = 0.0
    for i in range(m):
        z = w0 + w1 * salary[i] + w2 * experience[i]
        p = sigmoid(z)
        p_t = p if loan[i] == 1 else 1 - p
        weight = alpha * (1 - p_t) ** gamma
        loss = -weight * math.log(p_t)
        total_loss += loss
    return total_loss / m

def gradient_descent(salary, experience, loan, lr, iterations):
    m = len(loan)
    w0, w1, w2 = 0.0, 0.0, 0.0

    for epoch in range(iterations):
        grad0, grad1, grad2 = 0.0, 0.0, 0.0
        gamma = 2.0
        alpha = 1.0
        for i in range(m):
            z = w0 + w1 * salary[i] + w2 * experience[i]
            p = sigmoid(z)  # Predicted probability
            p_t = p if loan[i] == 1 else 1 - p  # True class probability

            # Focal loss weight
            weight = alpha * (1 - p_t) ** gamma
            error = weight * (-loan[i] + p) / (p_t)  # Gradient term

            grad0 += error
            grad1 += error * salary[i]
            grad2 += error * experience[i]

        grad0 /= m
        grad1 /= m
        grad2 /= m

        w0 -= lr * grad0
        w1 -= lr * grad1
        w2 -= lr * grad2

        loss = compute_focal_loss(w0, w1, w2, salary, experience, loan)
        print(f"Iter {epoch}/{iterations}: w0={w0:.4f}, w1={w1:.4f}, w2={w2:.4f}, loss={loss:.4f}")

    return w0, w1, w2

if __name__ == "__main__":
    filepath = './loan2.csv'
    learning_rate = 0.01
    iterations = 100


    salary, experience, loan = read_csv(filepath)

    w0, w1, w2 = gradient_descent(
        salary, experience, loan,
        learning_rate, iterations
    )
    print(f"\nFinal learned weights:\n  w0 = {w0:.4f}\n  w1 = {w1:.4f}\n  w2 = {w2:.4f}")