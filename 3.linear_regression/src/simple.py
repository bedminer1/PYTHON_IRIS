import numpy as np

def linear_regression(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples = len(X)
    m = 0
    b = 0

    for _ in range(n_iterations):
        y_pred = m * X + b
        # Calculate partial derivatives
        dm = (2/n_samples) * np.sum(X * (y_pred - y))
        db = (2/n_samples) * np.sum(y_pred - y)
        # Update parameters
        m = m - learning_rate * dm
        b = b - learning_rate * db
    return m, b

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m, b = linear_regression(X, y)
print("Slope (m):", m)
print("Intercept (b): ", b)

X_new = np.array([6, 7])
y_pred = m * X_new + b
print("Predictions for X_new", y_pred)