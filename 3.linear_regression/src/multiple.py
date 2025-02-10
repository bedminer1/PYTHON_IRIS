import numpy as np

def multiple_linear_regression(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features, = X.shape
    X = np.concatenate((np.ones((n_samples, 1)), X), axis=1)

    w = np.zeros(n_features + 1)

    for _ in range(n_iterations):
        y_pred = X @ w

        error = y_pred - y
        dw = (2/n_samples) * (X.T @ error)
        w = w - learning_rate * dw

    return w


X = np.array([[1, 2], [2, 3], [3, 1], [4, 4], [5, 2]])  # Two features
y = np.array([3, 5, 2, 7, 6])  # Outcome

w = multiple_linear_regression(X, y)
print("Coefficients (including intercept):", w)

# Make Predictions
X_new = np.array([[6,3], [2,1]])
X_new = np.concatenate((np.ones((X_new.shape[0], 1)), X_new), axis=1) # Add bias term to X_new
y_pred = X_new @ w
print("Predictions for X_new", y_pred)
