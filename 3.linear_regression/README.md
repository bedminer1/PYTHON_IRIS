# Linear Regression Implementation from Scratch

This project implements linear regression (both simple and multiple) from scratch using Python and NumPy, without relying on machine learning libraries like scikit-learn.  It serves as a learning exercise to understand the underlying mechanics of linear regression and gradient descent.

## Challenge Aim

The core challenge was to build linear regression models from the ground up, implementing the key algorithms and mathematical concepts involved. This includes:

*   Implementing the cost function (Mean Squared Error).
*   Calculating the gradients of the cost function with respect to the model parameters (slope and intercept for simple linear regression, coefficients for multiple linear regression).
*   Implementing the gradient descent optimization algorithm to find the optimal model parameters that minimize the cost function.
*   Handling both simple linear regression (one predictor variable) and multiple linear regression (multiple predictor variables).

## Implementation

The project consists of two Python files:

*   `simple.py`: Implements simple linear regression.  It takes a 1D NumPy array `X` (predictor variable) and a 1D NumPy array `y` (outcome variable) as input.  It returns the learned slope (`m`) and intercept (`b`) of the linear regression line.

*   `multiple.py`: Implements multiple linear regression. It takes a 2D NumPy array `X` (predictor variables) and a 1D NumPy array `y` (outcome variable) as input. It returns the learned coefficients (`w`), including the intercept.  It uses vectorized operations with NumPy for efficiency.

Both implementations use the following approach:

1.  **Initialization:** Initialize the model parameters (slope and intercept or coefficients) to zero.
2.  **Iteration:** Repeat the following steps for a fixed number of iterations or until convergence:
    *   **Prediction:** Calculate the predicted outcome values using the current model parameters.
    *   **Gradient Calculation:** Calculate the partial derivatives of the cost function (Mean Squared Error) with respect to each model parameter.  These derivatives indicate the direction of the steepest ascent of the cost function.
    *   **Parameter Update:** Update the model parameters by moving in the *opposite* direction of the gradient (steepest descent). The learning rate controls the step size.
3.  **Return:** Return the learned model parameters.

## Learning Points

This project provided several valuable learning experiences:

*   **Understanding Linear Regression:** Gained a deep understanding of how linear regression works, including the mathematical foundations and the role of the cost function and gradient descent.
*   **Gradient Descent:** Learned how to implement and apply the gradient descent optimization algorithm, a fundamental algorithm in machine learning.
*   **Vectorization:**  Practiced vectorizing code using NumPy, which is essential for efficient numerical computation in machine learning, especially when dealing with multiple features.
*   **NumPy:**  Improved proficiency in using NumPy arrays and functions for numerical computation.
*   **Implementation from Scratch:** The process of building linear regression from scratch solidified my understanding of the algorithm and its underlying principles.
*   **Debugging:**  Encountered and solved common issues related to array shapes, data types, and numerical stability, improving debugging skills.

## How to Run

1.  Make sure you have Python and NumPy installed.
2.  Clone the repository (if applicable).
3.  Navigate to the directory containing the Python files.
4.  Run the scripts using `python simple.py` and `python multiple.py`.  The output will show the learned model parameters and predictions.

## Future Improvements

*   **Feature Scaling:** Implement feature scaling (e.g., standardization or normalization) to improve the performance of gradient descent, especially for datasets with features on different scales.
*   **Regularization:** Add regularization techniques (L1 or L2) to prevent overfitting, which is particularly important for multiple linear regression with many features.
*   **Convergence Criteria:** Implement a more robust convergence criterion for gradient descent, rather than just a fixed number of iterations.  This could involve monitoring the cost function and stopping when it stops decreasing significantly.
*   **More Advanced Optimization:** Explore other optimization algorithms, such as stochastic gradient descent (SGD) or Adam, which can often converge faster and more reliably than standard gradient descent.