import numpy as np
from compute_cost import compute_cost

#  gradient descent algorithm for model with multi features


def gradient_descent_multi(x, y, theta, alpha, n):
    j_history = np.zeros((n, 1))
    am = alpha / len(y)
    for i in range(n):
        theta -= am * np.dot(np.transpose(x), np.dot(x, theta) - y)
        j_history[i] = compute_cost(x, y, theta)
    return theta, j_history
