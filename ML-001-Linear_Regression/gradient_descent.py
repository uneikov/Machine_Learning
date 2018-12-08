import numpy as np
from compute_cost import compute_cost

#  gradient descent algorithm for model with n (n = 1,2,...) features


def gradient_descent(x, y, learning_step, number_of_iterations):
    #  initial theta vector set to zero
    theta = np.zeros((x.shape[1], 1))
    # initial cost function history vector set to zero
    j_history = np.zeros((number_of_iterations, 1))
    am = learning_step/len(y)
    for i in range(number_of_iterations):
        theta -= am*np.dot(np.transpose(x), (np.dot(x, theta)-y))
        j_history[i] = compute_cost(x, y, theta)
    return theta, j_history
