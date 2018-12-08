import math
import numpy as np


def sigmoid_function(z):
    size = z.size
    g = np.zeros((size, 1))
    z = z.ravel()
    for i in range(size):
        g[i] = sigmoid(z[i])
    return g


def sigmoid(x):
    return 1/(1 + math.exp(-x))
