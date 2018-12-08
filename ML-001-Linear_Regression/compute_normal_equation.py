import numpy as np

# compute theta by solving normal equation


def compute_normal_equation(x, y):
    xt = np.transpose(x)
    return np.dot(np.linalg.inv(np.dot(xt, x)), np.dot(xt, y))
