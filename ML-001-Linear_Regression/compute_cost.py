import numpy as np

#  cost function computation


# def compute_cost(x, y, t):
#     return sum(np.square(np.dot(x, t)-y), 1)/(2*len(y))

def compute_cost(x, y, t):
    v = np.dot(x, t) - y
    return np.dot(np.transpose(v), v)/(2*len(y))
