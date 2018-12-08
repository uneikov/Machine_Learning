import numpy as np
import math as math


def cost_function(t, x, y):
    m, n = x.shape
    t = t.reshape((n, 1))
    # ----------- compute sigmoid function ---------------
    hx = h(x @ t)
    # ----------------- compute cost --------------------
    cost = - (y.T@ln(hx) + (1 - y).T@ln(1 - hx))/m
    # --------------- compute gradient -------------------
    gradient = x.T@(hx - y)/m

    return cost, gradient

def h(z):
    vexp = np.vectorize(math.exp)
    return 1/(1+vexp(-z))


def ln(x):
    vlog = np.vectorize(math.log)
    return vlog(x)

