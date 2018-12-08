import numpy as np
from logreg_cost_function import h, ln


def cost_function_reg(t, x, y, _lambda):
    m, n = x.shape
    t = t.reshape((n, 1))  # make column
    # ----------------- compute cost --------------------
    reg_term = np.sum(t[1:m] ** 2) * _lambda / (2 * m)
    hx = h(x @ t)
    hx[hx == 1] = 0.99999 # log(1) = 0 => div error
    cost = - (y.T @ ln(hx) + (1 - y).T @ ln(1 - hx)) / m + reg_term
    # --------------- compute gradient -------------------
    pre_grad = x.T @ (hx - y) / m
    grad = pre_grad + t * _lambda / m
    grad[0] = pre_grad[0]

    return cost, grad
