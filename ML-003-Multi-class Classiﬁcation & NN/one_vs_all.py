import numpy as np
import scipy.optimize as optimum
from cost_function_with_reg import cost_function_reg


def one_vs_all(x, y, num_labels, _lambda):
    m, n = x.shape
    all_theta = np.zeros(shape=(num_labels, n))
    success = np.zeros(shape=(num_labels, 1))

    for num in range(num_labels):
        initial_theta = np.zeros(shape=(n, 1))
        label = (y == num).astype(int)
        result = optimum.minimize(
            fun=cost_function_reg,
            x0=initial_theta,
            args=(x, label, _lambda),
            method='TNC',
            jac=True)
        all_theta[num, :] = result.x
        success[num, :] = result.success

    return success.all(), all_theta
