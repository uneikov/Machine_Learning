import numpy as np
import statistics as stat


def feature_normalize(x):
    rows = x.shape[0]
    mean = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - np.reshape(list(mean)*rows, (rows, 2)))/sigma

    # for i in range(rows):
    #     for j in range(cols):
    #         x_norm[i, j] = (x[i, j]- mean[j]) / sigma[j]

    return x_norm, mean, sigma
