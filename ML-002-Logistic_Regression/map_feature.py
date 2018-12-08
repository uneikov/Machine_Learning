import numpy as np


def map_feature(x1, x2):
    # mapping dimension
    degree = 6
    # if x1 and x2 are int - turn it to 1 element array
    if len(np.shape(x1)) == 0:
        x1 = np.array((x1,))
    if len(np.shape(x2)) == 0:
        x2 = np.array((x2,))
    rows = len(x1)
    x1 = x1.reshape((rows, 1))
    x2 = x2.reshape((rows, 1))
    out = np.ones((rows, 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            xpow = x1**(i-j)*(x2**j)
            out = np.hstack((out, xpow))
    return out
