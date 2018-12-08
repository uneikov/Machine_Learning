from logreg_cost_function import h


def predict(x, theta):
    row = x.shape[0]
    prd = h(x @ theta)
    for i in range(row):
        if prd[i] >= 0.5:
            prd[i] = 1
        else:
            prd[i] = 0
    return prd.reshape((row, 1))
