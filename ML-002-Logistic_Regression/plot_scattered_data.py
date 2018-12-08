import matplotlib.pyplot as plt
import numpy as np
from plotDecisionBoundary import plot_2d_decision_boundary, plot_1d_decision_boundary


def plot_scattered_data_1(x, y, t, with_boundary):
    rows, cols = x.shape
    x0 = np.zeros((1, cols))
    x1 = np.zeros((1, cols))

    for row in range(rows):
        if y[row] == 0:
            x0 = np.vstack((x0, x[row]))
        else:
            x1 = np.vstack((x1, x[row]))
    x0 = np.delete(x0, 0, axis=0)
    x1 = np.delete(x1, 0, axis=0)

    plt.figure()
    plt.title('Figure 1:  Scatter plot of training data\n')
    plt.xlabel('Exam #1 score')
    plt.ylabel('Exam #2 score')
    plt.plot(np.delete(x0, 1, axis=1), np.delete(x0, 0, axis=1), 'ro', label='not admitted')
    plt.plot(np.delete(x1, 1, axis=1), np.delete(x1, 0, axis=1), 'go', label='admitted')
    plt.legend(loc='lower left')
    if with_boundary:
        plot_1d_decision_boundary(x, t)

    return


def plot_scattered_data_2(x, y, t, with_boundary):
    rows, cols = x.shape
    x0 = np.zeros((1, cols))
    x1 = np.zeros((1, cols))

    for row in range(rows):
        if y[row] == 0:
            x0 = np.vstack((x0, x[row]))
        else:
            x1 = np.vstack((x1, x[row]))
    x0 = np.delete(x0, 0, axis=0)
    x1 = np.delete(x1, 0, axis=0)

    fig = plt.figure()
    plt.title('2d plot of positive and negative microchip tests\n' + r'$\lambda=$' + str(1))
    plt.xlabel('Microchip test #1')
    plt.ylabel('Microchip test #2')
    plt.plot(np.delete(x0, 1, axis=1), np.delete(x0, 0, axis=1), 'ro', label='not admitted')
    plt.plot(np.delete(x1, 1, axis=1), np.delete(x1, 0, axis=1), 'go', label='admitted')
    plt.legend(loc='lower left')
    if with_boundary:
        plot_2d_decision_boundary(t)

    return
