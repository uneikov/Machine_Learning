import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(cost1, cost2, cost3):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Convergence of gradient descent versus learning rate\n')
    ax.set_xlabel(r'Number of steps')
    ax.set_ylabel(r'Cost function value')
    ax.plot(np.arange(0, 50), cost1[0:50], 'r-', label=r'$\alpha$=0.01')
    ax.plot(np.arange(0, 50), cost2[0:50], 'g-', label=r'$\alpha$=0.03')
    ax.plot(np.arange(0, 50), cost3[0:50], 'b-', label=r'$\alpha$=0.1')
    ax.legend()
    plt.show()
