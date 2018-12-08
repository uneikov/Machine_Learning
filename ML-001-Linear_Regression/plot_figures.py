import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
#import matplotlib.cm as cm
#from matplotlib import cm

import numpy as np
from compute_cost import compute_cost


def plot_2d(x, y, t):
    fig2d = plt.figure()
    ax = fig2d.add_subplot(1, 1, 1)
    ax.set_title('Linear regression example')
    ax.set_xlabel('Population in 10.000s')
    ax.set_ylabel('Profit in \$10.000s')
    ax.set_xlim([4., 24.])
    ax.set_ylim([-5., 25.])
    ax.plot(x, y, 'rx', label='training data')
    ax.plot(x, np.dot(x, t), 'b-', label='linear regression')
    ax.legend()
    plt.show()
    return


def plot_figures(x, y, theta):
    # Make data.
    theta0_values = np.linspace(-10, 10, 100)
    theta1_values = np.linspace(-1, 4, 100)
    j_values = np.zeros((len(theta0_values), len(theta1_values)))

    # Fill out J_values
    for i in range(1, len(theta0_values)):
        for j in range(1, len(theta1_values)):
            t = np.transpose(np.matrix([theta0_values[i], theta1_values[j]]))
            j_values[i, j] = compute_cost(x, y, t)

    x_plot, y_plot = np.meshgrid(theta0_values, theta1_values, indexing='ij')

    plt.figure()
    cs = plt.contour(x_plot, y_plot, j_values, np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx')
    plt.xlabel(r'$\theta0$')
    plt.ylabel(r'$\theta1$')
    plt.clabel(cs, inline=1, fontsize=8)
    plt.title('Contour plot for cost function J()\n')
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(x_plot, y_plot, j_values, cmap='bwr')
    ax.set_xlabel(r'$\theta0$')
    ax.set_ylabel(r'$\theta1$')
    ax.set_zlabel('Cost function')

    # Customize the z axis.
    # ax.set_zlim(0, 700)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%3.0f'))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    return
