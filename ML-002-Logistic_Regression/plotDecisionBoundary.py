import numpy as np
import matplotlib.pyplot as plt
from map_feature import map_feature


def plot_1d_decision_boundary(x, t):

    # see Логистическая регрессия.docx
    x_plot = np.array([np.amin(x) - 2.0, np.amax(x) + 2.0])
    y_plot = -(t[0] + t[1]*x_plot)/t[2]
    plt.plot(x_plot, y_plot, 'b-')

    return


def plot_2d_decision_boundary(t):

    # Here is the grid range
    points = 50
    x1_points = np.linspace(-1, 1.25, points)
    x2_points = np.linspace(-1, 1.25, points)
    z = np.zeros((points, points))
    x1_plot, x2_plot = np.meshgrid(x1_points, x2_points, indexing='ij')

    # Evaluate  z = theta@x over the grid
    for i in range(1,points):
        for j in range(1,points):
            # z[i, j] = map_feature(np.array((x1_points[i],)), np.array((x2_points[j],))) @ t
            z[i, j] = map_feature(x1_points[i], x2_points[j]) @ t

    # z = theta@x = 0 corresponds to sigmoid function equal 0.5 (our threshold)
    plt.contour(x1_plot, x2_plot, z, [0.0, np.amax(z)])

    return
