import numpy as np
from numpy import genfromtxt
from gradient_descent import gradient_descent
from plot_figures import plot_figures
from plot_figures import plot_2d

file_path = 'C:/Users/uran-desktop/Documents/Octave/mlclass-ex1/mlclass-ex1/ex1data1.txt'

data = genfromtxt(file_path, delimiter=',')
X = np.delete(data, 1, axis=1)
Y = np.delete(data, 0, axis=1)
m = len(Y)

# ------------------ make design function ----------------------------
X = np.append(np.ones((m, 1)), X, axis=1)
num_iters = 1500
alpha = 0.01

# ------------- derive optimal theta by gradient descent -------------
[theta, jh] = gradient_descent(X, Y, alpha, num_iters)

# ------------------------- plot figures -----------------------------
plot_2d(X, Y, theta)
plot_figures(X, Y, theta)
