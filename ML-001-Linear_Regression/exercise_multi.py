import numpy as np
from numpy import genfromtxt
from feature_normalize import feature_normalize
from gradient_descent import gradient_descent
from plot_convergence import plot_convergence
from compute_normal_equation import compute_normal_equation

file_path = 'C:/Users/uran-desktop/Documents/Octave/mlclass-ex1/mlclass-ex1/ex1data2.txt'

data = genfromtxt(file_path, delimiter=',')
shape = data.shape
rows = shape[0]
cols = shape[1]

X = np.delete(data, cols-1, axis=1)
Y = np.delete(data, np.arange(cols - 1), axis=1)

# ----------------- make design matrix for gradient descent ----------------
[X_norm, mean, sigma] = feature_normalize(X)
X_norm_1 = np.append(np.ones((rows, 1)), X_norm, axis=1)

# ----------------- make design matrix for normal equation -----------------
X_1 = np.append(np.ones((rows, 1)), X, axis=1)

# ------ find theta by gradient descent for different learning step --------
num_iters = 100
alpha = 0.01
[theta1, jh1] = gradient_descent(X_norm_1, Y, alpha, num_iters)
alpha = 0.03
[theta2, jh2] = gradient_descent(X_norm_1, Y, alpha, num_iters)
alpha = 0.1
[theta3, jh3] = gradient_descent(X_norm_1, Y, alpha, num_iters)
plot_convergence(jh1, jh2, jh3)

print('Theta values derived by gradient descent: \n', str(theta3))


# ------------------- find theta by solving normal equation ----------------
theta_norm_eq = compute_normal_equation(X_1, Y)
print('Theta values derived by solving normal equation: \n', str(theta_norm_eq))

# ------------------- testing for real data set -----------------------------
test_house_area = 1650
test_house_num_bedroom = 3
test_house = np.array([test_house_area, test_house_num_bedroom])
# ------------------- house price by gradient descent -----------------------
test_example = (test_house - mean)/sigma
test_example = np.append(1, test_example)
price_GD = np.dot(test_example, theta3)
print('Price of the test house (GD method) =' + str(price_GD))
# ------------------- house price by normal equation -----------------------
price_NE = np.dot(np.append(1, test_house), theta_norm_eq)
print('Price of the test house (NE method) =' + str(price_NE))
