import numpy as np
import scipy.optimize as optimum
from numpy import genfromtxt
from logreg_cost_function import cost_function, h
from prediction import predict
from plot_scattered_data import plot_scattered_data_1
from plotDecisionBoundary import plot_1d_decision_boundary
import matplotlib.pyplot as plt

file_path = 'C:/Users/uran-desktop/Documents/Octave/mlclass-ex2/mlclass-ex2/ex2data1.txt'

data = genfromtxt(fname=file_path, delimiter=',')

d_rows, d_cols = data.shape
#X = data[:, np.arange(0, d_cols-1)].reshape((d_rows, d_cols-1))
#Y = data[:, d_cols-1].reshape((d_rows, 1))
X = np.delete(data, d_cols-1, axis=1)
Y = np.delete(data, np.arange(d_cols - 1), axis=1)

# --------- plot positive and negative examples ---------
rows, cols = X.shape
initial_theta = np.zeros((cols+1, 1))
plot_scattered_data_1(X, Y, initial_theta, False)
plt.show()

# -----make design matrix and get cost & gradient--------
X1 = np.append(np.ones((rows, 1)), X, axis=1)
cost, grad = cost_function(initial_theta, X1, Y)

m, n = X1.shape
theta_opt = np.zeros((n, 1))
initial_theta = np.zeros((n, 1))

result = optimum.minimize(
    fun=cost_function,
    x0=initial_theta,
    args=(X1, Y),
    method='TNC',
    jac=True)

if result.success:
    theta_opt = result.x
    print('Success!\n')
    print("Result of optimization: {0:3s}".format(result.message))
    print("Iterations made: {0:3d}".format(result.nit))
    print("Cost function value: {}".format(result.fun))
    print("Theta optimum values: {}".format(theta_opt))
else:
    print('Not successful')

XTest = np.array([1, 45, 85])
prob = h(XTest@theta_opt)
print("Probability value for test data: {0:8.4f}".format(prob))

predicted = predict(X1, theta_opt)
accuracy = float(np.mean(predicted == Y)*100)
print("Train Accuracy: %3.2f" % accuracy)

plot_scattered_data_1(X, Y, theta_opt, True)
plt.show()
