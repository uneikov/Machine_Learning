import numpy as np
from map_feature import map_feature
from numpy import genfromtxt
from cost_function_with_reg import cost_function_reg
from plot_scattered_data import plot_scattered_data_2
from prediction import predict
import scipy.optimize as optimum
import matplotlib.pyplot as plt


def pause(): input("Press the <ENTER> key to continue...")


# ------------------------------- read data -------------------------------
file_path = 'C:/Users/uran-desktop/Documents/Octave/mlclass-ex2/mlclass-ex2/ex2data2.txt'
data = genfromtxt(file_path, delimiter=',')
# ------------------------------- plot data -------------------------------
d_rows, d_cols = data.shape
X = np.delete(data, d_cols-1, axis=1)
Y = np.delete(data, np.arange(d_cols - 1), axis=1)
rows, cols = X.shape
initial_theta = np.zeros((cols + 1, 1))
plot_scattered_data_2(X, Y, initial_theta, False)
plt.show()

mapped_array = map_feature(X[:, 0], X[:, 1])
initial_theta = np.zeros((mapped_array.shape[1], 1))
# initial_theta = np.array([[1.273005],[0.624876],[1.177376],[-2.020142],[-0.912616],
# [-1.429907],[0.125668],[-0.368551],[-0.360033],[-0.171068],[-1.460894],[-0.052499],
# [-0.618889],[-0.273745],[-1.192301],[-0.240993],[-0.207934],[-0.047224],[-0.278327],
# [-0.296602],[-0.453957],[-1.045511],[0.026463],[-0.294330],[0.014381],[-0.328703],
# [-0.143796],[-0.924883]])
initial_lambda = 1

cost, grad = cost_function_reg(initial_theta, mapped_array, Y, initial_lambda)
print('Cost value for zero theta is: {0:.4f}'.format(float(cost)))

# ------------------------ find optimum theta ----------------------------
result = optimum.minimize(
    fun=cost_function_reg,
    x0=initial_theta,
    args=(mapped_array, Y, initial_lambda),
    method='TNC',
    jac=True)

if result.success:
    print('Success! ;-)')
else:
    print('Not successful :-(')

theta_opt = result.x
print("Result of optimization: {0:3s}".format(result.message))
print("Iterations made: {0:3d}".format(result.nit))
print("Cost function value: %.6f" % result.fun)
print("Theta optimum values: {}".format(theta_opt))

# ----------------- plot data with decision boundary ----------------
plot_scattered_data_2(X, Y, theta_opt, True)
plt.show()

p = predict(mapped_array, theta_opt)
accuracy = float(np.mean(p == Y)*100)
print("Train Accuracy: %3.2f" % accuracy)
