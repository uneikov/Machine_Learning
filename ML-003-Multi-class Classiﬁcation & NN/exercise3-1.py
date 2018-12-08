"""
In this part of the exercise, you will implement one-vs-all classiﬁcation
by training multiple regularized logistic regression classiﬁers, one for
each of the K classes in our dataset. In the handwritten digits dataset,
K = 10, but your code should work for any value of K.
"""

import numpy as np
from display_data import display_data
from one_vs_all import one_vs_all as ova
from logreg_cost_function import h


# # --------------------- load data from .mat file ----------------
# file_path = 'C:/Users/uran-desktop/Documents/Octave/mlclass-ex3/mlclass-ex3/ex3data1.mat'
# mat = sio.loadmat(file_path)
#
# # -------------- slice digit images (X) and corresponding real values (Y)  --------
# X = mat['X']
# Y = mat['y']
# np.place(Y, Y == 10, 0)   # replace the label 10 with 0

file_path_X = 'C:/Users/uran-desktop/Documents/Octave/mlclass-ex3/mlclass-ex3/ex3data1X.npy'
file_path_Y = 'C:/Users/uran-desktop/Documents/Octave/mlclass-ex3/mlclass-ex3/ex3data1Y.npy'

# ----------------- load input data  -----------
X = np.load(file_path_X)
Y = np.load(file_path_Y)

numExamples = X.shape[0]  # 5000 examples
numFeatures = X.shape[1]  # 400 features
numLabels = 10            # digits from 0 to 9
X1 = np.append(np.ones((numExamples, 1)), X, axis=1)

# ------------- select random 100 rows from original data and display it -----------
select = X[np.random.permutation(numExamples)[0:100], :]
display_data(select)

# ----------------------------------- train model ----------------------------------
_lambda = 0.1
success, all_theta = ova(X1, Y, numLabels, _lambda)
if success:
    print("Result of optimisation successful")

# ---------------------------- compute training accuracy ---------------------------
predictions = h(X1 @ all_theta.T).argmax(axis=1).reshape((numExamples, 1))
# argmax() returns the indices of the maximum values along an axis.
accuracy = float(np.mean(predictions == Y)*100)
print("Train Accuracy: %5.2f" % accuracy)

# ------------------------ test arbitrary training set -----------------------------
if h(X1[519, :] @ all_theta[1, :].T) >= 0.5:
    print('success: it`s digit 1')
else:
    print('recognition fails')
