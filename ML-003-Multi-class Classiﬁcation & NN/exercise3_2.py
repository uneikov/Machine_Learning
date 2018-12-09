"""
Neural Network - Model Presentation
Here we have implementation a neural network to recognize handwritten digits.
This neural network able to represent complex models that form non-linear hypotheses.
We will be using parameters from a neural network that we have already trained.
Our goal is to implement the feedforward propagation algorithm to use our weights for prediction.
In next topic, we will write the backpropagation algorithm for learning the neural network parameters.
"""

import numpy as np
from logreg_cost_function import h
from pathlib import Path

ML_dir = Path.cwd().parent
file_path_X = ML_dir / 'Training data/ex3data1X.npy'
file_path_Y = ML_dir / 'Training data/ex3data1Y.npy'
file_path_T1 = ML_dir / 'Training data/ex3weights1.npy'
file_path_T2 = ML_dir / 'Training data/ex3weights2.npy'

# Load input data
X = np.load(file_path_X)
Y = np.load(file_path_Y)
theta1 = np.load(file_path_T1)
theta2 = np.load(file_path_T2)

numExamples = X.shape[0]  # 5000 examples
numFeatures = X.shape[1]  # 400 features
numLabels = 10            # digits from 0 to 9
X1 = np.append(np.ones((numExamples, 1)), X, axis=1)

a1 = np.append(np.ones((numExamples, 1)), X, axis=1)
a2 = h(a1 @ theta1.T)
a2 = np.append(np.ones((numExamples, 1)), a2, axis=1)
a3 = h(a2 @ theta2.T)
predictions = a3.argmax(axis=1)
# argmax() returns the indices of the maximum values along an axis.
predictions = predictions.reshape((numExamples, 1))
accuracy = float(np.mean(predictions == Y)*100)
print("Train Accuracy: %5.2f" % accuracy)
