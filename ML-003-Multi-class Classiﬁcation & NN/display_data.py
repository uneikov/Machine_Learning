import matplotlib.pyplot as plt
import numpy as np
import math


def display_data(sel):
    ew = round(math.sqrt(len(sel[0, :])))
    #pad = np.ones((ew, 1)) * 255
    big_img = np.empty((ew*10, ew*10))
    for row in range(10):
        for col in range(10):
            #big_img[row*ew:row*ew+ew, col*ew:col*ew+ew] = np.concatenate((select[row*10+col, :].reshape((ew, ew)), pad), 1)
            big_img[row*ew:row*ew+ew, col*ew:col*ew+ew] = sel[row*10+col, :].reshape((ew, ew), order='F')
    plt.imshow(big_img, cmap='Greys')
    plt.show()
    return
