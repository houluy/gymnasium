import matplotlib.pyplot as plt
import numpy as np


def fourier_feature(s, c):
    return np.cos(np.pi * np.dot(s, c))


def fourier_basis(s_dim):

    #fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4))
    x = np.linspace(0, 1, 10000)
    y = np.linspace(0, 1, 10000)
    X, Y = np.meshgrid(x, y)
    allc = [(0, 1), (1, 0), (1, 1), (0, 5), (2, 5), (5, 2)]
    for ind, c in enumerate(allc):
        ax = plt.subplot(2, 3, ind+1)
        Xv = X*c[0]
        Yv = Y*c[1]
        Zv = np.cos(np.pi * (Xv + Yv))
        ax.contour(X, Y, Zv)
    plt.show()


fourier_basis(1)
