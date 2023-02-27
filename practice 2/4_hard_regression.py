"""
Аналогично как в предыдущем, но тут
у нас градиент расппределён не равонмерно а 
как то более сложно
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import make_moons


def make_regression_hard(num_points, noise=0.1):
    x, _ = make_moons(num_points, noise=noise)
    weights_1 = np.random.randn(3)
    weights_2 = np.random.randn(3)
    y1 = x.dot(weights_1[:2]) + weights_1[-1]
    y2 = np.sin(x.dot(weights_2[:2]) + weights_2[-1])
    y = y1 + (y2 + 1) * np.abs(y1)**0.5
    return x, y


def plot_data_regr2d(x, y, gap=0.1):
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * gap, x_lim[1]\
            + (x_lim[1] - x_lim[0]) * gap

    plt.figure(figsize=(6, 6))
    plt.xlim(*x_lim)
    plt.ylim(*x_lim)
   
    cm = mpl.colormaps["plasma"]
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cm)

    plt.grid()
    plt.tight_layout()


x, y = make_regression_hard(150, noise=0.1)
plot_data_regr2d(x, y)
plt.show()
