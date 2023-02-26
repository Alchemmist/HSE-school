"""
Раскрашиваем точки по линейной регрессии
тоесть градиент по точкам равномерно распределёно
по точкам, от жёлтого к синему
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.colors import Normalize
import matplotlib as mpl
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_blobs, make_moons


def make_liner_regression(num_points, noise=0.1):
    x, _ = make_moons(num_points, noise=noise)
    weights = np.random.randn(3)
    y = x.dot(weights[:2]) + weights[-1]
    y = y + noise * np.random.randn(len(y))
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


x, y = make_liner_regression(150, noise=0.1)
plot_data_regr2d(x, y)
plt.show()
