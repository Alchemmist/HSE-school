import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs, make_moons
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from matplotlib.colors import Normalize
import matplotlib as mpl
from sklearn.pipeline import Pipeline 


"""
def plot_data_classification(x, y, gap=0.1):
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.1, x_lim[1] + (x_lim[1] - x_lim[0]) * 0.1

    plt.figure(figsize=(8, 8))
    plt.xlim(*x_lim)
    plt.ylim(*x_lim)

    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200), 
                         np.linspace(x_lim[0], x_lim[1], 200))
    x_mesh = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    y_mesh = model.predict(x_mesh)
    #y_pred = model.predict(x)
    y_unique = np.unique(y_mesh)
    colors = [np.random.rand(3) for i in y_unique]

    cm = ListedColormap(colors)
    plt.pcolormesh(xx, yy, y_mesh.reshape(xx.shape), cmap=cm, shading="ground")
    #plt.scatter(xx.reshape(-1), yy.reshape(-1), color="black", s=1)

    plt.scatter(x[y==0, 0], x[y==0, 1], label="Класс 0", color=(0.8, 0., 0.))
    plt.scatter(x[y==1, 0], x[y==1, 1], label="Класс 1", color=(0., 0.8, 0.))

    plt.grid()
    plt.legend()
    plt.tight_layout()
"""

def plot_data_classification_w_model(x, y, model, prep=None, gap=0.1):
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.1, x_lim[1] + (x_lim[1] - x_lim[0]) * 0.1

    plt.figure(figsize=(8, 8))
    plt.xlim(*x_lim)
    plt.ylim(*x_lim)

    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200), 
                         np.linspace(x_lim[0], x_lim[1], 200))
    x_mesh = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    if prep is not None:
        x_mesh = prep.transform(x_mesh)
    y_mesh = model.predict(x_mesh)
    #y_pred = model.predict(x)
    y_unique = np.unique(y_mesh)
    colors = [np.random.rand(3) for i in y_unique]

    cm = ListedColormap(colors)
    plt.pcolormesh(xx, yy, y_mesh.reshape(xx.shape), cmap=cm, shading="ground")
    #plt.scatter(xx.reshape(-1), yy.reshape(-1), color="black", s=1)

    plt.scatter(x[y==0, 0], x[y==0, 1], label="Класс 0", color=(0.8, 0., 0.))
    plt.scatter(x[y==1, 0], x[y==1, 1], label="Класс 1", color=(0., 0.8, 0.))
#    plt.scatter(x_test[y_pred==0, 0], x_test[y_pred==0, 1], label="Класс 0 (предсказание)", edgecolor="black", color=(0.8, 0., 0.))
#    plt.scatter(x_test[y_pred==1, 0], x_test[y_pred==1, 1], label="Класс 1 (предсказание)", edgecolor="black", color=(0., 0.8, 0.))

    plt.grid()
    plt.legend()
    plt.tight_layout()

def make_linear_regression(num_points, noise=0.1):
    x, _ = make_moons(num_points, noise=noise)
    weights = np.random.randn(3)
    y = x.dot(weights[:2] + weights[-1])
    y = y + noise * np.random.randn(len(y))
    return x, y

def plot_data_regr2d(x, y, gap=0.1):
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.1, x_lim[1] + (x_lim[1] - x_lim[0]) * 0.1

    plt.figure(figsize=(8, 8))
    plt.xlim(*x_lim)
    plt.ylim(*x_lim)

    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200), 
                         np.linspace(x_lim[0], x_lim[1], 200))
    x_mesh = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    y_mesh = model.predict(x_mesh)
    #y_pred = model.predict(x)
    y_unique = np.unique(y_mesh)
    colors = [np.random.rand(3) for i in y_unique]

    cm = mpl.colormaps("plasma")
    plt.pcolormesh(xx, yy, y_mesh.reshape(xx.shape), cmap=cm, shading="ground")
    #plt.scatter(xx.reshape(-1), yy.reshape(-1), color="black", s=1)
    plt.scatter(x[:, 0], x[:, 0])
    #plt.scatter(x[y==0, 0], x[y==0, 1], label="Класс 0", color=(0.8, 0., 0.))
    #plt.scatter(x[y==1, 0], x[y==1, 1], label="Класс 1", color=(0., 0.8, 0.))

    plt.grid()
    plt.tight_layout()

def make_regression_hard(num_points, noise=0.1):
    x, _ = make_moons(num_points, noise=noise)
    weights_1 = np.random.randn(3)
    weights_2 = np.random.randn(3)
    y1 = x.dot(weights_1[:2]) + weights_1[-1]
    
    


# prep = PolynomialFeatures(5)
# model = LogisticRegression()
# x_transformed = prep.fit_transform(x)
model =  Pipeline((
    ("scaler", StandardScaler()), 
    ("pf", PolynomialFeatures(4)),
    ("clf", LogisticRegression())
))

x, y = make_linear_regression(150, noise=0.09)
#model.fit(x, y)
plot_data_regr2d(x, y)
plt.show()
