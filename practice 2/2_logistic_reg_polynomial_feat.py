"""
Тут мы строим логистическую регрессию
Но она пока что линейная, поэтому её разделение оставляет
желать лучшего. 
"""


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures


def plot_data_classification_w_model(x, y, model, prep=None, gap=0.1):
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * gap, x_lim[1] + (x_lim[1] - x_lim[0]) * gap

    plt.figure(figsize=(7, 7))
    plt.xlim(*x_lim) # Задаем границы координаты осей координат
    plt.ylim(*x_lim)

    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200),
                         np.linspace(x_lim[0], x_lim[1], 200))
    x_mesh = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    if prep is not None:
        x_mesh = prep.transform(x_mesh)
        
    y_mesh = model.predict(x_mesh)

    cm = ListedColormap([
        [0.9, 0.7, 0.7],
        [0.7, 0.9, 0.7]
    ])
    plt.pcolormesh(xx, yy, y_mesh.reshape(xx.shape), cmap=cm, shading='gouraud')

    plt.scatter(x[y==0, 0], 
                x[y==0, 1], 
                s=50, 
                label="Класс 0", 
                color=(0.8, 0., 0.), 
                linewidth=1, 
                edgecolor="black")
    plt.scatter(x[y==1, 0], 
                x[y==1, 1], 
                s=50, 
                label="Класс 1", 
                color=(0., 0.8, 0.), 
                linewidth=1, 
                edgecolor="black")

    plt.grid()
    plt.legend()
    plt.tight_layout()


x, y = make_moons(150, noise=0.1)
prep = PolynomialFeatures(3)
x_transformed = prep.fit_transform(x)

model = LogisticRegression()
model.fit(x_transformed, y)

plot_data_classification_w_model(x, y, model)
plt.show()
