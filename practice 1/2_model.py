"""
Добавляем модель, её предсказание это закрашенные области.
То есть модель считает что в красной области будут крассные точки
а в зелёной зелёные. Вот и предсказание
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap


def plot_data(x, y, x_test, y_pred, model, gap=0.1): 
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * gap, x_lim[1]\
            + (x_lim[1] - x_lim[0]) * gap 

    plt.figure(figsize=(6, 6))
    plt.xlim(*x_lim)
    plt.ylim(*x_lim)

    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200), 
                         np.linspace(x_lim[0], x_lim[1], 200))
    x_mesh = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    y_mesh = model.predict(x_mesh)

    cm = ListedColormap([
        [0.9, 0.7, 0.7],
        [0.7, 0.9, 0.7]
    ])
    plt.pcolormesh(xx, yy, y_mesh.reshape(xx.shape), cmap=cm)

    plt.scatter(x[y==0, 0], 
                x[y==0, 1], 
                label='класс 1', 
                color=(0.8, 0., 0.))
    plt.scatter(x[y==1, 0], 
                x[y==1, 1], 
                label='класс 2', 
                color=(0., 0.8, 0.))

    plt.scatter(x_test[y_pred==0, 0], 
                x_test[y_pred==0, 1], 
                label="класс 1 (предсказание)", 
                edgecolor="black", 
                color=(0.8, 0., 0.))
    plt.scatter(x_test[y_pred==1, 0], 
                x_test[y_pred==1, 1], 
                label="класс 2 (предсказание)", 
                edgecolor="black", 
                color=(0., 0.8, 0.))

    plt.grid()
    plt.legend()
    plt.tight_layout()


x, y = make_moons(150, noise=0.1)
model = KNeighborsClassifier(5)  # создаём модель
model.fit(x, y)  # обучаем модель

x_new = np.random.randn(10, 2)
y_pred = model.predict(x_new)
plot_data(x, y, x_new, y_pred, model)
plt.show()
