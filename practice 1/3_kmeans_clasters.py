"""
Тут решаем задачу класторизации.
Мы берём 3 кучки точек и разбиваем пространство вокруг них
на клсатеры в соответствии с кучками точек.
То есть есть кластер с первой кучкой, кластер со второй и 
кластер с тертьей.

Тоже самое можно было сделать и с лунами, но там ПРАВИЛЬНО 
поделить уже не получится, потому что кластеры определяются 
прямыми
"""


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


def plot_clusters(x, model, gap=0.1, alpha=0.8):
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * gap, x_lim[1] + (x_lim[1] - x_lim[0]) * gap

    plt.figure(figsize=(6, 6))
    plt.xlim(*x_lim)

    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200), 
                         np.linspace(x_lim[0], x_lim[1], 200))
    x_mesh = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    y_mesh = model.predict(x_mesh)
    y_unique = np.unique(y_mesh)
    colors = [np.random.rand(3) for _ in y_unique]

    cm = ListedColormap([alpha * np.array([1., 1., 1]) + (1 - alpha) * c for c in colors])
    plt.pcolormesh(xx, yy, y_mesh.reshape(xx.shape), cmap=cm, shading='gouraud')     # фон

    y_pred = model.predict(x)
    for y_v, c in zip(y_unique, colors):
        plt.scatter(x[y_pred==y_v, 0], 
                    x[y_pred==y_v, 1], 
                    s=30, 
                    label=f"Class {y_v}", 
                    color=c, 
                    linewidth=1,
                    edgecolor="black")

    for (_x, _y), c in zip(model.cluster_centers_, colors):
        plt.scatter([_x],
                    [_y], 
                    s=150, 
                    color=c, 
                    linewidth=3, 
                    edgecolor="black")

    plt.grid()
    plt.legend()
    plt.tight_layout()


x, _ = make_blobs(250)
model = KMeans(3)
model.fit(x)

plot_clusters(x, model)
plt.show()
