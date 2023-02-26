import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN


def plot_clusters(x, y, gap=0.1, alpha=0.8):
    x_lim = np.min(x), np.max(x)  # координаты крайних точек
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * gap, x_lim[1] + (x_lim[1] - x_lim[0]) * gap  # отступы по краям

    plt.figure(figsize=(9, 9))  # рисует прямоугольник
    plt.xlim(x_lim)  # задает границы осей координат
    plt.ylim(x_lim)

    y_unique = np.unique(y)  # выделяет уникальные значения в массиве цветов
    colors = [np.random.rand(3) for _ in y_unique]  # подбирает рандомные значения цветов

    # покраска вершин в нужные цвета
    for y_v, c in zip(y_unique, colors):
        plt.scatter(x[y == y_v, 0], x[y == y_v, 1], s=70, label=f"Класс {y_v}", color=c, linewidths=1,
                    edgecolors="black")

    plt.grid()  # делает сеточку
    plt.legend()  # выводит легенду
    plt.tight_layout()  # выравнивает сеточку


x, _ = make_moons(250, noise=0.1)
model = DBSCAN(0.2, min_samples=4)
y = model.fit_predict(x)
plot_clusters(x, y)





plt.show()
