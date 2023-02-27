"""
Тут мы делим точки на группы по метду
графов. 
У нас есть некое расстояние epsila которое определяет когда мы связываем
2 точки ребром. По итогу у нас получается 2 больших связных графа, это и
есть наши луны
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


def plot_clusters(x, y, gap=0.1):
    x_lim = np.min(x), np.max(x)  # координаты крайних точек
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * gap, x_lim[1]\
            + (x_lim[1] - x_lim[0]) * gap  # отступы по краям

    plt.figure(figsize=(6, 6))  # рисует прямоугольник
    plt.xlim(x_lim)  # задает границы осей координат
    plt.ylim(x_lim)

    y_unique = np.unique(y)  # определяет сколько нужно цветов, \
                             # исходя из того сколько классов точек
    colors = [np.random.rand(3) for _ in y_unique]  # подбирает рандомные значения цветов

    # покраска вершин в нужные цвета
    for y_v, c in zip(y_unique, colors):
        plt.scatter(x[y == y_v, 0], 
                    x[y == y_v, 1], 
                    s=70, 
                    label=f"Класс {y_v}", 
                    color=c, 
                    linewidths=1,
                    edgecolors="black")

    plt.grid()  # делает сеточку
    plt.legend()  # выводит легенду
    plt.tight_layout()  # выравнивает сеточку


x, _ = make_moons(250, noise=0.1)
model = DBSCAN(0.2, min_samples=4)
y = model.fit_predict(x)
plot_clusters(x, y)

plt.show()
