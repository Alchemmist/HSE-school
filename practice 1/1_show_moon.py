"""
Пока нет ни какого ML просто генерим
датасет с двумя лунами и показываем их
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


def plot_data(x, y, gap=0.1): 
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * gap, x_lim[1]\
            + (x_lim[1] - x_lim[0]) * gap 

    plt.figure(figsize=(6, 6))
    plt.xlim(*x_lim)
    plt.ylim(*x_lim)
        
    plt.scatter(x[y==0, 0], x[y==0, 1], label='класс 1')
    #       x[y==0]   -------->     ([0.23, 1.23])
    #       x[y==0, 0]   -------->     0.23
    plt.scatter(x[y==1, 0], x[y==1, 1], label='класс 2')

    plt.grid()
    plt.legend()
    plt.tight_layout()


x, y = make_moons(150, noise=0.1)
plot_data(x, y)
plt.show()
