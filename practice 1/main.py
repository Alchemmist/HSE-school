import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


def plot_data(x, y, x_test, y_pred, gap=0.1):
    x_lim = np.min(x), np.max(x)
    x_lim = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.1, x_lim[1] + (x_lim[1] - x_lim[0]) * 0.1

    plt.figure(figsize=(8, 8))
    plt.xlim(*x_lim)


    xx, yy = np.meshgrid(np.linspace(x_lim[0], x_lim[1], 200), 
                         np.linspace(x_lim[0], x_lim[1], 200))
    x_mesh = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    y_mesh = model.predict(x_mesh)
    #y_pred = model.predict(x)
    y_unique = np.unique(y_mesh)
    colors = [np.random.rand(3) for i in y_unique]

    cm = ListedColormap(colors)
    plt.pcolormesh(xx, yy, y_mesh.reshape(xx.shape), cmap=cm)
    #plt.scatter(xx.reshape(-1), yy.reshape(-1), color="black", s=1)

    plt.scatter(x[y==0, 0], x[y==0, 1], label="Класс 0", color=(0.8, 0., 0.))
    plt.scatter(x[y==1, 0], x[y==1, 1], label="Класс 1", color=(0., 0.8, 0.))
    plt.scatter(x_test[y_pred==0, 0], x_test[y_pred==0, 1], label="Класс 0 (предсказание)", edgecolor="black", color=(0.8, 0., 0.))
    plt.scatter(x_test[y_pred==1, 0], x_test[y_pred==1, 1], label="Класс 1 (предсказание)", edgecolor="black", color=(0., 0.8, 0.))

    plt.grid()
    plt.legend()
    plt.tight_layout()


x, y = make_moons(250, noise=0.1)
model = KMeans(5)
model.fit(x)

x_new = np.random.randn(20, 2) * 0.5
y_pred = model.predict(x_new)

plot_data(x, y, x_new, y_pred)

plt.show()
