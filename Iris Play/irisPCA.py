import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris = load_iris()
X = iris.data
y = iris.target

#CHANGE THE NUMBER OF PCA COMPONENTS HERE!!!
n_components = 3

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

def plot2d(X_reduced, y):
    colors = ['navy', 'turquoise', 'darkorange']

    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
                    color=color, lw=2, label=target_name)


    plt.title( "PCA of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])

    plt.show()

def plot3d(X_reduced, y):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    colors = ['navy', 'turquoise', 'darkorange']
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        ax.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], X_reduced[y == i, 2], color=color, lw=2, label=target_name ,edgecolor='k', s=40)


    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.set_ylabel("2nd eigenvector")
    ax.set_zlabel("3rd eigenvector")
    ax.w_xaxis.set_ticklabels([])   
    ax.w_yaxis.set_ticklabels([])   
    ax.w_zaxis.set_ticklabels([])
    ax.legend(loc="best", shadow=False, scatterpoints=1)
    plt.show()

if n_components ==2:
    plot2d(X_pca,y)
else:
    plot3d(X_pca,y)