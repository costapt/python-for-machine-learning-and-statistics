"""Helper methods for visualization."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import load_digits


def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC."""
    plot_decision_function(clf.decision_function, [-1, 0, 1], ax)


def plot_proba_function(clf, ax=None):
    """Plot the decision function for a classifier with predict_proba."""
    fn = lambda x: clf.predict_proba(x)[0][0]
    plot_decision_function(fn, [0, 0.5, 1], ax)


def plot_decision_function(fn, levels, ax=None):
    """Plot the decision function for a given classifier function."""
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            data_point = np.array([xi, yj]).reshape(1, -1)
            P[i, j] = fn(data_point)
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=levels, alpha=0.5,
               linestyles=['--', '-', '--'])


def plot_svm(N=5, std=0.60, kernel='linear'):
    """Plot SVM and its decision function.

    Parameters:
    N - Number of datapoints used to train the SVM.
    kernel - the kernel of the SVM.
    """
    X, y = make_blobs(n_samples=200, centers=2, random_state=0,
                      cluster_std=std)

    X_train, y_train = X[:N], y[:N]
    X_test, y_test = X[N:], y[N:]

    clf = SVC(kernel=str(kernel))
    clf.fit(X_train, y_train)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='spring')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='spring',
                alpha=0.2)
    plt.xlim(-1, 4)
    plt.ylim(-1, 6)
    plot_svc_decision_function(clf, plt.gca())
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=200, facecolors='none')

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test) if len(X_test) > 0 else 'NA'
    plt.title('Train Accuracy = {0}; Test Accuracy = {1}'.format(train_score,
                                                                 test_score))


def plot_iris_dataset():
    """Pair plot of the iris dataset."""
    iris = sns.load_dataset("iris")
    sns.pairplot(iris, hue='species')


def plot_dataset(X, classes):
    """Plot a 2-D dataset."""
    data = pd.DataFrame(X, columns=['x', 'y'])
    data['dataset'] = classes
    sns.lmplot('x', 'y', data=data, hue='dataset', fit_reg=False, size=10,
               palette=sns.color_palette("Set3", 10),
               scatter_kws={"s": 75})


def plot_digits():
    """Plot some example digits."""
    digits = load_digits()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(digits.images[i], cmap='binary')
        plt.axis('off')

    plt.show()
