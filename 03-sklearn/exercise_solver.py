import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from IPython.html.widgets import interact
from sklearn.linear_model import LogisticRegression
from visualization_helper import plot_proba_function
from sklearn.datasets.samples_generator import make_blobs


def solve_kmeans_exercise(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    for i in range(n_clusters if n_clusters <= 12 else 12):
        plt.subplot(4, 3, i+1)
        plt.imshow(kmeans.cluster_centers_[i].reshape((8, 8)))
        plt.axis('off')


def solve_logistic_regression_exercise():
    interact(plot_logistic_regression, N=[5, 200], C=[1, 1e3])


def solve_image_compression_exercise(image, n_clusters):
    X = (image / 255.0).reshape(-1, 3)

    clu = KMeans(n_clusters=n_clusters)
    clu.fit(X)

    colors = clu.cluster_centers_
    new_flower = colors[clu.predict(X)].reshape(image.shape)
    plt.imshow(new_flower)
    plt.grid(False)


def plot_logistic_regression(N=5, C=1):
    """Plot Logistic Regression and its decision function.

    Parameters:
    N - Number of datapoints used to train the SVM.
    C - the regularization term.
    """
    X, y = make_blobs(n_samples=200, centers=2, random_state=0,
                      cluster_std=0.60)

    X_train, y_train = X[:N], y[:N]
    X_test, y_test = X[N:], y[N:]

    clf = LogisticRegression(C=C)
    clf.fit(X_train, y_train)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='spring')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='spring',
                alpha=0.2)
    plt.xlim(-1, 4)
    plt.ylim(-1, 6)
    plot_proba_function(clf, plt.gca())

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test) if len(X_test) > 0 else 'NA'
    plt.title('Train Accuracy = {0}; Test Accuracy = {1}; coef = {2}'.format(
        train_score, test_score, clf.coef_))
