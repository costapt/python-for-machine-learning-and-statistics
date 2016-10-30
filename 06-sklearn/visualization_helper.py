"""Helper methods for visualization."""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits


def plot_wrong_predictions(clf, X, y, width=2, height=2):
    """Plot the wrongly classified digits with the wrong and true value."""
    y_pred = clf.predict(X)

    idx = y != y_pred
    X_wrong = X[idx]
    y_pred_wrong = y_pred[idx]
    y_true_wrong = y[idx]

    n = width * height
    if n > len(X_wrong):
        n = len(X_wrong)

    for i in range(n):
        ax = plt.subplot(height, width, i+1)
        plt.imshow(X_wrong[i].reshape(8, 8))
        ax.text(0.95, 0.01, '{0}'.format(y_pred_wrong[i]),
                verticalalignment='bottom', horizontalalignment='right',
                color='red', fontsize=15, transform=ax.transAxes)
        ax.text(0.01, 0.95, '{0}'.format(y_true_wrong[i]),
                color='green', fontsize=15, transform=ax.transAxes)
        plt.axis('off')

    plt.show()


def plot_digits():
    """Plot some example digits."""
    digits = load_digits()
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(digits.images[i], cmap='binary')
        plt.axis('off')

    plt.show()


def plot_confusion_matrix(confusion_matrix):
    """Plot the given confusion_matrix."""
    ax = sns.heatmap(confusion_matrix, annot=True)
    ax.set(xlabel="predicted", ylabel="true")


def convert_to_numerical(m):
    maps = {'Porto': 0, 'Lisbon': 1, 'Faro': 3, 'short': 0, 'medium': 1,
            'tall': 2}

    return [[maps.get(i, i) for i in mi] for mi in m]


def print_dataset_statistics(X):
    np.set_printoptions(precision=3, suppress=True)
    print 'Mean of features = {0}'.format(X.mean(axis=0))
    print 'Standard Deviation of features = {0}'.format(X.std(axis=0))


def plot_normalized_dataset(X, X_normalized, X_test_normalized, y, y_train,
                            y_test):
    plt.subplot(2, 2, 1)
    plt.title('Before')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
    plt.xlim((-4.25, 4.25))
    plt.ylim((-2.5, 6))
    plt.subplot(2, 2, 2)
    plt.title('After')
    plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=y_train, s=50,
                cmap='spring')
    plt.scatter(X_test_normalized[:, 0], X_test_normalized[:, 1], c=y_test,
                s=50, cmap='spring')
    plt.xlim((-4.25, 4.25))
    plt.ylim((-2.5, 6))


def plot_rbm_weights(rbm, shape=(8, 8)):
    idxs = np.random.permutation(range(len(rbm.components_)))
    for i, idx in enumerate(idxs[:16]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(rbm.components_[idx].reshape(shape))
        plt.axis('off')
    plt.show()


def plot_score_curves(train_scores, valid_scores, x_axis):
    aux = np.concatenate((train_scores.T[:, :, np.newaxis],
                          valid_scores.T[:, :, np.newaxis]), axis=2)
    sns.tsplot(data=aux, time=x_axis, condition=['Train scores', 'Validation scores'])
