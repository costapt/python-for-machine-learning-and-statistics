from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from visualization_helper import plot_rbm_weights


def deep_belief_network():
    """Train a two layer deep belief network."""
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        train_size=0.7, random_state=0)

    rbm1 = BernoulliRBM(n_components=144, n_iter=50, learning_rate=0.06)
    rbm2 = BernoulliRBM(n_components=100, n_iter=50, learning_rate=0.06)
    logistic = LogisticRegression(C=6000)
    scaler = MinMaxScaler()

    pipeline = Pipeline([('scaler', scaler), ('rbm1', rbm1), ('rbm2', rbm2),
                         ('logistic', logistic)])

    pipeline.fit(X_train, y_train)
    print 'Score on train set =', pipeline.score(X_train, y_train)
    print 'Score on test set =', pipeline.score(X_test, y_test)

    plot_rbm_weights(rbm1)
    plot_rbm_weights(rbm2, shape=(12, 12))
    return pipeline


def logistic_regression_search(X_train, y_train, X_test, y_test):
    """Search for a logistic regression's hyper-parameters."""
    clf = LogisticRegression()

    param_grid = [
        {'C': [1, 1e1, 1e2], 'penalty': ['l1', 'l2']},
    ]

    grid = GridSearchCV(clf, param_grid=param_grid, cv=3)
    grid.fit(X_train, y_train)

    print 'Grid Search best score =', grid.best_score_
    print 'Grid Search best parameters =', grid.best_params_
    print 'Accuracy (train) =', grid.score(X_train, y_train)
    print 'Accuracy (test) =', grid.score(X_test, y_test)
