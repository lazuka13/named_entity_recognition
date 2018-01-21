class Estimator:
    def __init__(self, method):
        self.method = method

    def cv(self):
        pass

    def estimate(self):
        pass


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class CoolEstimator(BaseEstimator):

    def __init__(self, method='tokens'):
        self.method = method

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):

        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X)

        return self.y_[closest]


# Импортируем все
from sklearn.model_selection import GridSearchCV

parameters_logistic_regression = [
    {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
]
clf = GridSearchCV(svc, parameters_logistic_regression)
