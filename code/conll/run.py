from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator
from scipy.sparse import vstack

from sklearn.linear_model import LogisticRegression

import numpy as np

import estimator as est
import generator as gen
import utils
import corpus


class LogisticRegressionTokens(BaseEstimator):
    def __init__(self, **params):
        self.cls = LogisticRegression
        self.obj = self.cls(**params)

    def fit(self, X_st, y_st):
        X = vstack(X_st, dtype=np.int8)
        y = []
        for y_el in y_st:
            y += y_el
        self.obj.fit(X, y)
        return self

    def predict(self, X_st):
        X = vstack(X_st, dtype=np.int8)
        y_pred = self.obj.predict(X)
        y_pred_sent = []
        index = 0
        for sent in X_st:
            length = sent.shape[0]
            y_pred_sent.append(y_pred[index:index + length])
            index += length
        return y_pred_sent

    def score(self, X_st, y_st):
        enc = utils.LabelEncoder()
        y_pred_st = self.predict(X_st)
        y_real_st = [[enc.get(el) for el in arr] for arr in y_st]
        labels = ["PER", "ORG", "LOC", "MISC"]
        return est.Estimator.get_total_f1(labels, y_pred_st, y_real_st, enc)

    def get_params(self, deep=True):
        return self.obj.get_params()

    def set_params(self, **params):
        self.obj.set_params(**params)
        return self


if __name__ == '__main__':
    parameters_logistic_regression = [
        {
            "C": [1],
        }
    ]

    TRAINSET_PATH = "./prepared_data/conll_trainset.npz"

    conll_trainset = corpus.ConllDataReader('./dataset',
                                            fileids='eng.train.txt',
                                            columntypes=('words', 'pos', 'chunk', 'ne'))

    gen = gen.Generator(column_types=['WORD', 'POS', 'CHUNK'], context_len=2, language='en')

    y = [el[1] for el in conll_trainset.get_ne()]
    X = gen.fit_transform(conll_trainset.get_tags(tags=['words', 'pos', 'chunk']), y,
                          path=TRAINSET_PATH)
    X_sent = []
    y_sent = []
    index = 0
    for sent in conll_trainset.sents():
        length = len(sent)
        X_sent.append(X[index:index + length])
        y_sent.append(y[index:index + length])
        index += length

    clf = GridSearchCV(LogisticRegressionTokens(), parameters_logistic_regression, n_jobs=3)
    clf.fit(X_sent, y_sent)
    print(clf.best_params_)
