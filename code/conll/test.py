from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator

from sklearn.linear_model import LogisticRegression

from itertools import accumulate

import estimator
import generator
import corpus
import utils


class Scorer:
    def __init__(self, X, y, X_st, n_splits=5):
        self.X_ = X
        self.y_ = y
        self.X_st_ = X_st
        self.n_splits_ = n_splits

        # индексы текущих train и test
        self.curr_train_st_ = None
        self.curr_test_st_ = None

    def cv(self):
        kf = KFold(self.n_splits_)
        for train, test in kf.split(self.X_st_):
            self.curr_train_st_ = train, self.curr_test_st_ = test
            yield (self.get_arr_indexes(train), self.get_arr_indexes(test))

    def get_arr_indexes(self, index_arr):
        total = []
        for index in index_arr:
            begin = accumulate([len(x) for x in self.X_st_[:index]])
            total += list(range(begin, begin + len(self.X_st_[index]), 1))
        return total

    def score(self, y_pred, y_ideal):
        encoder = utils.LabelEncoder()
        y_pred_sent = []
        y_ideal_sent = []

        index = 0
        for sent in self.get_sents_(self.curr_test_st_):
            length = len(sent)
            y_pred_sent.append([encoder.get(el) for el in y_pred[index:index + length]])
            y_ideal_sent.append([encoder.get(el) for el in y_ideal[index:index + length]])
            index += length

        labels = ["PER", "ORG", "LOC", "MISC"]
        return estimator.Estimator.get_total_f1(labels, y_pred_sent, y_ideal_sent, encoder)

    def get_sents_(self, index_arr):
        sents = []
        for index in index_arr:
            sents.append(self.X_st_[index])
        return sents


class EstimatorProxy(BaseEstimator):
    def __init__(self, method='tokens', cls=LogisticRegression):
        self.method = method
        self.cls = cls
        self.obj = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.obj = self.cls()
        self.obj.fit(X, y)
        return self

    def predict(self, X):
        return self.obj.predict(X)


if __name__ == '__main__':

    parameters_logistic_regression = [
        {
            "C": [0.01, 0.1, 1],
        }
    ]

    TRAINSET_PATH = "./prepared_data/conll_trainset.npz"

    conll_trainset = corpus.ConllDataReader('./dataset',
                                            fileids='eng.train.txt',
                                            columntypes=('words', 'pos', 'chunk', 'ne'))

    features_gen = generator.Generator(column_types=['WORD', 'POS', 'CHUNK'], context_len=2, language='en')

    y = [el[1] for el in conll_trainset.get_ne()]
    X = features_gen.fit_transform(conll_trainset.get_tags(tags=['words', 'pos', 'chunk']), y,
                                   path=TRAINSET_PATH)
    X_st = [x for x in conll_trainset.sents()]

    scorer = Scorer(X, y, X_st)

    clf = GridSearchCV(EstimatorProxy(), parameters_logistic_regression, scoring=scorer.score, cv=scorer.cv)
    clf.fit(X, y)
    print(clf.best_params_)
