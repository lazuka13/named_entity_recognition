from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from scipy.sparse import vstack

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.svm import LinearSVC

import numpy as np

import estimator as est
import generator as gen
import utils
import corpus


class TokensClassifier(BaseEstimator):
    """
    Обертка для классификаторов, работающих с отдельными токенами,
    переводящая их на уровень предложений
    """

    def __init__(self, **params):
        self.cls = params['cls']
        params.pop('cls')

        # TODO Хотелось бы передавать не строками, но пока не очень понятно,
        # что происходит внутри копирования классификатора (спросить?)
        if self.cls == 'LogisticRegression':
            self.obj = LogisticRegression(**params)
        if self.cls == 'RandomForestClassifier':
            self.obj = RandomForestClassifier(**params)
        if self.cls == 'GradientBoostingClassifier':
            self.obj = GradientBoostingClassifier(**params)
        if self.cls == 'LinearSVC':
            self.obj = LinearSVC(**params)

    def fit(self, X_st, y_st):
        """
        Отвечает за обучение внутреннего классификатора
        :param X_st: Данные в формате предложений
        :param y_st: Ответы в формате предложений
        :return:
        """
        X = vstack(X_st, dtype=np.int8)
        y = []
        for y_el in y_st:
            y += y_el
        self.obj.fit(X, y)
        return self

    def predict(self, X_st):
        """
        Отвечает за предсказание ответа на данных
        :param X_st: Данные для предсказания в формате предложений
        :return:
        """
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
        """
        Отвечает за оценку результатов на некоторых данных
        :param X_st: Данные для оценки в формате предложений
        :param y_st: Ответ на данных в формате предложений
        :return:
        """
        enc = utils.LabelEncoder()
        y_pred_st = [[enc.get(el) for el in arr] for arr in self.predict(X_st)]
        y_real_st = [[enc.get(el) for el in arr] for arr in y_st]
        labels = ["PER", "ORG", "LOC", "MISC"]
        result = est.Estimator.get_total_f1(labels, y_pred_st, y_real_st, enc)
        return result

    def get_params(self, deep=True):
        """
        Отвечает за получение параметров внутреннего классификатора
        :param deep:
        :return:
        """
        params = self.obj.get_params()
        params['cls'] = self.cls
        return params

    def set_params(self, **params):
        """
        Отвечает за установку параметров внутреннего классификатора
        :param params:
        :return:
        """
        if 'cls' in params:
            params.pop('cls')
        self.obj.set_params(**params)
        return self


if __name__ == '__main__':

    TRAINSET_PATH = "./prepared_data/conll_trainset.npz"
    dataset = corpus.ConllDataReader('./dataset',
                                     fileids='eng.train.txt',
                                     columntypes=(
                                         'words', 'pos', 'chunk', 'ne'))
    generator = gen.Generator(column_types=['WORD', 'POS', 'CHUNK'],
                              context_len=2, language='en')

    y = [el[1] for el in dataset.get_ne()]
    X = generator.fit_transform(
        dataset.get_tags(tags=['words', 'pos', 'chunk']), y,
        path=TRAINSET_PATH)
    X_sent = []
    y_sent = []
    index = 0
    for sent in dataset.sents():
        length = len(sent)
        X_sent.append(X[index:index + length])
        y_sent.append(y[index:index + length])
        index += length

    parameters_logistic_regression = [
        {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        }
    ]

    parameters_linear_svc = [
        {
            "C": [0.001, 0.01, 0.1, 1, 10]
        }
    ]

    parameters_gradient_boosting = [
        {
            'max_depth': [5, 10, 15],
            'min_samples_split': [200, 500, 1000],
            'n_estimators': [10, 50, 100, 500],
            'max_features': [0.5, 0.75, 0.90],
            'subsample': [0.6, 0.8, 0.9]
        }
    ]

    parameters_random_forest = [
        {
            "criterion": ["gini", "entropy"],
            "n_estimators": [100, 500, 1000],
            "max_depth": [3, 5],
            "min_samples_split": [15, 20],
            "min_samples_leaf": [5, 10],
            "max_leaf_nodes": [20, 40],
            "min_weight_fraction_leaf": [0.1]
        }
    ]

    refit = False

    file = open('./baselines.txt', 'a+')

    clf = GridSearchCV(TokensClassifier(cls="LogisticRegression"),
                       parameters_logistic_regression, n_jobs=4, cv=3,
                       refit=refit)
    #clf.fit(X_sent, y_sent)
    #file.write('## LogisticRegression ##\n')
    #file.write(f"best parameters: {clf.best_params_}\n")
    #file.write(f"best result: {clf.best_score_}\n")
    #file.write("\n")

    clf = GridSearchCV(TokensClassifier(cls="LinearSVC"), parameters_linear_svc,
                       n_jobs=4, cv=3, refit=refit)
    #clf.fit(X_sent, y_sent)
    #file.write('## LinearSVC ##\n')
    #file.write(f"best parameters: {clf.best_params_}\n")
    #file.write(f"best result: {clf.best_score_}\n")
    #file.write("\n")

    clf = GridSearchCV(TokensClassifier(cls="GradientBoostingClassifier"),
                       parameters_gradient_boosting,
                       n_jobs=4, cv=3, refit=refit)
    clf.fit(X_sent, y_sent)
    file.write('## GradientBoostingClassifier ##\n')
    file.write(f"best parameters: {clf.best_params_}\n")
    file.write(f"best result: {clf.best_score_}\n")
    file.write("\n")

    clf = GridSearchCV(TokensClassifier(cls="RandomForestClassifier"),
                       parameters_random_forest,
                       n_jobs=4, cv=3, refit=refit)
    clf.fit(X_sent, y_sent)
    file.write('## RandomForestClassifier ##\n')
    file.write(f"best parameters: {clf.best_params_}\n")
    file.write(f"best result: {clf.best_score_}\n")
    file.write("\n")

    file.close()
