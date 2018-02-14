from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from scipy.sparse import vstack

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

import numpy as np
import datetime as dt
import itertools
import click

import features
import scorer
import reader
import utils


class TokensClassifier(BaseEstimator):
    """
    Обертка для классификаторов, работающих с отдельными токенами,
    переводящая их на уровень документов, чтобы можно было работать
    с кросс-валидацией в GridSearchCV и получать оценку scorer-а

    TODO - Хотелось бы передавать класс не строкой, но это ломает "copy"
    """

    def __init__(self, **params):
        self.cls = params['cls']
        params.pop('cls')

        if self.cls == 'LogisticRegression':
            self.obj = LogisticRegression(**params)
        if self.cls == 'RandomForestClassifier':
            self.obj = RandomForestClassifier(**params)
        if self.cls == 'XGBClassifier':
            self.obj = XGBClassifier(**params)
        if self.cls == 'LinearSVC':
            self.obj = LinearSVC(**params)

    def fit(self, X_docs, y_docs):
        """
        Отвечает за обучение внутреннего классификатора
        :param X_docs: Данные в формате документов
        :param y_docs: Ответы в формате документов
        :return:
        """
        # print(X_docs[:2])
        X_sent = list(itertools.chain.from_iterable(X_docs))
        y_sent = list(itertools.chain.from_iterable(y_docs))

        X = vstack(X_sent, dtype=np.int8)
        y = list(itertools.chain(*y_sent))
        self.obj.fit(X, y)
        return self

    def predict(self, X_docs):
        """
        Отвечает за предсказание ответа на данных
        :param X_docs: Данные для предсказания в формате документов
        :return:
        """
        # print(X_docs[:2])
        X_sent = list(itertools.chain.from_iterable(X_docs))
        X = vstack(X_sent, dtype=np.int8)
        y_pred = self.obj.predict(X)

        y_pred_sent = []
        index = 0
        for sent in X_sent:
            length = sent.shape[0]
            y_pred_sent.append(y_pred[index:index + length])
            index += length

        y_pred_docs = []
        index = 0
        for doc in X_docs:
            length = len(doc)
            if length == 0:
                continue
            y_pred_docs.append(y_pred_sent[index:index + length])
            index += length

        return y_pred_docs

    def score(self, X_docs, y_real_docs):
        """
        Отвечает за оценку результатов на некоторых данных
        :param X_docs: Данные для оценки в формате документов
        :param y_docs: Ответ на данных в формате документов
        :return:
        """
        y_pred_docs = self.predict(X_docs)

        y_pred_sent = list(itertools.chain(*y_pred_docs))
        y_real_sent = list(itertools.chain(*y_real_docs))

        enc = utils.LabelEncoder()
        y_pred_sent = [[enc.get(el) for el in arr] for arr in y_pred_sent]
        y_real_sent = [[enc.get(el) for el in arr] for arr in y_real_sent]
        labels = ["PER", "ORG", "LOC", "MISC"]
        result = scorer.Scorer.get_total_f1(labels, y_pred_sent, y_real_sent, enc)
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


@click.command()
@click.option('--mode', '-m', default='demo',
              help='Mode of running (demo, log, svn, gb, rf, all).')
def run_baselines(mode):
    print(f"Running in {mode} mode")

    dataset_path = "./prepared_data/conll_trainset.npz"
    dataset = reader.DataReader('./dataset', fileids='eng.train.txt',
                                columntypes=('words', 'pos', 'chunk', 'ne'))
    gen = features.Generator(column_types=['WORD', 'POS', 'CHUNK'], context_len=2, language='en')

    # В самой простой форме - просто массив слов подряд
    y = [el[1] for el in dataset.get_ne()]
    X = gen.fit_transform(dataset.get_tags(tags=['words', 'pos', 'chunk']), y, dataset_path)

    # Разбиваем массив на предложения
    X_sent = []
    y_sent = []

    index = 0
    for sent in dataset.sents():
        length = len(sent)
        if length == 0:
            continue
        X_sent.append(X[index:index + length])
        y_sent.append(y[index:index + length])
        index += length

    # Теперь разбиваем предложения на документы
    X_docs = []
    y_docs = []
    index = 0
    for doc in dataset.docs():
        length = len(doc)
        if length == 0:
            continue
        X_docs.append(X_sent[index:index + length])
        y_docs.append(y_sent[index:index + length])
        index += length

    parameters_gradient_boosting_demo = [{"booster": ["gbtree"]}]
    parameters_logistic_regression = [{"C": [0.001, 0.01, 0.1, 1, 10, 100]}]
    parameters_linear_svc = [{"C": [0.001, 0.01, 0.1, 1, 10]}]

    parameters_gradient_boosting = [{
        'max_depth': [5, 10, 15],
        'min_samples_split': [200, 500, 1000],
        'n_estimators': [10, 50, 100, 500],
        'max_features': [0.5, 0.75, 0.90],
        'subsample': [0.6, 0.8, 0.9]
    }]

    parameters_random_forest = [{
        "criterion": ["gini", "entropy"],
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 5],
        "min_samples_split": [15, 20],
        "min_samples_leaf": [5, 10],
        "max_leaf_nodes": [20, 40],
        "min_weight_fraction_leaf": [0.1]
    }]

    refit = False
    file = open('./baselines.txt', 'a+')

    if mode == 'demo':
        file.write('## XGBClassifier DEMO ##\n')
        file.write(f"started: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        clf = GridSearchCV(TokensClassifier(cls="XGBClassifier"),
                           parameters_gradient_boosting_demo, n_jobs=-1, cv=3,
                           refit=refit)
        clf.fit(X_docs, y_docs)
        file.write(f"best parameters: {clf.best_params_}\n")
        file.write(f"best result: {clf.best_score_}\n")
        file.write(f"ended: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        file.write("\n")

    if mode == 'log' or mode == 'all':
        file.write('## LogisticRegression ##\n')
        file.write(f"started: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        clf = GridSearchCV(TokensClassifier(cls="LogisticRegression"),
                           parameters_logistic_regression, n_jobs=-1, cv=3,
                           refit=refit)
        clf.fit(X_docs, y_docs)
        file.write(f"best parameters: {clf.best_params_}\n")
        file.write(f"best result: {clf.best_score_}\n")
        file.write(f"ended: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        file.write("\n")

    if mode == 'svn' or mode == 'all':
        file.write('## LinearSVC ##\n')
        file.write(f"started: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        clf = GridSearchCV(TokensClassifier(cls="LinearSVC"),
                           parameters_linear_svc,
                           n_jobs=-1, cv=3, refit=refit)
        clf.fit(X_docs, y_docs)
        file.write(f"best parameters: {clf.best_params_}\n")
        file.write(f"best result: {clf.best_score_}\n")
        file.write(f"ended: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        file.write("\n")

    if mode == 'gb' or mode == 'all':
        file.write('## XGBClassifier ##\n')
        file.write(f"started: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        clf = GridSearchCV(TokensClassifier(cls="XGBClassifier"),
                           parameters_gradient_boosting,
                           n_jobs=-1, cv=3, refit=refit)
        clf.fit(X_docs, y_docs)
        file.write(f"best parameters: {clf.best_params_}\n")
        file.write(f"best result: {clf.best_score_}\n")
        file.write(f"ended: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        file.write("\n")

    if mode == 'rf' or mode == 'all':
        file.write('## RandomForestClassifier ##\n')
        file.write(f"started: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        clf = GridSearchCV(TokensClassifier(cls="RandomForestClassifier"),
                           parameters_random_forest,
                           n_jobs=-1, cv=3, refit=refit)
        clf.fit(X_docs, y_docs)
        file.write(f"best parameters: {clf.best_params_}\n")
        file.write(f"best result: {clf.best_score_}\n")
        file.write(f"ended: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        file.write("\n")

        file.close()


if __name__ == '__main__':
    run_baselines()