from sklearn.base import BaseEstimator
from scipy.sparse import vstack

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .xgbclassifier import XGBClassifier
from sklearn.svm import LinearSVC

import numpy as np
import itertools

import scorer
import utils


class TokenClassifier(BaseEstimator):
    """
    Обертка для классификаторов, работающих с отдельными токенами,
    переводящая их на уровень документов, чтобы можно было работать
    с кросс-валидацией в GridSearchCV и получать оценку scorer-а
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

    def fit(self, x_docs, y_docs):
        """
        Отвечает за обучение внутреннего классификатора
        :param x_docs: Данные в формате документов
        :param y_docs: Ответы в формате документов
        :return:
        """
        x_sent = list(itertools.chain.from_iterable(x_docs))
        y_sent = list(itertools.chain.from_iterable(y_docs))

        x = vstack(x_sent, dtype=np.int8)
        y = list(itertools.chain(*y_sent))
        if self.cls != 'XGBClassifier':
            self.obj.fit(x, y)
        else:
            self.obj.fit(x, y)
        return self

    def predict(self, x_docs):
        """
        Отвечает за предсказание ответа на данных
        :param x_docs: Данные для предсказания в формате документов
        :return:
        """
        x_sent = list(itertools.chain.from_iterable(x_docs))
        x = vstack(x_sent, dtype=np.int8)
        y_pred = self.obj.predict(x)

        y_pred_sent = []
        index = 0
        for sent in x_sent:
            length = sent.shape[0]
            y_pred_sent.append(y_pred[index:index + length])
            index += length

        y_pred_docs = []
        index = 0
        for doc in x_docs:
            length = len(doc)
            if length == 0:
                continue
            y_pred_docs.append(y_pred_sent[index:index + length])
            index += length

        return y_pred_docs

    def score(self, x_docs, y_real_docs):
        """
        Отвечает за оценку результатов на некоторых данных
        :param x_docs: Данные для оценки в формате документов
        :param y_real_docs: Ответ на данных в формате документов
        :return:
        """
        y_pred_docs = self.predict(x_docs)

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