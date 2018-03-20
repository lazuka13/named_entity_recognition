import itertools
from uuid import uuid4

import sklearn_crfsuite
from sklearn.base import BaseEstimator

import scorer
import utils


class SequenceClassifier(BaseEstimator):
    """
    Обертка для классификаторов, работающих с последовательностями токенов,
    переводящая их на уровень документов, чтобы можно было работать
    с кросс-валидацией в GridSearchCV и получать оценку scorer-а
    """

    def __init__(self, **params):
        self.cls = params['cls']
        self.file_name = str(uuid4()) + '.crfsuite'
        params.pop('cls')
        
        algorithm = params['algorithm'] if 'algorithm' in params else 'lbfgs'
        c1 = params['c1'] if 'c1' in params else 0.1
        c2 = params['c2'] if 'c2' in params else 0.1
        max_iterations = params['max_iterations'] if 'max_iterations' in params else 100
        all_possible_transitions = params['all_possible_transitions'] if 'all_possible_transitions' in params else True

        if self.cls == 'CRF':
            self.obj = sklearn_crfsuite.CRF(
                algorithm=algorithm,
                c1=c1,
                c2=c2,
                max_iterations=max_iterations,
                all_possible_transitions=all_possible_transitions
            )

    def fit(self, x_docs, y_docs):
        """
        Отвечает за обучение внутреннего классификатора
        :param x_docs: Данные в формате документов
        :param y_docs: Ответы в формате документов
        :return:
        """
        x_sents = list(itertools.chain.from_iterable(x_docs))
        y_sents = list(itertools.chain.from_iterable(y_docs))

        self.obj.fit(x_sents, y_sents)
        return self

    def predict(self, x_docs):
        """
        Отвечает за предсказание ответа на данных
        :param x_docs: Данные для предсказания в формате документов
        :return:
        """
        x_sents = list(itertools.chain.from_iterable(x_docs))

        y_pred_sents = self.obj.predict(x_sents)

        y_pred_docs = []
        index = 0
        for doc in x_docs:
            length = len(doc)
            if length == 0:
                continue
            y_pred_docs.append(y_pred_sents[index:index + length])
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
    
    def get_full_score(self, x_docs, y_real_docs):
        """
        Отвечает за получение полного отчета
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
        scorer.Scorer.get_full_score(labels, y_pred_sent, y_real_sent, enc)

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
