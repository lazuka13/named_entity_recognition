import itertools
from uuid import uuid4

import pycrfsuite
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

        if self.cls == 'CRF':
            self.trainer_obj = pycrfsuite.Trainer(verbose=False)
            self.tagger_obj = pycrfsuite.Tagger()

    def fit(self, x_docs, y_docs):
        """
        Отвечает за обучение внутреннего классификатора
        :param x_docs: Данные в формате документов
        :param y_docs: Ответы в формате документов
        :return:
        """
        x_sents = list(itertools.chain.from_iterable(x_docs))
        y_sents = list(itertools.chain.from_iterable(y_docs))

        for x_sent, y_sent in zip(x_sents, y_sents):
            self.trainer_obj.append(x_sent, y_sent)

        self.trainer_obj.train(model=self.file_name)
        return self

    def predict(self, x_docs):
        """
        Отвечает за предсказание ответа на данных
        :param x_docs: Данные для предсказания в формате документов
        :return:
        """
        x_sents = list(itertools.chain.from_iterable(x_docs))

        self.tagger_obj.open(self.file_name)
        y_pred_sents = []
        for x_sent in x_sents:
            y_pred_sents.append(self.tagger_obj.tag(x_sent))

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

    def get_params(self, deep=True):
        """
        Отвечает за получение параметров внутреннего классификатора
        :param deep:
        :return:
        """
        params = self.trainer_obj.get_params()
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
        self.trainer_obj.set_params(**params)
        return self
