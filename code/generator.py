import pymorphy2
import re
import string
import os
from sklearn import preprocessing
import numpy as np


class ColumnApplier(object):
    def __init__(self, column_stages):
        self._column_stages = column_stages

    def fit(self, X, y):
        for i, k in self._column_stages.items():
            k.fit(X[:, i])
        return self

    def transform(self, X):
        X = X.copy()
        for i, k in self._column_stages.items():
            X[:, i] = k.transform(X[:, i])
        return X


class Generator:
    def __init__(self, column_types=None, context_len=2):
        """
        Конструктор
        :param features_types: Список желаемых признаков.
        :param context_len: Длина контекста.
        :param column_types: Типы передаваемых столбцов. ("word", "pos", "chunk")
        """
        self._column_types = column_types if column_types is not None else ["word"]
        self._context_len = context_len
        self._morph = pymorphy2.MorphAnalyzer()

    def generate(self, data, path):
        """
        Генерация признаков из данных
        :param data: Список слов для извлечения признаков.
        :return: features_list: Список признакiов (Лист листов)
        """
        if os.path.exists(path):
            sparse_features_list = np.load(path)
            return sparse_features_list

        data = [["" for i in range(len(self._column_types))] for i in range(self._context_len)] + data
        data = data + [["" for i in range(len(self._column_types))] for i in range(self._context_len)]
        features_list = []

        word_index = self._column_types.index("word")

        if "pos" in self._column_types:
            pos_index = self._column_types.index("pos")
        else:
            pos_index = None
        if "chunk" in self._column_types:
            chunk_index = self._column_types.index("chunk")
        else:
            chunk_index = None
        for k in range(len(data) - 2 * self._context_len):
            arr = []

            i = k + self._context_len
            word_arr = [data[i][word_index]]
            for j in range(1, self._context_len + 1):
                word_arr.append(data[i - j][word_index])
                word_arr.append(data[i + j][word_index])
            arr += word_arr

            if pos_index is not None:
                pos_arr = [data[i][pos_index]]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(data[i - j][pos_index])
                    pos_arr.append(data[i + j][pos_index])
            else:
                pos_arr = [self.get_pos_tag(data[i][word_index])]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(self.get_pos_tag(data[i][word_index]))
                    pos_arr.append(self.get_pos_tag(data[i][word_index]))
            arr += pos_arr

            if chunk_index is not None:
                chunk_arr = [data[i][chunk_index]]
                for j in range(1, self._context_len + 1):
                    chunk_arr.append(data[i - j][chunk_index])
                    chunk_arr.append(data[i + j][chunk_index])
                arr += chunk_arr

            capital_arr = [self.get_capital(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                capital_arr.append(self.get_capital(data[i - j][word_index]))
                capital_arr.append(self.get_capital(data[i + j][word_index]))
            arr += capital_arr

            is_punct_arr = [self.get_is_punct(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                is_punct_arr.append(self.get_is_punct(data[i - j][word_index]))
                is_punct_arr.append(self.get_is_punct(data[i + j][word_index]))
            arr += is_punct_arr

            is_number_arr = [self.get_is_number(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                is_number_arr.append(self.get_is_number(data[i - j][word_index]))
                is_number_arr.append(self.get_is_number(data[i + j][word_index]))
            arr += is_number_arr

            initial_arr = [self.get_initial(data[i][word_index])]
            for j in range(1, self._context_len + 1):
                initial_arr.append(self.get_initial(data[i - j][word_index]))
                initial_arr.append(self.get_initial(data[i + j][word_index]))
            arr += initial_arr

            features_list.append(arr)
        features_list = np.array([np.array(line) for line in features_list])

        multi_encoder = ColumnApplier(dict([(i, preprocessing.LabelEncoder()) for i in range(len(features_list[0]))]))
        sparse_features_list = multi_encoder.fit(features_list, None).transform(features_list)
        enc = preprocessing.OneHotEncoder(dtype=np.bool_)
        enc.fit(sparse_features_list)
        sparse_features_list = enc.transform(sparse_features_list).toarray()
        np.save(path, sparse_features_list)
        return sparse_features_list

    def get_pos_tag(self, token):
        pos = self._morph.parse(token)[0].tag.POS
        if pos is not None:
            return pos
        else:
            return "none"

    def get_capital(self, token):
        pattern = re.compile("[{}]+$".format(re.escape(string.punctuation)))
        if pattern.match(token):
            return "none"
        if len(token) == 0:
            return "none"
        if token.islower():
            return "lower"
        elif token.isupper():
            return "upper"
        elif token[0].isupper() and len(token) == 1:
            return "proper"
        elif token[0].isupper() and token[1:].islower():
            return "proper"
        else:
            return "camel"

    def get_is_number(self, token):
        try:
            complex(token)
        except ValueError:
            return "no"
        return "yes"

    def get_initial(self, token):
        initial = self._morph.parse(token)[0].normal_form
        if initial is not None:
            return initial
        else:
            return "none"

    def get_is_punct(self, token):
        pattern = re.compile("[{}]+$".format(re.escape(string.punctuation)))
        if pattern.match(token):
            return "yes"
        else:
            return "no"
