import pymorphy2
import string
import re
import os

from sklearn import preprocessing
from scipy.sparse import csr_matrix
from sklearn.ensemble import ExtraTreesClassifier
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter


class Generator:

    def __init__(self,
                 column_types=None,
                 context_len=2,
                 language='ru',
                 number_of_occurences=5,
                 weight_percentage=0.9):

        # Частота, ниже которой лейбл считается "редким" #
        self.NUMBER_OF_OCCURENCES = number_of_occurences

        # Процент веса признаков, который нужно оставить
        self.WEIGHT_PERCENTAGE = weight_percentage  #

        # Информация о подаваемых столбцах (может быть WORD, POS, CHUNK) #
        self._column_types = column_types if column_types is not None else ["WORD"]

        # Длина рассматриваемого контекста (context_len влево и context_len вправо) #
        self._context_len = context_len

        # Анализатор (для POS-тега и начальной формы) #
        self._morph = pymorphy2.MorphAnalyzer()
        self._lemmatizer = WordNetLemmatizer()

        # Язык датасета (определяет используемые модули) #
        self._lang = language

        # OneHotEncoder, хранится после FIT-а #
        self._enc = None

        # ColumnApplier, хранится после FIT-а #
        self._multi_encoder = None

        # Словари распознаваемых слов, хранятся после FIT-а #
        self._counters = []

        # Число столбцов в "сырой" матрице признаков #
        self._number_of_columns = None

        # Индексы столбцов признаков, оставленных после отсева #
        self._columns_to_keep = None

    def fit_transform(self, data, answers, path, clf=ExtraTreesClassifier()):

        # Eсли данные сохранены - просто берем их из файла #
        if os.path.exists(path):
            sparse_features_list = self.load_sparse_csr(path)
            return sparse_features_list

        # Добавляем пустые "слова" в начало и конец (для контекста) #
        data = [["" for i in range(len(self._column_types))] for i in range(self._context_len)] + data
        data = data + [["" for i in range(len(self._column_types))] for i in range(self._context_len)]

        # Находим индексы столбцов в переданных данных #
        word_index = self._column_types.index("WORD")
        if "POS" in self._column_types:
            pos_index = self._column_types.index("POS")
        else:
            pos_index = None
        if "POS" in self._column_types:
            chunk_index = self._column_types.index("CHUNK")
        else:
            chunk_index = None

        # Список признаков (строка == набор признаков для слова из массива data) #
        features_list = []

        # Заполнение массива features_list "сырыми" данными (без отсева) #
        for k in range(len(data) - 2 * self._context_len):
            arr = []
            i = k + self._context_len

            if pos_index is not None:
                pos_arr = [data[i][pos_index]]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(data[i - j][pos_index])
                    pos_arr.append(data[i + j][pos_index])
            else:
                pos_arr = [self.get_pos_tag(data[i][word_index])]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(self.get_pos_tag(data[i - j][word_index]))
                    pos_arr.append(self.get_pos_tag(data[i + j][word_index]))
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

        # Теперь это массив сырых признаков (в строковом представлении, без отсева) #
        features_list = np.array([np.array(line) for line in features_list])

        # Выкинем из этого массива классы, встретившиеся менее NUMBER_OF_OCCURENCES раз #
        # Посчитаем частоту лейблов в столбце #
        self._number_of_columns = features_list.shape[1]
        for u in range(self._number_of_columns):
            arr = features_list[:, u]
            counter = Counter(arr)
            self._counters.append(counter)

        # Избавимся от редких лейблов (частота < NUMBER_OF_OCC) #
        for y in range(len(features_list)):
            for x in range(self._number_of_columns):
                features_list[y][x] = self.get_feature(x, features_list[y][x])

        # Оставшиеся признаки бинаризуем #
        self._multi_encoder = ColumnApplier(
            dict([(i, preprocessing.LabelEncoder()) for i in range(len(features_list[0]))]))
        features_list = self._multi_encoder.fit(features_list, None).transform(features_list)
        self._enc = preprocessing.OneHotEncoder(dtype=np.bool_, sparse=True)
        self._enc.fit(features_list)
        features_list = self._enc.transform(features_list)

        # Избавляемся от неинформативных признаков (WEIGHT = WEIGHT_PERC * TOTAL_WEIGHT)#
        clf.fit(features_list, answers)
        features_importances = [(i, el) for i, el in enumerate(clf.feature_importances_)]

        features_importances = sorted(features_importances, key=lambda el: -el[1])
        current_weight = 0.0
        self._columns_to_keep = []
        for el in features_importances:
            self._columns_to_keep.append(el[0])
            current_weight += el[1]
            if current_weight > self.WEIGHT_PERCENTAGE:
                break

        features_list = features_list[:, self._columns_to_keep]

        # Сохраняем матрицу в файл #
        self.save_sparse_csr(path, features_list)

        # Возвращаем матрицу #
        return features_list

    def transform(self, data, path):

        # Eсли данные сохранены - просто берем их из файла #
        if os.path.exists(path):
            sparse_features_list = self.load_sparse_csr(path)
            return sparse_features_list

        # Добавляем пустые "слова" в начало и конец (для контекста) #
        data = [["" for i in range(len(self._column_types))] for i in range(self._context_len)] + data
        data = data + [["" for i in range(len(self._column_types))] for i in range(self._context_len)]

        # Находим индексы столбцов в переданных данных #
        word_index = self._column_types.index("WORD")
        if "POS" in self._column_types:
            pos_index = self._column_types.index("POS")
        else:
            pos_index = None
        if "CHUNK" in self._column_types:
            chunk_index = self._column_types.index("CHUNK")
        else:
            chunk_index = None

        # Список признаков (строка == набор признаков для слова из массива data) #
        features_list = []

        # Заполнение массива features_list "сырыми" данными (без отсева) #
        for k in range(len(data) - 2 * self._context_len):
            arr = []
            i = k + self._context_len

            if pos_index is not None:
                pos_arr = [data[i][pos_index]]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(data[i - j][pos_index])
                    pos_arr.append(data[i + j][pos_index])
            else:
                pos_arr = [self.get_pos_tag(data[i][word_index])]
                for j in range(1, self._context_len + 1):
                    pos_arr.append(self.get_pos_tag(data[i - j][word_index]))
                    pos_arr.append(self.get_pos_tag(data[i + j][word_index]))
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

        # Теперь это массив сырых признаков (в строковом представлении, без отсева) #
        features_list = np.array([np.array(line) for line in features_list])

        # Выкинем из этого массива классы, встретившиеся менее NUMBER_OF_OCCURENCES раз #
        self._number_of_columns = features_list.shape[1]
        for y in range(len(features_list)):
            for x in range(self._number_of_columns):
                features_list[y][x] = self.get_feature(x, features_list[y][x])

        # Оставшиеся признаки бинаризуем #
        features_list = self._multi_encoder.transform(features_list)
        features_list = self._enc.transform(features_list)

        # Избавляемся от неинформативных признаков (WEIGHT = WEIGHT_PERC * TOTAL_WEIGHT)#
        features_list = features_list[:, self._columns_to_keep]

        # Сохраняем матрицу в файл #
        self.save_sparse_csr(path, features_list)

        # Возвращаем матрицу #
        return features_list

    # Заменяет лейбл на "*", если он "редкий" #
    def get_feature(self, f, feature):
        if feature in self._counters[f].keys() and self._counters[f][feature] > self.NUMBER_OF_OCCURENCES:
            return feature
        else:
            return "*"

    # Сохраняет матрицу в файл #
    def save_sparse_csr(self, filename, array):
        np.savez(filename,
                 data=array.data,
                 indices=array.indices,
                 indptr=array.indptr,
                 shape=array.shape)

    # Загружает матрицу из файла #
    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return csr_matrix((loader['data'],
                           loader['indices'],
                           loader['indptr']),
                          shape=loader['shape'])

    # Возвращает POS-тег слова #
    def get_pos_tag(self, token):
        if self._lang == 'ru':
            pos = self._morph.parse(token)[0].tag.POS
        else:
            pos = None
        if pos is not None:
            return pos
        else:
            return "none"

    # Возвращает тип регистра слова #
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

    # Признак того, является ли слово числом #
    def get_is_number(self, token):
        try:
            complex(token)
        except ValueError:
            return "no"
        return "yes"

    # Возвращает начальную форму слова #
    def get_initial(self, token):
        if self._lang == 'ru':
            initial = self._morph.parse(token)[0].normal_form
        else:
            initial = self._lemmatizer.lemmatize(token)

        if initial is not None:
            return initial
        else:
            return "none"

    # Признак того, является ли слово пунктуацией #
    def get_is_punct(self, token):
        pattern = re.compile("[{}]+$".format(re.escape(string.punctuation)))
        if pattern.match(token):
            return "yes"
        else:
            return "no"


# Переводит категории в числовое представление #
class ColumnApplier(object):
    def __init__(self, column_stages):
        self._column_stages = column_stages

    def fit(self, x, y):
        for i, k in self._column_stages.items():
            k.fit(x[:, i])
        return self

    def transform(self, x):
        x = x.copy()
        for i, k in self._column_stages.items():
            x[:, i] = k.transform(x[:, i])
        return x
