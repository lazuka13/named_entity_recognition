from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from scipy.sparse import csr_matrix

from nltk.stem import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer

import numpy as np

import collections
import logging
import string
import re
import os

logger = logging.getLogger('logger')

PUNCT_PATTERN = re.compile("[{}]+$".format(re.escape(string.punctuation)))


class Generator:
    def __init__(self, columntypes=None, context_len=2, language='ru',
                 rare_count=5, min_weight=0.9, rewrite=False, history=False):
        """
        Конструктор генератора признаков
        :param columntypes: типы столбцов в верной посл-ти, например ['words', 'pos', 'chunk']
        :param context_len: длина учитываемого контекста
        :param language: язык датасета
        :param rare_count: минимально допустимая частота вхождения метки
        :param min_weight: процент веса признаков, которые оставляем
        :param rewrite: перезаписывать ли файл с признаками
        :param history: использовать ли историю предыдущих вхождений
        """
        logger.info(f'Параметры генерации признаков: context_len = {context_len}, language = {language}, '
                    f'rare_count = {rare_count}, min_weight = {min_weight}, history = {history}')

        self._column_types = columntypes
        self._context_len = context_len
        self._rare_count = rare_count
        self._min_weight = min_weight
        self._language = language
        self._rewrite = rewrite
        self._history = history

        if self._language == 'ru':
            class Parser:
                def __init__(self):
                    self.inner = MorphAnalyzer()

                def initial(self, token):
                    return self.inner.parse(token)[0].normal_form

                def pos(self, token):
                    return self.inner.parse(token)[0].tag.POS

            self.parser = Parser()
        else:
            class Parser:
                def __init__(self):
                    self.inner = WordNetLemmatizer()

                def pos(self, token):
                    raise ValueError('У английского POS есть!')

                def initial(self, token):
                    return self.inner.lemmatize(token)

            self.parser = Parser()

        self._encoder, self._binarizer = None, None
        self._counters = []
        self.counter = None

        self._number_of_columns = 0
        self._columns_to_keep = 0

    def fit_generate(self, x_docs, y_docs, path, clf=ExtraTreesClassifier()):
        """
        Отвечает за генерацию признаков и настройку генератора для отбора верных признаков
        Принимает данные в формате документов, возвращает признаки в таком же формате
        (побитыми на документы)
        :param x_docs: данные для генерации признаков
        :param y_docs: правильные ответы на данных
        :param path: путь к файлу для загрузки/сохранения созданных данных
        :param clf: классификатор для отбора признаков по их весу
        :return:
        """

        if os.path.exists(path) and not self._rewrite:
            logger.debug('Загружаем разреженную матрицу признаков!')
            sparse_features_list = self.load_csr(path)
            return sparse_features_list

        x_docs_flat, y_docs_flat = self.flatten_docs(x_docs, y_docs)  # представление без предложений
        y_tokens, x_tokens = [], []
        for y_doc_flat, x_doc_flat in zip(y_docs_flat, x_docs_flat):
            y_tokens += y_doc_flat
            x_tokens += x_doc_flat

        # посчитаем частоту признаков
        logger.debug('Подсчет числа признаков, удаление редких признаков')
        word_index = self._column_types.index("words")
        words_all = [(el[word_index]) for el in x_tokens]
        initial_all = [self.get_initial(el) for el in words_all]
        self.counter = collections.Counter(initial_all)

        features_list = []
        features_indexes = dict()
        index_ = 0

        total_count = 0
        history_count = 0

        first_time = True
        for x_doc, y_doc in zip(x_docs_flat, y_docs_flat):

            x_doc = [['' for _ in range(len(self._column_types))] for _ in range(self._context_len)] + x_doc
            x_doc = x_doc + [['' for _ in range(len(self._column_types))] for _ in range(self._context_len)]

            word = [(el[word_index]) for el in x_doc]
            initial = [self.get_initial_counted(el) for el in word]
            is_punct = [self.get_is_punct(el) for el in word]
            letters_type = [self.letters_type(el) for el in word]
            is_number = [self.get_is_number(el) for el in word]
            if 'pos' in self._column_types:
                pos_index = self._column_types.index('pos')
                pos_tag = [(el[pos_index]) for el in x_doc]
            else:
                pos_tag = [self.get_pos_tag(el[word_index]) for el in x_doc]
            if 'chunk' in self._column_types:
                chunk_index = self._column_types.index('chunk')
                chunk_tag = [(el[chunk_index]) for el in x_doc]
            else:
                chunk_index = None

            doc_features_list = []

            for k in range(len(x_doc) - 2 * self._context_len):
                line = []
                i = k + self._context_len

                pos_tag_line = [pos_tag[i]]
                for j in range(1, self._context_len + 1):
                    pos_tag_line.append(pos_tag[i - j])
                    pos_tag_line.append(pos_tag[i + j])
                line += pos_tag_line
                if first_time:
                    features_indexes['pos_tag'] = [g for g in range(index_, index_ + len(pos_tag_line), 1)]
                    index_ += len(pos_tag_line)

                if chunk_index is not None:
                    chunk_tag_line = [chunk_tag[i]]
                    for j in range(1, self._context_len + 1):
                        chunk_tag_line.append(chunk_tag[i - j])
                        chunk_tag_line.append(chunk_tag[i + j])
                    line += chunk_tag_line
                    if first_time:
                        features_indexes['chunk_tag'] = [g for g in range(index_, index_ + len(chunk_tag_line), 1)]
                        index_ += len(chunk_tag_line)

                letters_type_line = [letters_type[i]]
                for j in range(1, self._context_len + 1):
                    letters_type_line.append(letters_type[i - j])
                    letters_type_line.append(letters_type[i + j])
                line += letters_type_line
                if first_time:
                    features_indexes['letters_type'] = [g for g in range(index_, index_ + len(letters_type_line), 1)]
                    index_ += len(letters_type_line)

                is_punct_line = [is_punct[i]]
                for j in range(1, self._context_len + 1):
                    is_punct_line.append(is_punct[i - j])
                    is_punct_line.append(is_punct[i + j])
                line += is_punct_line
                if first_time:
                    features_indexes['is_punct'] = [g for g in range(index_, index_ + len(is_punct_line), 1)]
                    index_ += len(is_punct_line)

                is_number_line = [is_number[i]]
                for j in range(1, self._context_len + 1):
                    is_number_line.append(is_number[i - j])
                    is_number_line.append(is_number[i + j])
                line += is_number_line
                if first_time:
                    features_indexes['is_number'] = [g for g in range(index_, index_ + len(is_number_line), 1)]
                    index_ += len(is_number_line)

                initial_line = [initial[i]]
                for j in range(1, self._context_len + 1):
                    initial_line.append(initial[i - j])
                    initial_line.append(initial[i + j])
                line += initial_line
                if first_time:
                    features_indexes['initial'] = [g for g in range(index_, index_ + len(initial_line), 1)]
                    index_ += len(initial_line)

                length_without_history = len(line)
                line = np.array(line)

                if self._history:
                    history_line = None
                    for a in range(i - 1, self._context_len if i - 1000 < self._context_len else i - 1000, -1):
                        if initial[a] == initial[i] and is_punct[i] == 1 and is_number[i] == 1:
                            history_line_full = doc_features_list[a - self._context_len]
                            history_line = history_line_full[:length_without_history]
                            break
                    if history_line is None:
                        history_line = np.zeros(length_without_history)
                    else:
                        history_count += 1
                    if first_time:
                        features_indexes['history'] = [g for g in range(index_, index_ + len(history_line), 1)]
                        index_ += len(history_line)

                    line = np.append(line, history_line)

                doc_features_list.append(line)
                total_count += 1
                first_time = False
            features_list += doc_features_list


        features_list = np.array(features_list)

        logger.debug(f'Генерация исходных признаков завершена!')
        logger.debug(f'Для каждого токена создано {features_list.shape[1]} признаков!')
        logger.debug(f'Обработано {total_count} токенов!')
        logger.debug(f'{history_count} токенов имеют историю в рамках документа!')

        # Обработка признаков LabelEncoder-ом
        self._binarizer = ColumnApplier({i: LabelEncoder() for i in range(features_list.shape[1])})
        features_list = self._binarizer.fit(features_list).transform(features_list)

        # Применяем OneHotEncoder к признакам
        self._encoder = OneHotEncoder(dtype=np.int8, sparse=True)
        features_list = self._encoder.fit(features_list).transform(features_list)

        features_indexes_encoded = dict()
        for key, value in features_indexes.items():
            indexes_encoded = []
            for index in value:
                indexes_encoded += list(range(self._encoder.feature_indices_[index],
                                              self._encoder.feature_indices_[index + 1] + 1, 1))
            features_indexes_encoded[key] = indexes_encoded

        logger.debug(f'Удаление неинформативных признаков! Оставляемый вес признаков '
                     f'(по feature_importances) - {self._min_weight}!')
        clf.fit(features_list, y_tokens)
        features_weight = sorted([(i, el) for i, el in enumerate(clf.feature_importances_)],
                                 key=lambda el: -el[1])
        current_weight = 0.0
        self._columns_to_keep = []
        for el in features_weight:
            self._columns_to_keep.append(el[0])
            current_weight += el[1]
            if current_weight > self._min_weight:
                break
        features_list = features_list[:, self._columns_to_keep]

        logger.debug('После удаления неинформативных признаков следующее число признаков осталось:')
        for key, value in features_indexes_encoded.items():
            indexes = []
            for index in value:
                if index in self._columns_to_keep:
                    indexes.append(index)
            features_indexes_encoded[key] = indexes
            logger.debug(f'В категории {key} осталось {len(indexes)} признаков, было {len(value)} признаков!')
            if len(indexes) == 0:
                continue
            weights = []
            for index in indexes:
                weights.append(clf.feature_importances_[index])
            logger.debug(f'Общий вес категории - {sum(weights)}, минимальный вес - {min(weights)}, '
                         f'максимальный вес - {max(weights)}')

        logger.debug(f'Всего осталось признаков - {len(self._columns_to_keep)}!')

        logger.debug('Сохраняем разреженную матрицу признаков!')
        self.save_csr(path, features_list)
        return features_list

    @staticmethod
    def flatten_docs(x_docs, y_docs):
        x_docs_flat, y_docs_flat = [], []
        for x_doc, y_doc in zip(x_docs, y_docs):
            x_doc_flat, y_doc_flat = [], []
            for x_sent, y_sent in zip(x_doc, y_doc):
                x_doc_flat += x_sent
                y_doc_flat += y_sent
            x_docs_flat.append(x_doc_flat)
            y_docs_flat.append(y_doc_flat)
        return x_docs_flat, y_docs_flat

    def generate(self, x_docs, path):
        """
        Отвечает за генерацию признаков обученным генератором
        :param x_docs: данные для генерации признаков
        :param path: путь к файлу для загрузк/сохранения созданных данных
        :return:
        """

        if os.path.exists(path) and not self._rewrite:
            logger.debug('Загружаем разреженную матрицу признаков!')
            sparse_features_list = self.load_csr(path)
            return sparse_features_list

        big_x_docs = []
        for x_doc in x_docs:
            big_x_doc = []
            for x_sent in x_doc:
                big_x_doc += x_sent
            big_x_docs.append(big_x_doc)
        x_docs = big_x_docs

        logger.debug('Приступаем к созданию признаков!')
        features_list = []
        for x_doc in x_docs:

            x_doc = [["" for _ in range(len(self._column_types))] for _ in range(self._context_len)] + x_doc
            x_doc = x_doc + [["" for _ in range(len(self._column_types))] for _ in range(self._context_len)]

            logger.debug('Рассчитываем значения для всех элементов датасета')
            word_index = self._column_types.index("words")
            word = [(el[word_index]) for el in x_doc]
            initial = [self.get_initial(el) for el in word]
            is_punct = [self.get_is_punct(el) for el in word]
            letters_type = [self.letters_type(el) for el in word]
            is_number = [self.get_is_number(el) for el in word]
            if 'pos' in self._column_types:
                pos_index = self._column_types.index('pos')
                pos_tag = [(el[pos_index]) for el in x_doc]
            else:
                pos_tag = [self.get_pos_tag(el[word_index]) for el in x_doc]
            if 'chunk' in self._column_types:
                chunk_index = self._column_types.index('chunk')
                chunk_tag = [(el[chunk_index]) for el in x_doc]
            else:
                chunk_index = None

            logger.debug('Учет контекста!')
            doc_features_list = []

            for k in range(len(x_doc) - 2 * self._context_len):
                line = []
                i = k + self._context_len

                pos_tag_line = [pos_tag[i]]
                for j in range(1, self._context_len + 1):
                    pos_tag_line.append(pos_tag[i - j])
                    pos_tag_line.append(pos_tag[i + j])
                line += pos_tag_line

                if chunk_index is not None:
                    chunk_tag_line = [chunk_tag[i]]
                    for j in range(1, self._context_len + 1):
                        chunk_tag_line.append(chunk_tag[i - j])
                        chunk_tag_line.append(chunk_tag[i + j])
                    line += chunk_tag_line

                letters_type_line = [letters_type[i]]
                for j in range(1, self._context_len + 1):
                    letters_type_line.append(letters_type[i - j])
                    letters_type_line.append(letters_type[i + j])
                line += letters_type_line

                is_punct_line = [is_punct[i]]
                for j in range(1, self._context_len + 1):
                    is_punct_line.append(is_punct[i - j])
                    is_punct_line.append(is_punct[i + j])
                line += is_punct_line

                is_number_line = [is_number[i]]
                for j in range(1, self._context_len + 1):
                    is_number_line.append(is_number[i - j])
                    is_number_line.append(is_number[i + j])
                line += is_number_line

                initial_line = [initial[i]]
                for j in range(1, self._context_len + 1):
                    initial_line.append(initial[i - j])
                    initial_line.append(initial[i + j])
                line += initial_line

                length_without_history = len(line)
                line = np.array(line)

                if self._history:
                    history_line = None
                    for a in range(i - 2, -1 if i - 1001 < -1 else i - 1001, -1):
                        if initial[a] == initial[i] and is_punct[i] == 0 and is_number[i] == 0:
                            history_line = features_list[a - self._context_len][:length_without_history]
                            break
                    if history_line is None:
                        history_line = np.zeros(length_without_history)
                    line = np.append(line, history_line)

                doc_features_list.append(line)
            features_list += doc_features_list

        features_list = np.array(features_list)

        logger.debug(f'Удаление редких значений! Граница - {self._rare_count}!')
        self._number_of_columns = features_list.shape[1]
        for s in range(len(features_list)):
            for m in range(self._number_of_columns):
                features_list[s][m] = self.get_feature(m, features_list[s][m])

        logger.debug(f'Бинаризация оставшихся признаков!')
        features_list = self._binarizer.transform(features_list)
        features_list = self._encoder.transform(features_list)

        logger.debug(f'Удаление слабо информативных признаков! Граница - {self._min_weight}!')
        features_list = features_list[:, self._columns_to_keep]

        logger.debug('Сохраняем разреженную матрицу признаков!')
        self.save_csr(path, features_list)

        return features_list

    def get_pos_tag(self, token):
        return self.parser.pos(token)

    @staticmethod
    def letters_type(token):
        if PUNCT_PATTERN.match(token):
            return np.int8(1)
        if len(token) == 0:
            return np.int8(1)
        if token.islower():
            return np.int8(2)
        elif token.isupper():
            return np.int8(3)
        elif token[0].isupper() and len(token) == 1:
            return np.int8(4)
        elif token[0].isupper() and token[1:].islower():
            return np.int8(5)
        else:
            return np.int8(6)

    @staticmethod
    def get_is_number(token):
        try:
            complex(token)
        except ValueError:
            return np.int8(1)
        return np.int8(1)

    def get_initial(self, token):
        result = self.parser.initial(token)
        return result if result is not None else np.int8(0)

    def get_initial_counted(self, token):
        result = self.parser.initial(token)
        if result is not None and self.counter[result] > self._rare_count:
            return result
        else:
            return np.int8(0)

    @staticmethod
    def get_is_punct(token):
        if PUNCT_PATTERN.match(token):
            return np.int8(2)
        else:
            return np.int8(1)

    @staticmethod
    def save_csr(filename, array):
        np.savez(filename,
                 data=array.data,
                 indices=array.indices,
                 indptr=array.indptr,
                 shape=array.shape)

    @staticmethod
    def load_csr(filename):
        loader = np.load(filename)
        return csr_matrix((loader['data'],
                           loader['indices'],
                           loader['indptr']),
                          shape=loader['shape'])

    def get_feature(self, f, feature):
        """
        Отвечает за избавление от редких признаков
        :param f:
        :param feature:
        :return:
        """
        if feature in self._counters[f].keys() and \
                self._counters[f][feature] > self._rare_count:
            return feature
        else:
            return -1


class ColumnApplier(object):
    def __init__(self, column_stages):
        self._column_stages = column_stages

    def fit(self, x):
        for i, k in self._column_stages.items():
            k.fit(x[:, i])
        return self

    def transform(self, x):
        x = x.copy()
        for i, k in self._column_stages.items():
            x[:, i] = k.transform(x[:, i])
        return x
