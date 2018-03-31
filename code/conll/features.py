from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from scipy.sparse import csr_matrix, hstack, vstack

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


def delete_row_csr(matrix, index):
    if not isinstance(matrix, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = matrix.indptr[index + 1] - matrix.indptr[index]
    if n > 0:
        matrix.data[matrix.indptr[index]:-n] = matrix.data[matrix.indptr[index + 1]:]
        matrix.data = matrix.data[:-n]
        matrix.indices[matrix.indptr[index]:-n] = matrix.indices[matrix.indptr[index + 1]:]
        matrix.indices = matrix.indices[:-n]
    matrix.indptr[index:-1] = matrix.indptr[index + 1:]
    matrix.indptr[index:] -= n
    matrix.indptr = matrix.indptr[:-1]
    matrix._shape = (matrix._shape[0] - 1, matrix._shape[1])


class Generator:
    def __init__(self,
                 columntypes=None,
                 context_len=2,
                 language='ru',
                 rare_count=5,
                 min_weight=0.9,
                 rewrite=False,
                 history=False):
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

        self.column_types_ = columntypes
        self.context_len_ = context_len
        self.rare_count_ = rare_count
        self.total_weight_ = min_weight
        self.lang_ = language
        self.rewrite_data_ = rewrite
        self.history_ = history

        if self.lang_ == 'ru':
            class Parser:
                def __init__(self):
                    self.inner = MorphAnalyzer()

                def initial(self, token):
                    return self.inner.parse(token)[0].normal_form

                def pos(self, token):
                    return self.inner.parse(token)[0].tag.POS

            self.parser_ = Parser()
        else:
            class Parser:
                def __init__(self):
                    self.inner = WordNetLemmatizer()

                def pos(self, token):
                    raise ValueError('У английского POS есть!')

                def initial(self, token):
                    return self.inner.lemmatize(token)

            self.parser_ = Parser()

        self.encoder_, self.binarizer_ = None, None
        self.features_classes_ = None
        self.counter_ = None
        self.features_length_ = None
        self.features_to_keep_ = None
        self.SAMPLE_HISTORY_LINE = None

    def fit_generate(self, x_docs, y_docs, path, clf=ExtraTreesClassifier()):
        if os.path.exists(path) and not self.rewrite_data_:
            sparse_features_list = self.load_csr(path)
            return sparse_features_list

        x_docs_flat, y_docs_flat = self.flatten_docs(x_docs, y_docs)
        y_tokens, x_tokens = [], []
        for y_doc_flat, x_doc_flat in zip(y_docs_flat, x_docs_flat):
            y_tokens += y_doc_flat
            x_tokens += x_doc_flat

        word_index = self.column_types_.index('words')
        words_all = [(el[word_index]) for el in x_tokens]
        initial_all = [self.get_initial(el) for el in words_all]
        self.counter_ = collections.Counter(initial_all)

        features_list = []
        hist_features_list = []

        features_indexes = dict()
        hist_features_indexes = dict()
        index_ = 0

        total_count = 0
        history_count = 0

        first_time = True

        for x_doc, y_doc in zip(x_docs_flat, y_docs_flat):

            x_doc = [['' for _ in range(len(self.column_types_))] for _ in range(self.context_len_)] + x_doc
            x_doc = x_doc + [['' for _ in range(len(self.column_types_))] for _ in range(self.context_len_)]

            word = [(el[word_index]) for el in x_doc]
            initial = [self.get_initial_counted(el) for el in word]
            is_punct = [self.get_is_punct(el) for el in word]
            letters_type = [self.letters_type(el) for el in word]
            is_number = [self.get_is_number(el) for el in word]

            if 'pos' in self.column_types_:
                pos_index = self.column_types_.index('pos')
                pos_tag = [(el[pos_index]) for el in x_doc]
            else:
                pos_tag = [self.get_pos_tag(el[word_index]) for el in x_doc]
            if 'chunk' in self.column_types_:
                chunk_index = self.column_types_.index('chunk')
                chunk_tag = [(el[chunk_index]) for el in x_doc]
            else:
                chunk_index = None

            doc_features_list = []
            doc_hist_features_list = []

            for k in range(len(x_doc) - 2 * self.context_len_):
                line = []
                i = k + self.context_len_

                pos_tag_line = [pos_tag[i]]
                for j in range(1, self.context_len_ + 1):
                    pos_tag_line.append(pos_tag[i - j])
                    pos_tag_line.append(pos_tag[i + j])
                line += pos_tag_line
                if first_time:
                    features_indexes['pos_tag_0'] = [index_]
                    index_ += 1
                    for j in range(1, self.context_len_ + 1):
                        features_indexes[f'pos_tag_{-j}'] = [index_]
                        index_ += 1
                        features_indexes[f'pos_tag_{j}'] = [index_]
                        index_ += 1

                if chunk_index is not None:
                    chunk_tag_line = [chunk_tag[i]]
                    for j in range(1, self.context_len_ + 1):
                        chunk_tag_line.append(chunk_tag[i - j])
                        chunk_tag_line.append(chunk_tag[i + j])
                    line += chunk_tag_line
                    if first_time:
                        features_indexes['chunk_tag_0'] = [index_]
                        index_ += 1
                        for j in range(1, self.context_len_ + 1):
                            features_indexes[f'chunk_tag_{-j}'] = [index_]
                            index_ += 1
                            features_indexes[f'chunk_tag_{j}'] = [index_]
                            index_ += 1

                letters_type_line = [letters_type[i]]
                for j in range(1, self.context_len_ + 1):
                    letters_type_line.append(letters_type[i - j])
                    letters_type_line.append(letters_type[i + j])
                line += letters_type_line
                if first_time:
                    features_indexes['letters_type_0'] = [index_]
                    index_ += 1
                    for j in range(1, self.context_len_ + 1):
                        features_indexes[f'letters_type_{-j}'] = [index_]
                        index_ += 1
                        features_indexes[f'letters_type_{j}'] = [index_]
                        index_ += 1

                is_punct_line = [is_punct[i]]
                for j in range(1, self.context_len_ + 1):
                    is_punct_line.append(is_punct[i - j])
                    is_punct_line.append(is_punct[i + j])
                line += is_punct_line
                if first_time:
                    features_indexes['is_punct_0'] = [index_]
                    index_ += 1
                    for j in range(1, self.context_len_ + 1):
                        features_indexes[f'is_punct_{-j}'] = [index_]
                        index_ += 1
                        features_indexes[f'is_punct_{j}'] = [index_]
                        index_ += 1

                is_number_line = [is_number[i]]
                for j in range(1, self.context_len_ + 1):
                    is_number_line.append(is_number[i - j])
                    is_number_line.append(is_number[i + j])
                line += is_number_line
                if first_time:
                    features_indexes['is_number_0'] = [index_]
                    index_ += 1
                    for j in range(1, self.context_len_ + 1):
                        features_indexes[f'is_number_{-j}'] = [index_]
                        index_ += 1
                        features_indexes[f'is_number_{j}'] = [index_]
                        index_ += 1

                initial_line = [initial[i]]
                for j in range(1, self.context_len_ + 1):
                    initial_line.append(initial[i - j])
                    initial_line.append(initial[i + j])
                line += initial_line
                if first_time:
                    features_indexes['initial_0'] = [index_]
                    index_ += 1
                    for j in range(1, self.context_len_ + 1):
                        features_indexes[f'initial_{-j}'] = [index_]
                        index_ += 1
                        features_indexes[f'initial_{j}'] = [index_]
                        index_ += 1

                if first_time:
                    self.features_length_ = len(line)
                    self.SAMPLE_HISTORY_LINE = np.array(['no_history' for _ in range(self.features_length_)])

                if self.history_:
                    for a in range(i - 1, self.context_len_ if i - 1000 < self.context_len_ else i - 1000, -1):
                        if initial[a] == initial[i] and is_punct[i] == 'false' and is_number[i] == 'false':
                            history_line_full = doc_features_list[a - self.context_len_]
                            history_line = history_line_full[:self.features_length_]
                            history_count += 1
                            break
                    else:
                        history_line = self.SAMPLE_HISTORY_LINE

                    if first_time:
                        for feature_class, indexes in features_indexes.items():
                            hist_features_indexes['history_' + feature_class] = [index for index in indexes]

                    doc_hist_features_list.append(history_line)

                doc_features_list.append(np.array(line))
                total_count += 1
                first_time = False

            features_list += doc_features_list
            if self.history_:
                hist_features_list += doc_hist_features_list

        # add SAMPLE_HISTORY_LINE to fit binarizer and encoder
        features_list.append(self.SAMPLE_HISTORY_LINE)

        hist_features_list = np.array(hist_features_list)
        features_list = np.array(features_list)

        print(f'Признаков в исходном виде: {self.features_length_}')
        print(f'Обработано {total_count} токенов!')
        if self.history_:
            print(f'{history_count} токенов имеют историю в рамках документа!')

        label_encoders = {i: LabelEncoder() for i in range(self.features_length_)}
        self.binarizer_ = LaberEncoderWide(label_encoders)
        features_list = self.binarizer_.fit(features_list).transform(features_list)

        self.features_classes_ = []
        for one_binarizer_classes in self.binarizer_.classes_:
            for class_name in one_binarizer_classes:
                self.features_classes_.append(class_name)

        features_classes_len = len(self.features_classes_)

        # duplicate for history features classes
        self.features_classes_ = self.features_classes_ + self.features_classes_

        self.encoder_ = OneHotEncoder(dtype=np.int8, sparse=True)
        features_list = self.encoder_.fit(features_list).transform(features_list)

        features_indexes_encoded = dict()
        for key, value in features_indexes.items():
            indexes_encoded = []
            for index in value:
                indexes_encoded += list(range(self.encoder_.feature_indices_[index],
                                              self.encoder_.feature_indices_[index + 1], 1))
            features_indexes_encoded[key] = indexes_encoded

        # features_indexes contains information without one-hot encoding:
        # features_indexes['initial_0'] = [1] # only one index
        # features_indexes_encoded contains information about indexes after one-hot encoding:
        # features_indexes_encoded['initial_0'] = [1, 2, 3, 4 ...] # all indexes that were created for this feature

        for key, values in features_indexes_encoded.items():
            for index in values:
                self.features_classes_[index] = key + ' - ' + self.features_classes_[index]

        # remove SAMPLE_HISTORY_LINE
        encoded_features_len = features_list.shape[0]
        delete_row_csr(features_list, encoded_features_len - 1)

        # hist_features_list
        if self.history_:
            hist_features_list = self.binarizer_.transform(hist_features_list)
            hist_features_list = self.encoder_.transform(hist_features_list)

            hist_features_indexes_encoded = dict()
            for key, value in hist_features_indexes.items():
                indexes_encoded = []
                for index in value:
                    indexes_encoded += list(range(self.encoder_.feature_indices_[index] + features_classes_len,
                                                  self.encoder_.feature_indices_[index + 1] + features_classes_len, 1))
                hist_features_indexes_encoded[key] = indexes_encoded

            for key, values in hist_features_indexes_encoded.items():
                for index in values:
                    self.features_classes_[index] = key + ' - ' + self.features_classes_[index]

            features_list = hstack([features_list, hist_features_list])
            features_list = csr_matrix(features_list)

        clf.fit(features_list, y_tokens)
        features_weight = sorted([(i, el) for i, el in enumerate(clf.feature_importances_)],
                                 key=lambda el: -el[1])
        current_weight = 0.0
        self.features_to_keep_ = []
        for el in features_weight:
            self.features_to_keep_.append(el[0])
            current_weight += el[1]
            if current_weight > self.total_weight_:
                break
        features_list = features_list[:, self.features_to_keep_]

        self.features_classes_ = np.take(np.array(self.features_classes_),
                                         self.features_to_keep_)

        print("Признаков осталось: " + str(len(self.features_to_keep_)))

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
        if os.path.exists(path) and not self.rewrite_data_:
            sparse_features_list = self.load_csr(path)
            return sparse_features_list

        big_x_docs = []
        for x_doc in x_docs:
            big_x_doc = []
            for x_sent in x_doc:
                big_x_doc += x_sent
            big_x_docs.append(big_x_doc)
        x_docs = big_x_docs

        features_list = []
        hist_features_list = []

        for x_doc in x_docs:

            x_doc = [["" for _ in range(len(self.column_types_))] for _ in range(self.context_len_)] + x_doc
            x_doc = x_doc + [["" for _ in range(len(self.column_types_))] for _ in range(self.context_len_)]

            word_index = self.column_types_.index("words")
            word = [(el[word_index]) for el in x_doc]
            initial = [self.get_initial_counted(el) for el in word]
            is_punct = [self.get_is_punct(el) for el in word]
            letters_type = [self.letters_type(el) for el in word]
            is_number = [self.get_is_number(el) for el in word]

            if 'pos' in self.column_types_:
                pos_index = self.column_types_.index('pos')
                pos_tag = [(el[pos_index]) for el in x_doc]
            else:
                pos_tag = [self.get_pos_tag(el[word_index]) for el in x_doc]
            if 'chunk' in self.column_types_:
                chunk_index = self.column_types_.index('chunk')
                chunk_tag = [(el[chunk_index]) for el in x_doc]
            else:
                chunk_index = None

            doc_features_list = []
            doc_hist_features_list = []

            for k in range(len(x_doc) - 2 * self.context_len_):
                line = []
                i = k + self.context_len_

                pos_tag_line = [pos_tag[i]]
                for j in range(1, self.context_len_ + 1):
                    pos_tag_line.append(pos_tag[i - j])
                    pos_tag_line.append(pos_tag[i + j])
                line += pos_tag_line

                if chunk_index is not None:
                    chunk_tag_line = [chunk_tag[i]]
                    for j in range(1, self.context_len_ + 1):
                        chunk_tag_line.append(chunk_tag[i - j])
                        chunk_tag_line.append(chunk_tag[i + j])
                    line += chunk_tag_line

                letters_type_line = [letters_type[i]]
                for j in range(1, self.context_len_ + 1):
                    letters_type_line.append(letters_type[i - j])
                    letters_type_line.append(letters_type[i + j])
                line += letters_type_line

                is_punct_line = [is_punct[i]]
                for j in range(1, self.context_len_ + 1):
                    is_punct_line.append(is_punct[i - j])
                    is_punct_line.append(is_punct[i + j])
                line += is_punct_line

                is_number_line = [is_number[i]]
                for j in range(1, self.context_len_ + 1):
                    is_number_line.append(is_number[i - j])
                    is_number_line.append(is_number[i + j])
                line += is_number_line

                initial_line = [initial[i]]
                for j in range(1, self.context_len_ + 1):
                    initial_line.append(initial[i - j])
                    initial_line.append(initial[i + j])
                line += initial_line

                length_without_history = len(line)

                if self.history_:
                    history_line = None
                    for a in range(i - 1, self.context_len_ if i - 1000 < self.context_len_ else i - 1000, -1):
                        if initial[a] == initial[i] and is_punct[i] == 'false' and is_number[i] == 'false':
                            history_line_full = doc_features_list[a - self.context_len_]
                            history_line = history_line_full[:length_without_history]
                            break
                    if history_line is None:
                        history_line = self.SAMPLE_HISTORY_LINE

                    doc_hist_features_list.append(np.array(history_line))

                doc_features_list.append(np.array(line))

            features_list += doc_features_list
            if self.history_:
                hist_features_list += doc_hist_features_list

        features_list = np.array(features_list)
        hist_features_list = np.array(hist_features_list)

        features_list = self.binarizer_.transform(features_list)
        features_list = self.encoder_.transform(features_list)

        if self.history_:
            hist_features_list = self.binarizer_.transform(hist_features_list)
            hist_features_list = self.encoder_.transform(hist_features_list)
            features_list = hstack([features_list, hist_features_list])
            features_list = csr_matrix(features_list)

        features_list = features_list[:, self.features_to_keep_]

        self.save_csr(path, features_list)
        return features_list

    def get_pos_tag(self, token):
        return self.parser_.pos(token)

    @staticmethod
    def letters_type(token):
        if PUNCT_PATTERN.match(token):
            return 'punct'
        if len(token) == 0:
            return 'empty'
        if token.islower():
            return 'lower'
        elif token.isupper():
            return 'upper'
        elif token[0].isupper() and len(token) == 1:
            return 'upper_one'
        elif token[0].isupper() and token[1:].islower():
            return 'upper_first'
        else:
            return 'other'

    @staticmethod
    def get_is_number(token):
        try:
            complex(token)
        except ValueError:
            return 'true'
        return 'false'

    def get_initial(self, token):
        result = self.parser_.initial(token)
        return result if result is not None else 'bad_initial_token'

    def get_initial_counted(self, token):
        result = self.parser_.initial(token)
        if result is not None and self.counter_[result] > self.rare_count_:
            return result
        else:
            return 'rare_initial_token'

    @staticmethod
    def get_is_punct(token):
        if PUNCT_PATTERN.match(token):
            return 'true'
        else:
            return 'false'

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


class LaberEncoderWide(object):
    def __init__(self, column_stages):
        self.column_stages_ = column_stages
        self.classes_ = []

    def fit(self, x):
        x = x.copy()
        for i, k in self.column_stages_.items():
            k.fit(x[:, i])
            self.classes_.append(k.classes_)
        return self

    def transform(self, x):
        x = x.copy()
        for i, k in self.column_stages_.items():
            x[:, i] = k.transform(x[:, i])
        return x
