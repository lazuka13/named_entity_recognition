from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

import logging

import features
import reader

from classifiers.token_classifier import TokenClassifier

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s',
                              datefmt='%H:%M:%S')

ch.setFormatter(formatter)

logger.addHandler(ch)


def run_best_features_weight():
    possible_weights = [i / 100 for i in range(80, 90, 1)] + \
                       [i / 200 for i in range(180, 201, 1)]
    results = []

    for possible_weight in possible_weights:
        dataset = reader.DataReader('../dataset', fileids='eng.train.txt',
                                    columntypes=('words', 'pos', 'chunk', 'ne'))
        gen = features.Generator(columntypes=('words', 'pos', 'chunk'), context_len=2, language='en',
                                 rare_count=5, min_weight=possible_weight, rewrite=True, history=True)

        y = [el[1] for el in dataset.get_ne()]
        x = dataset.get_tags(tags=['words', 'pos', 'chunk'])

        x_sent, y_sent = [], []
        index = 0
        for sent in dataset.sents():
            length = len(sent)
            if length == 0:
                continue
            x_sent.append(x[index:index + length])
            y_sent.append(y[index:index + length])
            index += length
        x_docs, y_docs = [], []
        index = 0
        for doc in dataset.docs():
            length = len(doc)
            if length == 0:
                continue
            x_docs.append(x_sent[index:index + length])
            y_docs.append(y_sent[index:index + length])
            index += length

        x = gen.fit_generate(x_docs, y_docs, "../prepared_data/conll_trainset.npz")
        x_sent = []
        index = 0
        for sent in dataset.sents():
            length = len(sent)
            if length == 0:
                continue
            x_sent.append(x[index:index + length])
            index += length

        x_docs = []
        index = 0
        for doc in dataset.docs():
            length = len(doc)
            if length == 0:
                continue
            x_docs.append(x_sent[index:index + length])
            index += length

        clf = TokenClassifier(cls='XGBClassifier')
        clf.set_params(booster='gbtree', colsample_bylevel=0.5, colsample_bytree=0.5, learning_rate=0.3, max_depth=14)
        results.append(np.mean(cross_val_score(clf, x_docs, y_docs, n_jobs=-1)))
    return possible_weights, results


if __name__ == '__main__':
    possible_weights, results = run_best_features_weight()
    print('possible_weights', possible_weights)
    print('results', results)
    plt.plot(possible_weights, results)
    plt.savefig('./graph.png')
