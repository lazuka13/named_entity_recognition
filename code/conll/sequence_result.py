import datetime as dt
import logging
import os

import features
import reader
from classifiers.sequence_classifier import SequenceClassifier

if not os.path.exists('./logs'):
    os.mkdir('./logs')

logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(f'logs/{dt.datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)

fh_clean = logging.FileHandler(f'logs/{dt.datetime.now().strftime("%Y%m%d")}_clean.log', encoding='utf-8')
fh_clean.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s',
                              datefmt='%H:%M:%S')
fh.setFormatter(formatter)
fh_clean.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(fh_clean)
logger.addHandler(ch)


def run():
    dataset = reader.DataReader('./dataset', fileids='eng.train.txt',
                                columntypes=('words', 'pos', 'chunk', 'ne'))
    gen = features.Generator(columntypes=('words', 'pos', 'chunk'), context_len=2, language='en',
                             rare_count=5, min_weight=0.95, rewrite=True, history=True)

    logger.debug(f"Загружаем признаки для обучения!")

    y = [el[1] for el in dataset.get_ne()]
    x = dataset.get_tags(tags=['words', 'pos', 'chunk'])

    logger.debug(f"Переводим данные в формат документов!")
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

    x = gen.fit_generate(x_docs, y_docs, "./prepared_data/conll_trainset.npz")
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

    clf = SequenceClassifier(cls='CRF')
    clf.fit(x_docs, y_docs)
    logger.info(clf.score(x_docs, y_docs))

    logger.debug(f"Прогон параметров завершен!")


if __name__ == '__main__':
    run()
