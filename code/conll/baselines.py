from sklearn.model_selection import GridSearchCV

import datetime as dt
import logging
import click

import features
import reader
import os

if not os.path.exists('./logs'):
    os.mkdir('./logs')

from classifiers.token_classifier import TokenClassifier

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


@click.command()
@click.option('--mode', '-m', default='demo',
              help='Mode of running (log, svn, gb, rf).')
def run_baselines(mode):
    definitions = {
        'log': {
            'clf': 'LogisticRegression',
            'parameters': [{"C": [0.001, 0.01, 0.1, 1, 10]}]
        },
        'svn': {
            'clf': 'LinearSVC',
            'parameters': [{"C": [0.001, 0.01, 0.1, 1, 10]}]
        },
        'gb': {
            'clf': 'XGBClassifier',
            'parameters': [{"booster": ['gbtree'],
                            'learning_rate': [0.05, 0.2, 0.3, 0.4],
                            'max_depth': [10, 14, 17],
                            'colsample_bytree': [0.5, 0.8, 1],
                            'colsample_bylevel': [0.5, 0.8, 1]}]
        },
        'rf': {
            'clf': 'RandomForestClassifier',
            'parameters': [{
                "criterion": ["gini"],
                "n_estimators": [500],
                "max_features": ['sqrt'],
            }]
        }
    }

    logger.info(f'Подсчет baseline-ов с классификатором {definitions[mode]["clf"]}!')
    dataset = reader.DataReader('./dataset', fileids='eng.train.txt',
                                columntypes=('words', 'pos', 'chunk', 'ne'))
    gen = features.Generator(columntypes=('words', 'pos', 'chunk'), context_len=2, language='en',
                             rare_count=5, min_weight=0.95, rewrite=True, history=True)

    logger.debug(f"Загружаем признаки для обучения!")
    y = [el[1] for el in dataset.get_ne()]
    x = gen.fit_generate(dataset.get_tags(tags=['words', 'pos', 'chunk']), y,
                         "./prepared_data/conll_trainset.npz")

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

    refit = False
    with open('./baselines.txt', 'a+') as file:
        logger.debug(f"Начинаем прогон параметров при помощи GridSearchCV!")
        file.write(f'## {definitions[mode]["clf"]} ##\n')
        file.write(f"started: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        clf = GridSearchCV(TokenClassifier(cls=definitions[mode]["clf"]),
                           definitions[mode]["parameters"], n_jobs=4, cv=3,
                           refit=refit)
        clf.fit(x_docs, y_docs)
        file.write(f"best parameters: {clf.best_params_}\n")
        file.write(f"best result: {clf.best_score_}\n")
        file.write(f"ended: {dt.datetime.now().strftime('%b %d %Y %H:%M:%S')}\n")
        file.write("\n")
    logger.debug(f"Прогон параметров завершен!")
    logger.info(f"Лучшие параметры: {clf.best_params_}")
    logger.info(f"Достигнутый результат: {clf.best_score_}\n")


if __name__ == '__main__':
    run_baselines()
