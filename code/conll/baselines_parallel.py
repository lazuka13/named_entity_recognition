from multiprocessing import Pool
import argparse

import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid

import generator
import corpus
import estimator
import utils

TRAINSET_PATH = "./prepared_data/conll_trainset.npz"
TESTSETA_PATH = "./prepared_data/conll_testseta.npz"
TESTSETB_PATH = "./prepared_data/conll_testsetb.npz"

conll_trainset = corpus.ConllDataReader('./dataset',
                                        fileids='eng.train.txt',
                                        columntypes=('words', 'pos', 'chunk', 'ne'))

conll_testseta = corpus.ConllDataReader('./dataset',
                                        fileids='eng.testa.dev.txt',
                                        columntypes=('words', 'pos', 'chunk', 'ne'))

gen = generator.Generator(column_types=['WORD', 'POS', 'CHUNK'], context_len=2, language='en')

Y_train = [el[1] for el in conll_trainset.get_ne()]
Y_testa = [el[1] for el in conll_testseta.get_ne()]

X_train = gen.fit_transform(conll_trainset.get_tags(tags=['words', 'pos', 'chunk']), Y_train, path=TRAINSET_PATH)
X_testa = gen.transform(conll_testseta.get_tags(tags=['words', 'pos', 'chunk']), path=TESTSETA_PATH)


def save_to_file(results, file_name):
    """
    Используется для записи результатов прогонов в файл
    :param results: Результаты прогона (dict)
    :param file_name: Имя файла (string)
    :return:
    """
    file = open(file_name, "a+")
    for result_el in results:

        file.write("parameters: {}\n".format(result_el["parameters"]))

        labels = ["PER", "ORG", "LOC", "MISC", "TOTAL"]
        for label in labels:
            file.write("{}: {}\n".format(label, result_el[label]))

        file.write("\n")


def run(data):
    """
    Используется для параллельного прогона параметров на классификаторе
    :param data: Набор параметров и данных (tuple)
    :return:
    """
    parameters, x_train, y_train, x_testa, y_testa, cls = data[0], data[1], data[2], data[3], data[4], data[5]

    clf = cls()
    clf.set_params(**parameters)
    clf.fit(x_train, y_train)
    y_preda = clf.predict(x_testa)

    # преобразуем данные для оценки
    encoder = utils.LabelEncoder()
    y_preda_sent = []
    y_testa_sent = []
    index = 0
    for sent in conll_testseta.sents():
        length = len(sent)
        y_preda_sent.append([encoder.get(el) for el in y_preda[index:index + length]])
        y_testa_sent.append([encoder.get(el) for el in y_testa[index:index + length]])
        index += length

    # производим оценку
    results = {"parameters": parameters}

    labels = ["PER", "ORG", "LOC", "MISC"]
    for label in labels:
        est = estimator.Estimator(y_preda_sent, y_testa_sent, label, labels, encoder)
        results[label] = est.compute_proper_f1()

    results["TOTAL"] = estimator.Estimator.get_total_f1(labels, y_preda_sent, y_testa_sent, encoder)
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GridSearch script for CoNLL Dataset')
    parser.add_argument('type', type=str,
                        help='Type of classifier (all, log, rf, gb, svc)')
    args = parser.parse_args()

    # создаем пул процессов #
    pool = Pool(2)

    if not os.path.exists("./baselines_results"):
        os.mkdir("./baselines_results")

    if args.type == "all" or args.type == "log":
        print("Прогон LogisticRegression начат...")
        parameters_logistic_regression = [
            {
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            }
        ]

        data_to_send = [(params,
                         X_train,
                         Y_train,
                         X_testa,
                         Y_testa,
                         LogisticRegression) for params in ParameterGrid(parameters_logistic_regression)]

        results_logistic_regression = pool.map(run, data_to_send)

        best_result = None
        best_total = 0
        for result in results_logistic_regression:
            if result["TOTAL"] > best_total:
                best_result = result
                best_total = result["TOTAL"]
        best_result_logistic_regression = [best_result]

        save_to_file(results_logistic_regression, file_name="baselines_results/logistic_regression.log")
        save_to_file(best_result_logistic_regression, file_name="baselines_results/logistic_regression.best")
        print("Прогон LogisticRegression завершен...")

    if args.type == "all" or args.type == "rf":
        print("Прогон RandomForest начат...")
        parameters_random_forest = [
            {
                "criterion": ["gini", "entropy"],
                "n_estimators": [100, 500, 1000],
                "max_depth": [3, 5],
                "min_samples_split": [15, 20],
                "min_samples_leaf": [5, 10],
                "max_leaf_nodes": [20, 40],
                "min_weight_fraction_leaf": [0.1]
            }
        ]

        data_to_send = [(params,
                         X_train,
                         Y_train,
                         X_testa,
                         Y_testa,
                         RandomForestClassifier) for params in ParameterGrid(parameters_random_forest)]

        results_random_forest = pool.map(run, data_to_send)

        best_result = None
        best_total = 0
        for result in results_random_forest:
            if result["TOTAL"] > best_total:
                best_result = result
                best_total = result["TOTAL"]
        best_result_random_forest = [best_result]

        save_to_file(results_random_forest, file_name="baselines_results/random_forest.log")
        save_to_file(best_result_random_forest, file_name="baselines_results/random_forest.best")
        print("Прогон RandomForest завершен...")

    if args.type == "all" or args.type == "svc":
        print("Прогон LinearSVC начат...")
        parameters_linear_svc = [
            {
                "C": [0.001, 0.01, 0.1, 1, 10]
            }
        ]

        data_to_send = [(params,
                         X_train,
                         Y_train,
                         X_testa,
                         Y_testa,
                         LinearSVC) for params in ParameterGrid(parameters_linear_svc)]

        results_linear_svc = pool.map(run, data_to_send)

        best_result = None
        best_total = 0
        for result in results_linear_svc:
            if result["TOTAL"] > best_total:
                best_result = result
                best_total = result["TOTAL"]
        best_result_linear_svc = [best_result]

        save_to_file(results_linear_svc, file_name="baselines_results/linear_svc.log")
        save_to_file(best_result_linear_svc, file_name="baselines_results/linear_svc.best")
        print("Прогон LinearSVC завершен...")

    if args.type == "all" or args.type == "gb":
        print("Прогон GradientBoosting начат...")
        parameters_gradient_boosting = [
            {
                'max_depth': [5, 10, 15],
                'min_samples_split': [200, 500, 1000],
                'n_estimators': [10, 50, 100, 500],
                'max_features': [50.0, 75.0, 90.0],
                'subsample': [0.6, 0.8, 0.9]
            }
        ]

        data_to_send = [(params,
                         X_train,
                         Y_train,
                         X_testa,
                         Y_testa,
                         GradientBoostingClassifier) for params in ParameterGrid(parameters_gradient_boosting)]

        results_gradient_boosting = pool.map(run, data_to_send)

        best_result = None
        best_total = 0
        for result in results_gradient_boosting:
            if result["TOTAL"] > best_total:
                best_result = result
                best_total = result["TOTAL"]
        best_result_gradient_boosting = [best_result]

        save_to_file(results_gradient_boosting, file_name="baselines_results/gradient_boosting.log")
        save_to_file(best_result_gradient_boosting, file_name="baselines_results/gradient_boosting.best")
        print("Прогон GradientBoosting завершен...")
