class Estimator:
    def __init__(self, predictions, correct, label, labels, label2idx):

        self._label = label
        self._labels = labels

        self._predictions = predictions
        self._correct = correct

        self._label2idx = label2idx

        self._recall = 0
        self._precision = 0

        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

    def get_precision(self):
        return self._precision

    def get_recall(self):
        return self._recall

    @staticmethod
    def get_total_f1(labels, y_pred_sent, y_test_sent, encoder):
        true_positive, false_positive, false_negative = 0, 0, 0
        for label in labels:
            est = Estimator(y_pred_sent, y_test_sent, label, labels, encoder)
            est.compute_proper_f1()
            true_positive += est._true_positive
            false_positive += est._false_positive
            false_negative += est._false_negative
        return (2 * true_positive) / (2 * true_positive + false_negative + false_positive)

    def get_f1_measure(self):
        if self._recall + self._precision > 0:
            return 2 * self._precision * self._recall / (self._recall + self._precision)
        else:
            return 0

    def get_weight(self):
        return self._false_negative + self._true_positive

    def compute_precision_and_recall(self):
        """
        Вычисляем точность и полноту по TP, FP и PN
        """
        if self._false_positive + self._true_positive > 0:
            self._precision = float(self._true_positive) / (self._true_positive + self._false_positive)
        else:
            self._precision = 0
        if self._false_negative + self._true_positive > 0:
            self._recall = float(self._true_positive) / (self._true_positive + self._false_negative)
        else:
            self._recall = 0
        return self._recall, self._precision

    def compute_entity_f1(self, beginning, ending):
        """
        Получаем на вход предстказанные и истинные метки, считаем посущностную точность и полноту.
        В true_positive  попадают те сущности, предсказанные начала и концы которых совпадают с истинными.
        false_positive увеличивается когда заканчивается только предсказанная сущность - ей нет пары среди истинных.
        Аналогично, истинная - когда заканчивается только истинная.
        :param beginning: если предыдущий токен не является частью сущности или является ее концом, считаем такой токен
        началом новой сущности
        :param ending: если предыдущий токен являлся частью сущности, а текущий токен такой, то считаем, что предыдущий
        токен был последним элементом сущности
        """

        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

        for guessed_sentence, correct_sentence in zip(self._predictions, self._correct):
            assert (len(guessed_sentence) == len(correct_sentence)), "Guessed and correct sentences do not match"

            # Индекс первого символа "активной" истиной сущности. -1 если активной сущности нет
            correct_ne_start = -1
            # Индекс первого символа "активной" предсказанной сущности. -1 если активной сущности нет
            guessed_ne_start = -1

            for j in range(len(guessed_sentence)):
                # Завершать сущности имеет смысл, только если они уже начаты
                correct_ends = (correct_ne_start != -1) and (correct_sentence[j] in ending)
                guessed_ends = (guessed_ne_start != -1) and (guessed_sentence[j] in ending)

                if correct_ends and guessed_ends and correct_ne_start == guessed_ne_start:
                    self._true_positive += 1
                    correct_ne_start = -1
                    guessed_ne_start = -1
                else:
                    if guessed_ends:
                        self._false_positive += 1
                        guessed_ne_start = -1
                    if correct_ends:
                        self._false_negative += 1
                        correct_ne_start = -1
                # Начинать новую сущность имеет смысл только если предыдущей не было, или она закончилась
                if (correct_ne_start == -1) and (correct_sentence[j] in beginning):
                    correct_ne_start = j
                if (guessed_ne_start == -1) and (guessed_sentence[j] in beginning):
                    guessed_ne_start = j

        self.compute_precision_and_recall()

    def compute_proper_f1(self):
        """
        Учточнение метода compute_entity_f1 для следующего "честного" способа склейки в сущности:
        1) Начинаем новую сущность с любой положительной метки, если предыдущий токен не был частью
        сущности или был ее концом.
        2) Заканчиваем текущую сущность если очередной токен или имеет отрицательную метку, или имеет метки, свойственные
         началу сущности - B или S
        """
        beginning = [self._label2idx.get('B-' + self._label), self._label2idx.get('E-' + self._label),
                     self._label2idx.get('S-' + self._label), self._label2idx.get('I-' + self._label)]
        ending = [self._label2idx.get('S-' + self._label), self._label2idx.get('B-' + self._label),
                  self._label2idx.get('O')]

        other_entity_types = [entity for entity in self._labels if entity != self._label]
        for other_entity_type in other_entity_types:
            ending.append(self._label2idx.get('S-' + other_entity_type))
            ending.append(self._label2idx.get('B-' + other_entity_type))
            ending.append(self._label2idx.get('I-' + other_entity_type))
            ending.append(self._label2idx.get('E-' + other_entity_type))

        beginning = {el for el in beginning if el is not None}
        ending = {el for el in ending if el is not None}

        self.compute_entity_f1(beginning, ending)

        return self.get_f1_measure()
