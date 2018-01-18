class LabelEncoder:
    """
    Используется для сокращения представления меток (в работе оценок)
    """

    def __init__(self):
        self.data = {}
        self.index = 0

    def get(self, label):
        if label in self.data:
            return self.data[label]
        else:
            self.data[label] = self.index
            self.index += 1
            return self.data[label]