import reader

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
        
def docs_from_dataset(folder_path, file_name, column_types, used_columns, sent2features):
    dataset = reader.DataReader(folder_path, fileids=file_name, columntypes=column_types)
    y = [el[1] for el in dataset.get_ne()]
    x = dataset.get_tags(tags=used_columns)
    x_sent_base, y_sent = [], []
    index = 0
    for sent in dataset.sents():
        length = len(sent)
        if length == 0:
            continue
        x_sent_base.append(x[index:index + length])
        y_sent.append(y[index:index + length])
        index += length

    x_sent = [sent2features(s) for s in x_sent_base]

    x_docs, y_docs = [], []
    index = 0
    for doc in dataset.docs():
        length = len(doc)
        if length == 0:
            continue
        x_docs.append(x_sent[index:index + length])
        y_docs.append(y_sent[index:index + length])
        index += length
    return x_docs, y_docs