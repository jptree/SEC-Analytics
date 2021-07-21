import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
import os
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import re
import math
warnings.filterwarnings('ignore')


PATH = '../../Data/10-Q Sample'


def bert_test():
    df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
    batch_1 = df[:2000]

    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()

    labels = batch_1[1]

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    print(lr_clf.score(test_features, test_labels))


def modify_formatting(text):
    return text.replace('\n', '\\n').replace('\t', '\\t')


def get_bert():
    df = pd.read_csv('../NewExtractionSupervised.csv', header=None, names=['file', 'mda'])
    file_names = list(df['file'])

    # corpus = [modify_formatting(open(f'{PATH}/{file}').read()) for file in file_names]

    model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)



    mda = list(df['mda'])
    X_text = []
    y = []
    # indices_history = []
    # file_history = []
    for m, file_name in enumerate(file_names):
        text = mda[m].replace('\\n', '\n')
        if text == '-9':
            continue
        file_text = open(f'{PATH}/{file_name}').read()
        open_index = file_text.index(text)
        management_indices = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]
        # item_indices = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
        interested_locations = [100, 200, 500, 700, 900, 1100, 1500, open_index]
        interested_locations += management_indices
        # interested_locations += item_indices

        for p in interested_locations:
            surrounding_text = file_text[p - 400: p + 400].split(' ')[1: -1]
            middle = math.trunc(len(surrounding_text) / 2)
                # a = len(surrounding_text[middle - 250: middle + 250])
            surrounding_text = ' '.join(surrounding_text[middle - 10: middle + 10])

            X_text.append(surrounding_text)
            y.append(1 if p == open_index else 0)

    tokenized = [tokenizer.encode(x, add_special_tokens=True) for x in X_text]

    max_len = 0
    for i in tokenized:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
    # padded = padded[:1000]
    # y = y[:1000]
    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()

    labels = y

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    parameters = {'max_depth': [6, 5, 7, 10],
                  'class_weight': ['balanced']}
    rf = RandomForestClassifier()
    lr_clf = GridSearchCV(rf, parameters, scoring='recall', n_jobs=4)
    # lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    # print(lr_clf.score(test_features, test_labels))
    plot_confusion_matrix(lr_clf, train_features, train_labels, normalize='true', cmap='Blues')
    plt.show()

    plot_confusion_matrix(lr_clf, test_features, test_labels, normalize='true', cmap='Blues')
    plt.show()






if __name__ == "__main__":
    # print('a')
    # bert_test()
    # get_file_corpus()
    get_bert()