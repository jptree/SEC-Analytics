import os
from gensim.models import Word2Vec, KeyedVectors
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import ast
import numpy as np
import pickle


PATH = '../Data/10-Q Sample'


def modify_formatting(text):
    return text.replace('\n', ' newline ').replace('\t', ' tab ').replace('  ', ' d_s ').replace('.', ' p ')


def get_n_surrounding_words(text, n, trailing=True):
    """
    Searches for text, and retrieves n words either side of the text
    """
    if trailing:
        return text.split()[1:n + 1]
    else:
        return text.split()[::-1][1:n + 1]


def create_vector_model():
    _, _, file_names = next(
        os.walk(PATH))

    corpus = [modify_formatting(open(f'{PATH}/{file}').read()) for file in file_names]
    tokens = [doc.split() for doc in corpus]
    model = Word2Vec(tokens).wv
    model.save('mda.wordvectors')


def create_open_model(wv, df_extracted):

    files = list(df_extracted['file'])
    mda = list(df_extracted['mda'])
    X = []
    y = []
    for m, file_name in enumerate(files):
        text = mda[m].replace('\\n', '\n')
        if text == '-9':
            continue
        file_text = open(f'{PATH}/{file_name}').read()
        open_index = file_text.index(text)
        management_indices = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]
        item_indices = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
        interested_locations = [0, 100, 200, 500, 700, 900, 1100, 1500, open_index]
        interested_locations += management_indices
        interested_locations += item_indices
        for p in interested_locations:
            # close_index = open_index + len(text)
            surrounding_text = file_text[p - 500: p + 500]
            leading_characters = surrounding_text[:500]
            trailing_characters = surrounding_text[500:]
            leading_characters = modify_formatting(leading_characters)
            trailing_characters = modify_formatting(trailing_characters)

            leading_words = get_n_surrounding_words(leading_characters, 20, trailing=False)
            trailing_words = get_n_surrounding_words(trailing_characters, 20)

            min_feature_leading = [0] * 100
            max_feature_leading = [0] * 100
            for word in leading_words:
                try:
                    embedding = wv[word]
                except KeyError:
                    continue

                for i in range(100):
                    if embedding[i] > max_feature_leading[i]:
                        max_feature_leading[i] = embedding[i]
                    if embedding[i] < min_feature_leading[i]:
                        min_feature_leading[i] = embedding[i]

            min_feature_trailing = [0] * 100
            max_feature_trailing = [0] * 100
            for word in trailing_words:
                try:
                    embedding = wv[word]
                except KeyError:
                    print(word)
                    continue

                for i in range(100):
                    if embedding[i] > max_feature_trailing[i]:
                        max_feature_trailing[i] = embedding[i]
                    if embedding[i] < min_feature_trailing[i]:
                        min_feature_trailing[i] = embedding[i]

            item_count = len(surrounding_text.lower().split('item'))

            X.append(min_feature_leading + max_feature_leading + min_feature_trailing + max_feature_trailing + [item_count])
            y.append(1 if p == open_index else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)
    parameters = {'max_depth': [4, 5],
                  'criterion': ('gini', 'entropy'),
                  'class_weight': ('balanced', 'balanced_subsample')}
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, scoring='recall')
    clf.fit(X, y)
    print(f'Best Score: {clf.best_score_}')
    print(f'Best Params: {clf.best_params_}')

    clf = clf.best_estimator_
    plot_confusion_matrix(clf, X_train, y_train, normalize='true', cmap='Blues')
    plt.show()
    plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap='Blues')
    plt.show()

    with open('../rf_quarterly_open_wv.pkl', 'wb') as f:
        pickle.dump(clf, f)


def testing(file_name, wv, clf):
    file_text = open(f'{PATH}/{file_name}').read()
    probabilities = []
    indices = []
    for p in range(len(file_text)):
        # close_index = open_index + len(text)
        surrounding_text = file_text[p - 500: p + 500]
        if ('item' not in surrounding_text.lower()) or ('management' not in surrounding_text.lower()) or ('analysis' not in surrounding_text.lower()):
            continue
        leading_characters = surrounding_text[:500]
        trailing_characters = surrounding_text[500:]
        leading_characters = modify_formatting(leading_characters)
        trailing_characters = modify_formatting(trailing_characters)

        leading_words = get_n_surrounding_words(leading_characters, 20, trailing=False)
        trailing_words = get_n_surrounding_words(trailing_characters, 20)

        min_feature_leading = [0] * 100
        max_feature_leading = [0] * 100
        for word in leading_words:
            try:
                embedding = wv[word]
            except KeyError:
                continue

            for i in range(100):
                if embedding[i] > max_feature_leading[i]:
                    max_feature_leading[i] = embedding[i]
                if embedding[i] < min_feature_leading[i]:
                    min_feature_leading[i] = embedding[i]

        min_feature_trailing = [0] * 100
        max_feature_trailing = [0] * 100
        for word in trailing_words:
            try:
                embedding = wv[word]
            except KeyError:
                print(word)
                continue

            for i in range(100):
                if embedding[i] > max_feature_trailing[i]:
                    max_feature_trailing[i] = embedding[i]
                if embedding[i] < min_feature_trailing[i]:
                    min_feature_trailing[i] = embedding[i]

        item_count = len(surrounding_text.lower().split('item'))

        X = [min_feature_leading + max_feature_leading + min_feature_trailing + max_feature_trailing + [item_count]]
        probabilities.append(clf.predict_proba(X)[0][1])
        indices.append(p)

    index = probabilities.index(max(probabilities))
    index = indices[index]
    print(f'Max Probability: {max(probabilities)}')
    print(file_text[index: index + 500])





if __name__ == "__main__":
    # create_vector_model()
    wv = KeyedVectors.load("mda.wordvectors", mmap='r')
    # create_open_model(wv)

    df = pd.read_csv('NewExtractionSupervised.csv', header=None, names=['file', 'mda'])
    create_open_model(wv, df)
    clf = pickle.load(open('../rf_quarterly_open_wv.pkl', 'rb'))
    # testing('20060202_10-Q_edgar_data_775158_0000897069-06-000275_1.txt', wv, clf)
    testing('19970331_10-Q_edgar_data_62262_0001017062-97-000588_1.txt', wv, clf)

