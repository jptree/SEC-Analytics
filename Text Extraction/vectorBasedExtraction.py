import os
from gensim.models import Word2Vec, KeyedVectors
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import ast
import numpy as np
import pickle
from shap import TreeExplainer, summary_plot

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
    indices_history = []
    file_history = []
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
            surrounding_text = file_text[p - 500: p + 500]
            features = get_open_features(surrounding_text, file_text, wv, p)

            X.append(features)
            y.append(1 if p == open_index else 0)

            indices_history.append(p)
            file_history.append(file_name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)
    # parameters = {'max_depth': [4, 5],
    #               'criterion': ('gini', 'entropy'),
    #               'class_weight': ('balanced', 'balanced_subsample')}
    # rf = RandomForestClassifier()
    # clf = GridSearchCV(rf, parameters, scoring='recall')
    # clf.fit(X, y)
    # print(f'Best Score: {clf.best_score_}')
    # print(f'Best Params: {clf.best_params_}')
    #
    # clf = clf.best_estimator_
    # plot_confusion_matrix(clf, X_train, y_train, normalize='true', cmap='Blues')
    # plt.show()
    # plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap='Blues')
    # plt.show()
    #
    parameters = {'max_depth': [6],
                  'class_weight': ['balanced']}
    rf = RandomForestClassifier()
    # rf = GradientBoostingClassifier()
    clf = GridSearchCV(rf, parameters, scoring='recall', n_jobs=4)
    # clf = GradientBoostingClassifier()
    # clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    print(f'Best Score: {clf.best_score_}')
    print(f'Best Params: {clf.best_params_}')
    #
    plot_confusion_matrix(clf, X_train, y_train, normalize='true', cmap='Blues')
    plt.show()
    plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap='Blues')
    plt.show()

    # explainer = TreeExplainer(clf)
    # shap_values = explainer.shap_values(np.array(X))
    # summary_plot(shap_values, X)

    with open('../rf_quarterly_open_wv.pkl', 'wb') as f:
        pickle.dump(clf, f)

    predicted = clf.predict(X)
    probs = clf.predict_proba(X)
    for i, file in enumerate(file_history):


        if predicted[i] == y[i]:
            continue
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        m = indices_history[i]

        output_file = open('IsThisTheSection.txt', 'w')
        file_text = f'{file_text[:m]}$%$%$%%${file_text[m:]}'
        modified_file_text = file_text[m - 600: m + 600]
        output_file.write(modified_file_text)
        output_file.close()

        os.startfile('IsThisTheSection.txt')

        print(f'Predicted: {predicted[i]}\nActual   : {y[i]}\nProbability: {probs[i]}')
        print(X[i][-5:])
        input('Continue?')


def testing(file_name, wv, clf):
    file_text = open(f'{PATH}/{file_name}').read()
    probabilities = []
    indices = []
    management_indices = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]
    item_indices = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
    interested_locations = management_indices + item_indices

    output_file = open('IsThisTheSection.txt', 'w')
    output_file.write(file_text)
    output_file.close()

    os.startfile('IsThisTheSection.txt')


    for p in interested_locations:
        # close_index = open_index + len(text)
        surrounding_text = file_text[p - 500: p + 500]
        # a = file_text[p: p + 500]
        features = get_open_features(surrounding_text, file_text, wv, p)

        X = [features]

        predicted_prob = clf.predict_proba(X)[0][1]


        probabilities.append(predicted_prob)
        indices.append(p)

    index = probabilities.index(max(probabilities))
    index = indices[index]
    print(f'Max Probability: {max(probabilities)}')
    print(file_text[index: index + 500])
    input('Continue?')


def get_open_features(surrounding_text, file_text, wv, index):
    leading_characters = surrounding_text[:500]
    trailing_characters = surrounding_text[500:]
    leading_characters = modify_formatting(leading_characters)
    trailing_characters = modify_formatting(trailing_characters)

    leading_words = get_n_surrounding_words(leading_characters, 30, trailing=False)
    trailing_words = get_n_surrounding_words(trailing_characters, 30)

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
            # print(word)
            continue

        for i in range(100):
            if embedding[i] > max_feature_trailing[i]:
                max_feature_trailing[i] = embedding[i]
            if embedding[i] < min_feature_trailing[i]:
                min_feature_trailing[i] = embedding[i]

    # item_count = len(surrounding_text.lower().split('item'))
    leading_item_count = len(file_text[index - 500: index].lower().split('item'))
    trailing_item_count = len(file_text[index: index + 500].lower().split('item'))
    leading_newline_count = len(file_text[index - 500: index].lower().split('\n\n'))
    trailing_newline_count = len(file_text[index: index + 500].lower().split('\n\n'))

    features = min_feature_leading + max_feature_leading + min_feature_trailing + max_feature_trailing + [
        leading_item_count, trailing_item_count, leading_newline_count, trailing_newline_count,
        leading_item_count * leading_newline_count, trailing_item_count * trailing_newline_count,
        index / len(file_text)
    ]

    return features




if __name__ == "__main__":
    # create_vector_model()
    wv = KeyedVectors.load("mda.wordvectors", mmap='r')
    # create_open_model(wv)

    # df = pd.read_csv('NewExtractionSupervised.csv', header=None, names=['file', 'mda'])
    # create_open_model(wv, df)
    clf = pickle.load(open('../rf_quarterly_open_wv.pkl', 'rb'))
    testing('20060202_10-Q_edgar_data_775158_0000897069-06-000275_1.txt', wv, clf)
    testing('19970331_10-Q_edgar_data_62262_0001017062-97-000588_1.txt', wv, clf)
    testing('19970331_10-Q_edgar_data_74154_0000950134-97-002523_1.txt', wv, clf)

