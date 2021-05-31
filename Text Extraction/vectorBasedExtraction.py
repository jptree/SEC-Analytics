import os
from gensim.models import Word2Vec, KeyedVectors
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pickle
# from shap import TreeExplainer, summary_plot

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

    parameters = {'max_depth': [6],
                  'class_weight': ['balanced']}
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, scoring='recall', n_jobs=4)

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

    with open('../../rf_quarterly_open_wv.pkl', 'wb') as f:
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


def testing_open_model(file_name, wv, clf, use_this_path=None):
    if use_this_path:
        file_text = open(f'{use_this_path}/{file_name}').read()
    else:
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
        a = file_text[p: p + 500]
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


def get_close_features(surrounding_text, file_text, wv, index):
    # Test: 0.93/0.9
    leading_characters = surrounding_text[:500]
    trailing_characters = surrounding_text[500:]
    leading_characters = modify_formatting(leading_characters)
    trailing_characters = modify_formatting(trailing_characters)

    leading_words = get_n_surrounding_words(leading_characters, 30, trailing=False)
    trailing_words = get_n_surrounding_words(trailing_characters, 30)

    min_feature_leading = [0] * 100
    max_feature_leading = [0] * 100

    isQuantitativeInDocument = 1 if len(re.findall(r'quantitative[\s\S]+qualitative', file_text, re.IGNORECASE)) > 0 else 0
    isControlsInDocument = 1 if len(re.findall(r'controls[\s\S]+procedures', file_text, re.IGNORECASE)) > 0 else 0

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
        index / len(file_text), isQuantitativeInDocument, isControlsInDocument
    ]

    return features


def get_close_features_2(surrounding_text, file_text, wv, index):
    #Test: 0.93/0.94
    leading_characters = surrounding_text[:500]
    trailing_characters = surrounding_text[500:]
    leading_characters = modify_formatting(leading_characters)
    trailing_characters = modify_formatting(trailing_characters)

    leading_words = get_n_surrounding_words(leading_characters, 30, trailing=False)
    trailing_words = get_n_surrounding_words(trailing_characters, 30)

    min_feature_leading = [0] * 100
    max_feature_leading = [0] * 100

    isQuantitativeInDocument = 1 if len(re.findall(r'quantitative[\s\S]+qualitative', file_text, re.IGNORECASE)) > 0 else 0
    isControlsInDocument = 1 if len(re.findall(r'controls[\s\S]+procedures', file_text, re.IGNORECASE)) > 0 else 0

    average_leading = [0] * 100
    leading_counter = 0
    for word in leading_words:
        try:
            embedding = wv[word]
            leading_counter += 1
            average_leading = np.add(average_leading, embedding)
        except KeyError:
            continue

        for i in range(100):
            if embedding[i] > max_feature_leading[i]:
                max_feature_leading[i] = embedding[i]
            if embedding[i] < min_feature_leading[i]:
                min_feature_leading[i] = embedding[i]

    average_leading = np.divide(average_leading, leading_counter)

    min_feature_trailing = [0] * 100
    max_feature_trailing = [0] * 100
    average_trailing = [0] * 100
    trailing_counter = 0
    for word in trailing_words:
        try:
            embedding = wv[word]
            trailing_counter += 1
            average_trailing = np.add(average_trailing, embedding)
        except KeyError:
            # print(word)
            continue

        for i in range(100):
            if embedding[i] > max_feature_trailing[i]:
                max_feature_trailing[i] = embedding[i]
            if embedding[i] < min_feature_trailing[i]:
                min_feature_trailing[i] = embedding[i]

    if trailing_counter != 0:
        average_trailing = np.divide(average_trailing, trailing_counter)

    # item_count = len(surrounding_text.lower().split('item'))
    leading_item_count = len(file_text[index - 500: index].lower().split('item'))
    trailing_item_count = len(file_text[index: index + 500].lower().split('item'))
    leading_newline_count = len(file_text[index - 500: index].lower().split('\n\n'))
    trailing_newline_count = len(file_text[index: index + 500].lower().split('\n\n'))

    features = min_feature_leading + max_feature_leading + min_feature_trailing + max_feature_trailing + [
        leading_item_count, trailing_item_count, leading_newline_count, trailing_newline_count,
        leading_item_count * leading_newline_count, trailing_item_count * trailing_newline_count,
        index / len(file_text), isQuantitativeInDocument, isControlsInDocument
    ] + list(average_leading) + list(average_trailing)

    return features


def get_close_features_3(surrounding_text, file_text, wv, index):
    #Test: 0.91/0.96
    leading_characters = surrounding_text[:500]
    trailing_characters = surrounding_text[500:]
    leading_characters = modify_formatting(leading_characters)
    trailing_characters = modify_formatting(trailing_characters)

    leading_words = get_n_surrounding_words(leading_characters, 30, trailing=False)
    trailing_words = get_n_surrounding_words(trailing_characters, 40)

    min_feature_leading = [0] * 100
    max_feature_leading = [0] * 100

    isQuantitativeInDocument = 1 if len(re.findall(r'quantitative[\s\S]+qualitative', file_text, re.IGNORECASE)) > 0 else 0
    isControlsInDocument = 1 if len(re.findall(r'controls[\s\S]+procedures', file_text, re.IGNORECASE)) > 0 else 0

    average_leading = [0] * 100
    leading_counter = 0
    for word in leading_words:
        try:
            embedding = wv[word]
            leading_counter += 1
            average_leading = np.add(average_leading, embedding)
        except KeyError:
            continue

        for i in range(100):
            if embedding[i] > max_feature_leading[i]:
                max_feature_leading[i] = embedding[i]
            if embedding[i] < min_feature_leading[i]:
                min_feature_leading[i] = embedding[i]

    average_leading = np.divide(average_leading, leading_counter)

    min_feature_trailing = [0] * 100
    max_feature_trailing = [0] * 100
    average_trailing = [0] * 100
    trailing_counter = 0
    for word in trailing_words:
        try:
            embedding = wv[word]
            trailing_counter += 1
            average_trailing = np.add(average_trailing, embedding)
        except KeyError:
            # print(word)
            continue

        for i in range(100):
            if embedding[i] > max_feature_trailing[i]:
                max_feature_trailing[i] = embedding[i]
            if embedding[i] < min_feature_trailing[i]:
                min_feature_trailing[i] = embedding[i]

    if trailing_counter != 0:
        average_trailing = np.divide(average_trailing, trailing_counter)

    # item_count = len(surrounding_text.lower().split('item'))
    leading_item_count = len(file_text[index - 500: index].lower().split('item'))
    trailing_item_count = len(file_text[index: index + 500].lower().split('item'))
    leading_newline_count = len(file_text[index - 500: index].lower().split('\n\n'))
    trailing_newline_count = len(file_text[index: index + 500].lower().split('\n\n'))

    features = min_feature_leading + max_feature_leading + min_feature_trailing + max_feature_trailing + [
        leading_item_count, trailing_item_count, leading_newline_count, trailing_newline_count,
        leading_item_count * leading_newline_count, trailing_item_count * trailing_newline_count,
        index / len(file_text), isQuantitativeInDocument, isControlsInDocument
    ] + list(average_leading) + list(average_trailing)

    return features


def create_close_model(wv, df_extracted):

    files = list(df_extracted['file'])
    mda = list(df_extracted['mda'])
    X = []
    y = []
    indices_history = []
    file_history = []
    for m, file_name in enumerate(files):
        text = mda[m].replace('\\n', '\n').rstrip()
        if text == '-9':
            continue
        file_text = open(f'{PATH}/{file_name}').read()
        close_index = file_text.index(text) + len(text)
        open_index = file_text.index(text)

        newline_indices = [m.start() for m in re.finditer('\n', file_text[open_index: open_index + len(text) + 700], re.IGNORECASE)]
        interested_locations = []
        for n in newline_indices:
            questionable_text = file_text[n + open_index: n + open_index + 500].lower()

            if 'item' in questionable_text:
                interested_locations.append(n + open_index)
            elif 'quantitative' in questionable_text:
                interested_locations.append(n + open_index)
            elif 'control' in questionable_text:
                interested_locations.append(n + open_index)
            elif n > len(file_text[open_index:]) - 1000:
                interested_locations.append(n + open_index)
            elif 'report' in questionable_text:
                interested_locations.append(n + open_index)



        closest_index_distance = 999
        closest_index = 0
        for i in interested_locations:
            if (i > close_index) and (i - close_index < closest_index_distance):
                closest_index_distance = i - close_index
                closest_index = i
                break

        if closest_index_distance > 30:
            print(file_name)
            output_file = open('IsThisTheSection.txt', 'w')
            output_file.write(file_text)
            output_file.close()

            os.startfile('IsThisTheSection.txt')
            input('Continue???')


        for p in interested_locations:
            surrounding_text = file_text[p - 500: p + 500]


            features = get_close_features_3(surrounding_text, file_text, wv, p)

            X.append(features)
            y.append(1 if p == closest_index else 0)


            indices_history.append(p)
            file_history.append(file_name)

        print(closest_index_distance)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)

    parameters = {'max_depth': [4, 5, 6],
                  'class_weight': ['balanced_subsample']}
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, parameters, scoring='recall', n_jobs=4)

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

    with open('../rf_quarterly_close_wv.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # predicted = clf.predict(X)
    # probs = clf.predict_proba(X)
    # for i, file in enumerate(file_history):
    #
    #
    #     if predicted[i] == y[i]:
    #         continue
    #     file_text = open(f'../Data/10-Q Sample//{file}').read()
    #     m = indices_history[i]
    #
    #     output_file = open('IsThisTheSection.txt', 'w')
    #     file_text = f'{file_text[:m]}$%$%$%%${file_text[m:]}'
    #     modified_file_text = file_text[m - 600: m + 600]
    #     output_file.write(modified_file_text)
    #     output_file.close()
    #
    #     os.startfile('IsThisTheSection.txt')
    #
    #     print(f'Predicted: {predicted[i]}\nActual   : {y[i]}\nProbability: {probs[i]}')
    #     print(X[i][-5:])
    #     input('Continue?')


def testing_close_model(file_name, wv, clf, use_this_path=None):
    if use_this_path:
        file_text = open(f'{use_this_path}/{file_name}').read()
    else:
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
        a = file_text[p: p + 500]
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


def get_mda(file_text, clf_open, clf_close, wv, add_extra = False):
    open_probabilities = []
    open_indices = []
    management_indices = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]
    item_indices = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
    interested_locations = management_indices + item_indices

    for p in interested_locations:
        surrounding_text = file_text[p - 500: p + 500]
        features = get_open_features(surrounding_text, file_text, wv, p)

        X = [features]

        predicted_prob = clf_open.predict_proba(X)[0][1]

        open_probabilities.append(predicted_prob)
        open_indices.append(p)

    index = open_probabilities.index(max(open_probabilities))
    open_index = open_indices[index]

    close_probabilities = []
    close_indices = []
    newline_indices = [m.start() for m in
                       re.finditer('\n', file_text[open_index:], re.IGNORECASE)]
    interested_locations = []
    for n in newline_indices:
        questionable_text = file_text[n + open_index: n + open_index + 500].lower()

        if 'item' in questionable_text:
            interested_locations.append(n + open_index)
        elif 'quantitative' in questionable_text:
            interested_locations.append(n + open_index)
        elif 'control' in questionable_text:
            interested_locations.append(n + open_index)
        elif n > len(file_text[open_index:]) - 1000:
            interested_locations.append(n + open_index)
        elif 'report' in questionable_text:
            interested_locations.append(n + open_index)

    found = False
    for p in interested_locations:
        surrounding_text = file_text[p - 500: p + 500]
        features = get_close_features_3(surrounding_text, file_text, wv, p)

        X = [features]

        predicted_prob = clf_close.predict_proba(X)[0][1]

        if predicted_prob > 0.7:
            close_index = p
            found = True
            break

        close_indices.append(p)
        close_probabilities.append(predicted_prob)

    if not found:
        index = close_probabilities.index(max(close_probabilities))
        close_index = close_indices[index]

    if add_extra:

        return file_text[open_index: close_index + 200]
    else:
        return file_text[open_index: close_index]


def mda_test():
    test_files = ['19971002_10-Q_edgar_data_49146_0000950116-97-001820_1.txt',
                  '19971002_10-Q_edgar_data_835909_0001037979-97-000009_1.txt',
                  '19971002_10-Q_edgar_data_940944_0000940944-97-000117_1.txt']

    for t in test_files:
        pa = f'D:/SEC Filing Data/10-X_C_1993-2000/1997/QTR4/{t}'
        clf_open = pickle.load(open('../rf_quarterly_open_wv.pkl', 'rb'))
        clf_close = pickle.load(open('../rf_quarterly_close_wv.pkl', 'rb'))
        file_text_test = open(pa).read()
        mda = get_mda(file_text_test, clf_open, clf_close, wv, add_extra=True)

        output_file = open('IsThisTheSection.txt', 'w')
        output_file.write(mda)
        output_file.close()

        os.startfile('IsThisTheSection.txt')

    test_files = ['20060202_10-Q_edgar_data_775158_0000897069-06-000275_1.txt',
                  '19970331_10-Q_edgar_data_62262_0001017062-97-000588_1.txt',
                  '19970331_10-Q_edgar_data_74154_0000950134-97-002523_1.txt']
    for t in test_files:
        pa = f'{PATH}/{t}'
        clf_open = pickle.load(open('../rf_quarterly_open_wv.pkl', 'rb'))
        clf_close = pickle.load(open('../rf_quarterly_close_wv.pkl', 'rb'))
        file_text_test = open(pa).read()
        mda = get_mda(file_text_test, clf_open, clf_close, wv, add_extra=True)

        output_file = open('IsThisTheSection.txt', 'w')
        output_file.write(mda)
        output_file.close()

        os.startfile('IsThisTheSection.txt')

    test_files = ['20010103_10-Q_edgar_data_70033_0000931763-01-000004_1.txt',
                  '20010103_10-Q_edgar_data_723254_0000892251-01-000001_1.txt',
                  '20010103_10-Q_edgar_data_799511_0001010412-01-500003_1.txt',
                  '20010103_10-Q_edgar_data_949301_0000950153-01-000004_1.txt']


    for t in test_files:
        pa = f'D:/SEC Filing Data/10-X_C_2001-2005/2001/QTR1/{t}'
        clf_open = pickle.load(open('../rf_quarterly_open_wv.pkl', 'rb'))
        clf_close = pickle.load(open('../rf_quarterly_close_wv.pkl', 'rb'))
        file_text_test = open(pa).read()
        mda = get_mda(file_text_test, clf_open, clf_close, wv, add_extra=True)

        output_file = open('IsThisTheSection.txt', 'w')
        output_file.write(mda)
        output_file.close()

        os.startfile('IsThisTheSection.txt')


if __name__ == "__main__":
    # create_vector_model()
    wv = KeyedVectors.load("mda.wordvectors", mmap='r')
    # create_open_model(wv)

    # df = pd.read_csv('NewExtractionSupervised.csv', header=None, names=['file', 'mda'])
    # create_open_model(wv, df)
    # clf = pickle.load(open('../rf_quarterly_open_wv.pkl', 'rb'))
    # testing('20060202_10-Q_edgar_data_775158_0000897069-06-000275_1.txt', wv, clf)
    # testing('19970331_10-Q_edgar_data_62262_0001017062-97-000588_1.txt', wv, clf)
    # testing('19970331_10-Q_edgar_data_74154_0000950134-97-002523_1.txt', wv, clf)
    # testing('19971002_10-Q_edgar_data_49146_0000950116-97-001820_1.txt', wv, clf, use_this_path='D:/SEC Filing Data/10-X_C_1993-2000/1997/QTR4')
    # testing('19971002_10-Q_edgar_data_835909_0001037979-97-000009_1.txt', wv, clf, use_this_path='D:/SEC Filing Data/10-X_C_1993-2000/1997/QTR4')
    # testing('19971002_10-Q_edgar_data_940944_0000940944-97-000117_1.txt', wv, clf, use_this_path='D:/SEC Filing Data/10-X_C_1993-2000/1997/QTR4')
    # testing_open_model('19971003_10-Q_edgar_data_51410_0000051410-97-000029_1.txt', wv, clf, use_this_path='D:/SEC Filing Data/10-X_C_1993-2000/1997/QTR4')


    # df = pd.read_csv('NewExtractionSupervised.csv', header=None, names=['file', 'mda'])
    # create_close_model(wv, df)


    mda_test()
