import os
import pickle
import re
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# from gensim.models import word2vec, Word2Vec
import tensorflow as tf
import datetime

# https://www.sec.gov/files/form10-q.pdf

PATH = '../Data/10-Q Sample'
REGEX_10Q = r"(Item[\s]+?2\.[\s\S]*?)(Item[\s]+?3\.)"

OPEN_INDEPENDENT_VARIABLES = ['position', 'trailing_management', 'trailing_period', 'trailing_2', 'trailing_newline',
                              'leading_newline', 'total_size', 'regex_open', 'trailing_analysis']
CLOSE_INDEPENDENT_VARIABLES = ['position', 'trailing_period', 'trailing_3', 'leading_newline', 'total_size',
                               'leading_tab', 'leading_spaces', 'regex_close', 'trailing_quantitative']

OPEN_CLASSIFIER = pickle.load(open('../open_quarterly_random_forest.pkl', 'rb'))
CLOSE_CLASSIFIER = pickle.load(open('../close_quarterly_random_forest.pkl', 'rb'))


def get_mda(clf_open, clf_close, file_text):

    items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]

    total_text_length = len(file_text)
    match = re.findall(REGEX_10Q, file_text, re.IGNORECASE)

    open_probabilities = []
    open_indices = []
    close_probabilities = []
    close_indices = []



    for index, item in enumerate(items):
        trailing_80 = file_text[item: item + 80].lower()
        trailing_management = 1 if 'management' in trailing_80 else 0
        trailing_analysis = 1 if 'analysis' in trailing_80 else 0
        trailing_quantitative = 1 if 'quantitative' in trailing_80 else 0

        if sum([trailing_management, trailing_analysis, trailing_quantitative]) < 1:
            continue

        trailing_50 = file_text[item: item + 50].lower()
        trailing_100 = file_text[item: item + 100].lower()
        trailing_20 = file_text[item: item + 20].lower()

        trailing_period = 1 if '.' in trailing_20 else 0
        trailing_2 = 1 if '2' in trailing_20 else 0
        trailing_3 = 1 if '3' in trailing_20 else 0
        trailing_newline = 1 if '\n' in trailing_100 else 0

        leading_newline = 1 if '\n' in file_text[item - 5: item] else 0
        leading_spaces = 1 if '  ' in file_text[item - 5: item] else 0
        leading_tab = 1 if '\t' in file_text[item - 5: item] else 0







        regex_open = 0
        regex_close = 0

        try:
            mda = match[-1][0]
            if file_text[item: item + len(mda)] == mda:
                regex_open = 1
        except IndexError:
            e = 1

        try:
            mda = match[-1][0]
            if file_text[item - len(mda): item] == mda:
                regex_close = 1
        except IndexError:
            e = 1


        data = {
            'position': item / total_text_length,
            'trailing_management': trailing_management,
            'trailing_period': trailing_period,
            'trailing_2': trailing_2,
            'trailing_newline': trailing_newline,
            'leading_newline': leading_newline,
            'total_size': total_text_length,
            'regex_open': regex_open,
            'trailing_analysis': trailing_analysis,
            'trailing_3': trailing_3,
            'leading_tab': leading_tab,
            'leading_spaces': leading_spaces,
            'regex_close': regex_close,
            'trailing_quantitative': trailing_quantitative
        }

        open_features = []
        for f in OPEN_INDEPENDENT_VARIABLES:
            open_features.append(data[f])

        close_features = []
        for f in CLOSE_INDEPENDENT_VARIABLES:
            close_features.append(data[f])


        open_probability = clf_open.predict([open_features])
        close_probability = clf_close.predict([close_features])

        open_probabilities.append(open_probability)
        close_probabilities.append(close_probability)

        open_indices.append(item)
        close_indices.append(item)

    try:
        open_index = open_probabilities.index(max(open_probabilities))
        close_index = close_probabilities.index(max(close_probabilities))
        # print('Open: ', open_index)
        # print('Close: ', close_index)

    except ValueError:
        return ''

    if open_index > close_index:
        return ''
    else:
        # return file_text[items[open_index]: items[close_index]]
        return file_text[open_indices[open_index]: (close_indices[close_index] + 200)].replace('\n', '\\n'), items.index(open_indices[open_index]), items.index(close_indices[close_index])


def get_supervised_mda(new_file_name):
    # already_done = []
    already_done = open(new_file_name).read()

    _, _, file_names = next(
        os.walk(PATH))

    for file_name in file_names:
        if file_name in already_done:
            print('file already done!')
            continue

        file_text = open(f'{PATH}\\{file_name}').read()

        mda = get_mda(OPEN_CLASSIFIER, CLOSE_CLASSIFIER, file_text)

        output_file = open('IsThisTheSection.txt', 'w')
        try:
            output_file.write(mda[0].replace('\\n', '\n'))
        except IndexError:
            output_file.write('')
        output_file.close()

        modified_file_text = ''
        for index, s in enumerate(re.split(r'item', file_text, flags=re.IGNORECASE)):
            modified_file_text += f'{s} item $%$ {index} $%$'

        modified_file = open('FileInQuestion.txt', 'w')
        modified_file.write(modified_file_text.replace('\\n', '\n'))
        modified_file.close()

        os.startfile('FileInQuestion.txt')
        os.startfile('IsThisTheSection.txt')

        try:
            print(f'\n{file_name}\nOpening: {mda[1]}\nClosing: {mda[2]}')
        except IndexError:
            print(f'\n{file_name}\nOpening: Nothing!\nClosing: Nothing!')

        actual_open = input('What is the open index? :')
        actual_close = input('What is the close index? :')

        if actual_open == '':
            actual_open = mda[1]

        if actual_close == '':
            actual_close = mda[2]

        supervised_file = open(new_file_name, 'a')
        supervised_file.write(f'{file_name},{actual_open},{actual_close}\n')
        supervised_file.close()


def convert_item_to_management(file_name):
    df = pd.read_csv(file_name, header=None, names=['file', 'open', 'close'])
    files = list(df['file'])
    open_indices = df['open']

    for i, file in enumerate(files):
        if open_indices[i] == -9:
            continue
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
        open_item_position = items[open_indices[i]]
        managements = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]

        management_index = -9
        for i_m, m in enumerate(managements):
            distance = m - open_item_position
            if distance > 0:
                management_index = i_m
                break

        supervised_file = open('supervisedManagement.csv', 'a')
        supervised_file.write(f'{file},{management_index}\n')
        supervised_file.close()


def convert_item_to_new_closing(file_name):
    df = pd.read_csv(file_name, header=None, names=['file', 'open', 'close'])
    files = list(df['file'])
    close_indices = df['close']

    for i, file in enumerate(files):
        if close_indices[i] == -9:
            continue
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
        close_item_position = items[close_indices[i]]
        quantitatives = [m.start() for m in re.finditer('quantitative', file_text, re.IGNORECASE)]
        controls = [m.start() for m in re.finditer('controls', file_text, re.IGNORECASE)]

        quantitative_index = -9
        control_index = -9
        quantitative_distance = None
        control_distance = None
        for i_m, m in enumerate(quantitatives):
            distance = m - close_item_position
            if distance > 0:
                quantitative_distance = distance
                quantitative_index = i_m
                break

        for i_m, m in enumerate(controls):
            distance = m - close_item_position
            if distance > 0:
                control_distance = distance
                control_index = i_m
                break

        quantitativeOrControl = None
        index = None

        if (quantitative_index == -9) and (control_index == -9):
            pass
        elif (quantitative_index == -9):
            quantitativeOrControl = 1
            index = control_index
        elif (control_index == -9):
            quantitativeOrControl = 0
            index = quantitative_index
        else:
            if quantitative_distance < control_distance:
                quantitativeOrControl = 0
                index = quantitative_index
            else:
                quantitativeOrControl = 1
                index = control_index

        supervised_file = open('supervisedClosing.csv', 'a')
        supervised_file.write(f'{file},{quantitativeOrControl},{index}\n')
        supervised_file.close()



def threshold_predict(classifier, X, threshold):
    y = classifier.predict_proba(X)
    return [1 if prob[1] >= threshold else 0 for prob in y]


def create_feature_space_opening_model():
    data = pd.DataFrame(columns=['file', 'open_target', 'position', 'text_size', 'trailing_analysis',
                                 'trailing_discussion', 'leading_item', 'leading_newline_count', 'leading_item_count',
                                 'n_management', 'trailing_uppercase', 'next_newline_distance', 'trailing_newline_count',
                                 'trailing_continue', 'leading_2', 'trailing_item_count'])

    df = pd.read_csv('supervisedManagement.csv', header=None, names=['file', 'open', 'close'])
    files = list(df['file'])
    open_indices = df['open']

    for file_i, file in enumerate(files):
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        managements = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]

        y_open_index = open_indices[file_i]
        total_text_length = len(file_text)
        for i, index in enumerate(managements):
            trailing_text = file_text[index:index + 100].lower()
            trailing_text_large = file_text[index:index + 500].lower()
            leading_text_small_window = file_text[index - 40:index].lower()
            leading_text_medium_window = file_text[index - 100:index].lower()
            leading_text_large_window = file_text[index - 500:index].lower()

            trailing_discussion = 1 if 'discussion' in trailing_text else 0
            trailing_continue = 1 if 'continue' in trailing_text_large else 0
            trailing_analysis = 1 if 'analysis' in trailing_text else 0
            leading_item = 1 if 'item' in leading_text_medium_window else 0
            leading_2 = 1 if '2' in leading_text_medium_window else 0
            leading_newline_count = len(leading_text_small_window.split('\n'))
            trailing_newline_count = len(trailing_text.split('\n'))
            leading_item_count = len(leading_text_large_window.split('item'))
            trailing_item_count = len(trailing_text_large.split('item'))
            trailing_uppercase = 1 if file_text[index:index + 50].isupper() else 0

            next_newline_distance = 0
            try:
                next_newline_distance = trailing_text.index('\n')
            except ValueError:
                pass

            open_target = 1 if y_open_index == i else 0
            data = data.append(
                {
                    'file': file,
                    'open_target': open_target,
                    'position': index / total_text_length,
                    'text_size': total_text_length,
                    'trailing_analysis': trailing_analysis,
                    'trailing_discussion': trailing_discussion,
                    'leading_item': leading_item,
                    'leading_newline_count': leading_newline_count,
                    'leading_item_count': leading_item_count,
                    'n_management': i,
                    'trailing_uppercase': trailing_uppercase,
                    'next_newline_distance': next_newline_distance,
                    'trailing_newline_count': trailing_newline_count,
                    'leading_2': leading_2,
                    'trailing_continue': trailing_continue,
                    'trailing_item_count': trailing_item_count
                }, ignore_index=True)
    data.to_csv('supervisedManagement_open.csv')


def create_feature_space_closing_model():
    data = pd.DataFrame()

    df = pd.read_csv('supervisedClosing.csv', header=None, names=['file', 'type', 'index'])
    files = list(df['file'])
    close_indices = df['index']
    types = df['type']

    for file_i, file in enumerate(files):
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        quantitatives = [m.start() for m in re.finditer('quantitative', file_text, re.IGNORECASE)]
        controls = [m.start() for m in re.finditer('controls', file_text, re.IGNORECASE)]

        y_close_index = close_indices[file_i]
        y_type = types[file_i]

        isQuantitativeInDocument = 1 if len(re.findall(r'quantitative[\s\S]+qualitative', file_text, re.IGNORECASE)) > 0else 0
        isControlsInDocument = 1 if len(re.findall(r'controls[\s\S]+procedures', file_text, re.IGNORECASE)) > 0 else 0


        total_text_length = len(file_text)
        for i, index in enumerate(quantitatives):
            trailing_text = file_text[index:index + 100].lower()
            trailing_text_large = file_text[index:index + 500].lower()
            leading_text_small_window = file_text[index - 40:index].lower()
            leading_text_medium_window = file_text[index - 100:index].lower()
            leading_text_large_window = file_text[index - 500:index].lower()

            trailing_qualitative = 1 if 'qualitative' in trailing_text else 0
            trailing_procedures = 1 if 'procedures' in trailing_text else 0
            trailing_continue = 1 if 'continue' in trailing_text_large else 0
            leading_item = 1 if 'item' in leading_text_small_window else 0
            leading_3 = 1 if '3' in leading_text_small_window else 0
            leading_4 = 1 if '4' in leading_text_small_window else 0
            leading_newline_count = len(leading_text_small_window.split('\n'))
            trailing_newline_count = len(trailing_text.split('\n'))
            leading_item_count = len(leading_text_large_window.split('item'))
            trailing_item_count = len(trailing_text_large.split('item'))
            trailing_uppercase = 1 if file_text[index:index + 50].isupper() else 0
            try:
                trailing_word_count = len(trailing_text_large.split('\n\n')[0].split(' '))
            except IndexError:
                trailing_word_count = 50

            try:
                leading_word_count = len(leading_text_medium_window.split('\n\n')[0].split(' '))
            except IndexError:
                leading_word_count = 50

            next_newline_distance = 100
            try:
                next_newline_distance = trailing_text.index('\n')
            except ValueError:
                pass
            next_double_newline_distance = 500
            try:
                next_double_newline_distance = trailing_text_large.index('\n\n')
            except ValueError:
                pass

            close_target = 1 if (int(y_close_index) == i) and (int(y_type) == 0) else 0
            data = data.append(
                {
                    'file': file,
                    'type': y_type,
                    'close_target': close_target,
                    'position': index / total_text_length,
                    'text_size': total_text_length,
                    'trailing_qualitative': trailing_qualitative,
                    'trailing_procedures': trailing_procedures,
                    'leading_item': leading_item,
                    'leading_newline_count': leading_newline_count,
                    'leading_item_count': leading_item_count,
                    'n_index': i,
                    'trailing_uppercase': trailing_uppercase,
                    'next_newline_distance': next_newline_distance,
                    'trailing_newline_count': trailing_newline_count,
                    'leading_3': leading_3,
                    'leading_4': leading_4,
                    'trailing_continue': trailing_continue,
                    'trailing_item_count': trailing_item_count,
                    'isQuantitativeInDocument': isQuantitativeInDocument,
                    'isControlsInDocument': isControlsInDocument,
                    'trailing_word_count': trailing_word_count,
                    'leading_word_count': leading_word_count,
                    'next_double_newline_distance': next_double_newline_distance
                }, ignore_index=True)

        for i, index in enumerate(controls):
            trailing_text = file_text[index:index + 100].lower()
            trailing_text_large = file_text[index:index + 500].lower()
            leading_text_small_window = file_text[index - 40:index].lower()
            leading_text_medium_window = file_text[index - 100:index].lower()
            leading_text_large_window = file_text[index - 500:index].lower()

            trailing_qualitative = 1 if 'qualitative' in trailing_text else 0
            trailing_procedures = 1 if 'procedures' in trailing_text else 0
            trailing_continue = 1 if 'continue' in trailing_text_large else 0
            leading_item = 1 if 'item' in leading_text_small_window else 0
            leading_3 = 1 if '3' in leading_text_small_window else 0
            leading_4 = 1 if '4' in leading_text_small_window else 0
            leading_newline_count = len(leading_text_small_window.split('\n'))
            trailing_newline_count = len(trailing_text.split('\n'))
            leading_item_count = len(leading_text_large_window.split('item'))
            trailing_item_count = len(trailing_text_large.split('item'))
            trailing_uppercase = 1 if file_text[index:index + 50].isupper() else 0

            try:
                trailing_word_count = len(trailing_text_large.split('\n\n')[0].split(' '))
            except IndexError:
                trailing_word_count = 50

            next_newline_distance = 100
            next_double_newline_distance = 500
            try:
                next_newline_distance = trailing_text.index('\n')
            except ValueError:
                pass
            try:
                next_double_newline_distance = trailing_text_large.index('\n\n')
            except ValueError:
                pass

            try:
                leading_word_count = len(leading_text_medium_window.split('\n\n')[0].split(' '))
            except IndexError:
                leading_word_count = 50

            close_target = 1 if (int(y_close_index) == i) and (int(y_type) == 1) else 0
            data = data.append(
                {
                    'file': file,
                    'type': y_type,
                    'close_target': close_target,
                    'position': index / total_text_length,
                    'text_size': total_text_length,
                    'trailing_qualitative': trailing_qualitative,
                    'trailing_procedures': trailing_procedures,
                    'leading_item': leading_item,
                    'leading_newline_count': leading_newline_count,
                    'leading_item_count': leading_item_count,
                    'n_index': i,
                    'trailing_uppercase': trailing_uppercase,
                    'next_newline_distance': next_newline_distance,
                    'trailing_newline_count': trailing_newline_count,
                    'leading_3': leading_3,
                    'leading_4': leading_4,
                    'trailing_continue': trailing_continue,
                    'trailing_item_count': trailing_item_count,
                    'isQuantitativeInDocument': isQuantitativeInDocument,
                    'isControlsInDocument': isControlsInDocument,
                    'trailing_word_count': trailing_word_count,
                    'next_double_newline_distance': next_double_newline_distance,
                    'leading_word_count': leading_word_count
                }, ignore_index=True)

    data.to_csv('supervised_Close.csv')


def testing():
    df = pd.read_csv('supervisedManagement.csv', header=None, names=['file', 'open', 'close'])
    files = list(df['file'])
    open_indices = df['open']

    for i, file in enumerate(files):
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        managements = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]
        m = managements[open_indices[i]]
        text = file_text[m - 50: m + 300]
        output_file = open('IsThisTheSection.txt', 'w')

        output_file.write(text)
        output_file.close()

        os.startfile('IsThisTheSection.txt')
        input()


def testing2():
    df = pd.read_csv('supervisedClosing.csv', header=None, names=['file', 'type', 'index'])
    files = list(df['file'])
    indices = df['index']
    types = df['type']

    for i, file in enumerate(files):
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        t = int(types[i])
        if t == 0:
            anchors = [m.start() for m in re.finditer('quantitative', file_text, re.IGNORECASE)]
        else:
            anchors = [m.start() for m in re.finditer('controls', file_text, re.IGNORECASE)]

        a = anchors[int(indices[i])]
        text = file_text[a - 50: a + 300]
        output_file = open('IsThisTheSection.txt', 'w')

        output_file.write(text)
        output_file.close()

        os.startfile('IsThisTheSection.txt')
        input()


def train_open():
    df = pd.read_csv('supervisedManagement_open.csv', index_col=0)

    features = ['position', 'text_size', 'trailing_analysis', 'trailing_discussion', 'leading_item',
            'leading_newline_count', 'leading_item_count', 'n_management', 'trailing_uppercase',
            'next_newline_distance', 'trailing_newline_count', 'leading_2', 'trailing_continue', 'trailing_item_count']

    X = df[features]
    y = df['open_target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)
    # logistic = GradientBoostingClassifier()
    parameters = {'max_depth': [4, 5, 6, 7],
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

    with open('../rf_quarterly_open.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # explainer = shap.TreeExplainer(logistic)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X)

    # files = list(df['file'])
    # open_indices = df['open_target']
    # predicted = logistic.predict(X)
    # probs = logistic.predict_proba(X)
    # m_indices = list(df['n_management'])
    #
    #
    # for i, file in enumerate(files):
    #     if predicted[i] == open_indices[i]:
    #         continue
    #     file_text = open(f'../Data/10-Q Sample//{file}').read()
    #     managements = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]
    #     m = managements[m_indices[i]]
    #     output_file = open('IsThisTheSection.txt', 'w')
    #     file_text = f'{file_text[:m]}$%$%$%%${file_text[m:]}'
    #     modified_file_text = file_text[m - 600: m + 600]
    #     output_file.write(modified_file_text)
    #     output_file.close()
    #
    #     os.startfile('IsThisTheSection.txt')
    #
    #     print(f'Predicted: {predicted[i]}\nActual   : {open_indices[i]}\nProbability: {probs[i]}')
    #     print(X.iloc[i])
    #     input('Continue?')


def train_close():
    df = pd.read_csv('supervised_Close.csv', index_col=0)

    features = ['position', 'text_size', 'trailing_qualitative', 'trailing_procedures', 'leading_item',
                'leading_newline_count', 'leading_item_count', 'n_index', 'trailing_uppercase',
                'next_newline_distance', 'trailing_newline_count', 'leading_3', 'leading_4',
                'trailing_continue', 'trailing_item_count', 'isQuantitativeInDocument', 'isControlsInDocument',
                'trailing_word_count', 'next_double_newline_distance', 'leading_word_count']


    X = df[features]
    y = df['close_target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)
    parameters = {'max_depth': [4, 5, 6, 7],
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

    with open('../rf_quarterly_close.pkl', 'wb') as f:
        pickle.dump(clf, f)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)

    files = list(df['file'])
    types = list(df['type'])
    close_indices = df['close_target']
    predicted = clf.predict(X)
    probs = clf.predict_proba(X)
    m_indices = list(df['n_index'])


    for i, file in enumerate(files):
        if predicted[i] == close_indices[i]:
            continue
        file_text = open(f'../Data/10-Q Sample//{file}').read()
        if types[i] == 0:
            managements = [m.start() for m in re.finditer('quantitative', file_text, re.IGNORECASE)]
        else:
            managements = [m.start() for m in re.finditer('controls', file_text, re.IGNORECASE)]
        m = managements[int(m_indices[i])]
        output_file = open('IsThisTheSection.txt', 'w')
        file_text = f'{file_text[:m]}$%$%$%%${file_text[m:]}'
        modified_file_text = file_text[m - 600: m + 600]
        output_file.write(modified_file_text)
        output_file.close()

        os.startfile('IsThisTheSection.txt')

        print(f'Predicted: {predicted[i]}\nActual   : {close_indices[i]}\nProbability: {probs[i]}')
        print(X.iloc[i])
        input('Continue?')


def neural_network_open():
    df = pd.read_csv('supervisedManagement_open.csv', index_col=0)

    negative_count = len(df[df['open_target'] == 0])
    positive_count = len(df[df['open_target'] == 1])
    features = ['position', 'text_size', 'trailing_analysis', 'trailing_discussion', 'leading_item',
                'leading_newline_count', 'leading_item_count', 'n_management', 'trailing_uppercase',
                'next_newline_distance', 'trailing_newline_count', 'leading_2', 'trailing_continue',
                'trailing_item_count']

    X = df[features]
    y = df['open_target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128 * 3, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(64 * 1, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),
                  loss=tf.keras.losses.BinaryCrossentropy())
    log_dir = f'logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(x=X_train,
              y=y_train,
              epochs=300,
              validation_data=(X_test, y_test),
              callbacks=[tensorboard_callback],
              batch_size=15,
              class_weight={0: 1,
                            1: int(negative_count / positive_count)})

    # print(model.predict_classes(X_test))
    y_train_predicted = (model.predict(X_train) > 0.5).astype("int32")
    y_test_predicted = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_true=y_train, y_pred=y_train_predicted, normalize='true')
    f = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.show()

    cm = confusion_matrix(y_true=y_test, y_pred=y_test_predicted, normalize='true')
    f = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.show()

    # confusion_matrix()

if __name__ == "__main__":
    # get_supervised_mda('test.csv')
    # convert_item_to_management('test.csv')
    # create_feature_space_opening_model()
    train_open()

    # convert_item_to_new_closing('test.csv')
    # testing2()
    # create_feature_space_closing_model()
    # train_close()
    # neural_network_open()



