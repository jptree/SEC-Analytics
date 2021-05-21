import os
import pickle
import re
import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shap
import matplotlib.pyplot as plt

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


def create_feature_space_opening_model():
    data = pd.DataFrame(columns=['file', 'open_target', 'position', 'text_size', 'trailing_analysis',
                                 'trailing_discussion', 'leading_item', 'leading_newline_count', 'leading_item_count',
                                 'n_management'])

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
            leading_text_small_window = file_text[index - 40:index].lower()
            leading_text_large_window = file_text[index - 500:index].lower()

            trailing_discussion = 1 if 'discussion' in trailing_text else 0
            trailing_analysis = 1 if 'analysis' in trailing_text else 0
            leading_item = 1 if 'item' in trailing_text else 0
            leading_newline_count = len(leading_text_small_window.split('\n'))
            leading_item_count = len(leading_text_large_window.split('\n'))

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
                    'n_management': i
                }, ignore_index=True)
    data.to_csv('supervisedManagement_open.csv')


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


def train_open():
    df = pd.read_csv('supervisedManagement_open.csv', index_col=0)

    X = df[['position', 'text_size', 'trailing_analysis', 'trailing_discussion', 'leading_item',
            'leading_newline_count', 'leading_item_count', 'n_management']]
    y = df['open_target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)
    # logistic = GradientBoostingClassifier()
    logistic = RandomForestClassifier(class_weight='balanced', max_depth=4, random_state=699, n_estimators=1000)

    logistic.fit(X_train, y_train)


    plot_confusion_matrix(logistic, X_train, y_train, normalize='true', cmap='Blues')
    plt.show()
    plot_confusion_matrix(logistic, X_test, y_test, normalize='true', cmap='Blues')
    plt.show()

    explainer = shap.TreeExplainer(logistic)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


if __name__ == "__main__":
    # get_supervised_mda('test.csv')
    # convert_item_to_management('test.csv')
    # create_feature_space_opening_model()
    train_open()