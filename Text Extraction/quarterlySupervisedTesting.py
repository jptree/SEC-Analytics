import pickle
import os
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import cross_validate

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow.keras as keras
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns

PATH = "C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-Q Sample"
# PATH = "D:\\Python\\Projects\\FinTech\\SEC-Analytics\\Data\\10-Q Sample"
# PATH = "C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample"
REGEX_10K = r"(Item[\s]+?7\.[\s\S]*?)(Item[\s]+?8\.)"
REGEX_10Q = r"(Item[\s]+?2\.[\s\S]*?)(Item[\s]+?3\.)"

pd.options.display.width = 0
# pd.set_printoptions(max_rows=200, max_columns=10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)



def get_supervised(new_file_name):
    already_done = []
    already_done = open('QuarterlySupervised.csv').read()

    _, _, file_names = next(
        os.walk(PATH))
    # f'C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample'))

    for file_name in file_names:
        if file_name in already_done:
            print('file already done!')
            continue
        file_path = f'{PATH}\\{file_name}'
        file_text = open(file_path).read()
        match = re.findall(REGEX_10Q, file_text, re.IGNORECASE)

        items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]

        opening_index = None
        closing_index = None
        try:
            mda = match[-1][0]

            modified_file_text = ''
            for index, s in enumerate(re.split(r'item', file_text, flags=re.IGNORECASE)):
                modified_file_text += f'{s} item $%$ {index} $%$'

            for index, item in enumerate(items):
                # Check for beginning index
                if file_text[item: item + len(mda)] == mda:
                    opening_index = index
                    closing_index = min(range(len(items)), key=lambda i: abs(items[i] - (item + len(mda))))

            modified_file = open('modified.txt', 'w')
            modified_file.write(modified_file_text)
            modified_file.close()
            os.startfile('modified.txt')

            # os.startfile(file_path)
            output_file = open('output.txt', 'w')
            output_file.write(mda)
            output_file.close()
            os.startfile('output.txt')
        except Exception as e:
            print(e, file_name)

        print(f'\n{file_name}\nOpening: {opening_index}\nClosing: {closing_index}')
        actual_open = input('What is the open index? :')
        actual_close = input('What is the close index? :')

        if actual_open == '':
            actual_open = opening_index

        if actual_close == '':
            actual_close = closing_index


        supervised_file = open(new_file_name, 'a')
        supervised_file.write(f'{file_name},{actual_open},{actual_close}\n')
        supervised_file.close()


def get_classification_quarterly():
    df = pd.read_csv('QuarterlySupervised.csv')
    # print(df)
    df = df[df['open'] != 'None']
    df = df[df['close'] != 'None']

    data = pd.DataFrame(columns=['file', 'true_location_open', 'true_location_close', 'input_location', 'position', 'trailing_management',
                                 'trailing_period', 'trailing_2', 'y_open', 'y_close', 'trailing_newline',
                                 'leading_newline', 'total_size', 'leading_tab', 'leading_spaces', 'regex_open',
                                 'regex_close', 'trailing_financial', 'leading_words', 'leading_see',
                                 'leading_text', 'leading_double_newline', 'is_uppercase', 'trailing_omission', 'leading_with',
                                 'leading_table_of_contents', 'leading_newline_count', 'trailing_analysis', 'trailing_3', 'trailing_quantitative'])
    # X_open = pd.DataFrame()

    # X_close = pd.DataFrame()

    for df_index, row in df.iterrows():
        y_close = []
        y_open = []

        file_text = open(f'{PATH}\\{row.file}').read()
        items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
        close_items = [0] * len(items)
        close_items[int(row.close)] = 1
        open_items = [0] * len(items)
        open_items[int(row.open)] = 1

        y_close += close_items
        y_open += open_items

        total_text_length = len(file_text)
        match = re.findall(REGEX_10Q, file_text, re.IGNORECASE)

        for index, item in enumerate(items):

            # Probably shitty? Make smaller window.
            if 'management' in file_text[item: item + 80].lower():
                # if 'management' in file_text[item - 100: item + 100].lower():
                trailing_management = 1
            else:
                trailing_management = 0

            if 'quantitative' in file_text[item: item + 80].lower():
                trailing_quantitative = 1
            else:
                trailing_quantitative = 0

            if 'analysis' in file_text[item: item + 80].lower():
                # if 'management' in file_text[item - 100: item + 100].lower():
                trailing_analysis = 1
            else:
                trailing_analysis = 0

            if '.' in file_text[item: item + 20]:
                trailing_period = 1
            else:
                trailing_period = 0

            if '2' in file_text[item: item + 20]:
                trailing_2 = 1
            else:
                trailing_2 = 0

            if '3' in file_text[item: item + 20]:
                trailing_3 = 1
            else:
                trailing_3 = 0

            if '\n' in file_text[item: item + 100]:
                trailing_newline = 1
            else:
                trailing_newline = 0

            if file_text[item: item + 15].isupper():
                is_uppercase = 1
            else:
                is_uppercase = 0

            if '\n' in file_text[item - 5: item]:
                leading_newline = 1
            else:
                leading_newline = 0

            if '\n\n' in file_text[item - 5: item]:
                leading_double_newline = 1
            else:
                leading_double_newline = 0

            if '\t' in file_text[item - 5: item]:
                leading_tab = 1
            else:
                leading_tab = 0

            if '  ' in file_text[item - 5: item]:
                leading_spaces = 1
            else:
                leading_spaces = 0

            if (len(file_text[item - 40: item].split(' ')) > 2) and ((sum([len(w) for w in file_text[item - 40: item].split(' ')]) / len(file_text[item - 40: item].split(' '))) > 2):
                leading_words = 1
            else:
                leading_words = 0

            if 'see' in file_text[item - 10: item].lower():
                leading_see = 1
            else:
                leading_see = 0

            if 'with' in file_text[item - 10: item].lower():
                leading_with = 1
            else:
                leading_with = 0

            if len(re.findall(r'\w+', file_text[item - 5: item])) > 0:
                leading_text = 1
            else:
                leading_text = 0

            if 'financial' in file_text[item: item + 30].lower():
                trailing_financial = 1
            else:
                trailing_financial = 0

            if ('applicable' in file_text[item: item + 300].lower()) or ('omitted' in file_text[item: item + 300].lower()):
                trailing_omission = 1
            else:
                trailing_omission = 0

            if 'table of contents' in file_text[item - 50: item].lower():
                leading_table_of_contents = 1
            else:
                leading_table_of_contents = 0

            leading_newline_count = len(file_text[item - 20: item].split('\n'))

            regex_open = 0
            regex_close = 0

            try:
                mda = match[-1][0]
                if file_text[item: item + len(mda)] == mda:
                    regex_open = 1
                    # regex_close = 1
            except IndexError:
                e = 1

            try:
                mda = match[-1][0]
                if file_text[item - len(mda): item] == mda:
                    # regex_open = 1
                    regex_close = 1
            except IndexError:
                e = 1

            data = data.append(
                {
                    'file': row.file,
                    'position': item / total_text_length,
                    'y_open': y_open[index],
                    'trailing_management': trailing_management,
                    'y_close': y_close[index],
                    'trailing_period': trailing_period,
                    'trailing_2': trailing_2,
                    'trailing_newline': trailing_newline,
                    'leading_newline': leading_newline,
                    'total_size': total_text_length,
                    'leading_tab': leading_tab,
                    'leading_spaces': leading_spaces,
                    'regex_open': regex_open,
                    'regex_close': regex_close,
                    'trailing_financial': trailing_financial,
                    'true_location_open': row.open,
                    'true_location_close': row.close,
                    'input_location': index,
                    'leading_words': leading_words,
                    'leading_see': leading_see,
                    'leading_text': leading_text,
                    'leading_double_newline': leading_double_newline,
                    'is_uppercase': is_uppercase,
                    'trailing_omission': trailing_omission,
                    'leading_with': leading_with,
                    'leading_table_of_contents': leading_table_of_contents,
                    'leading_newline_count': leading_newline_count,
                    'trailing_analysis': trailing_analysis,
                    'trailing_3': trailing_3,
                    'trailing_quantitative': trailing_quantitative
                }, ignore_index=True)

    data.to_csv('quarterlyClassification.csv')


def open_train_quarterly():
    df = pd.read_csv('quarterlyClassification.csv', index_col=0)
    df = df[df['y_open'] != 'None']

    df['true_class'] = np.where(df['input_location'] == df['true_location_open'], 1, 0)
    # df['regex_class'] = np.where(df['regex_open'] == df['true_location_open'], 1, 0)

    X = df[list(df.columns)]
    y = df['y_open']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)

    # cm = confusion_matrix(y_true=y, y_pred=X['regex_open'])
    cm = confusion_matrix(y_true=y_train, y_pred=X_train['regex_open'], normalize='true')
    f = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.show()

    cm = confusion_matrix(y_true=y_test, y_pred=X_test['regex_open'], normalize='true')
    f = sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()

    # logistic = GradientBoostingClassifier(max_depth=12, random_state=699)
    # # logistic = DecisionTreeClassifier(class_weight='balanced')
    # logistic = LogisticRegression(class_weight='balanced', random_state=69)
    #
    X = df[['position', 'trailing_management', 'trailing_period', 'trailing_2', 'trailing_newline', 'leading_newline',
            'total_size', 'regex_open', 'trailing_analysis']]

    y = df['y_open'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)
    logistic = RandomForestClassifier(class_weight='balanced', max_depth=5, random_state=699, n_estimators=1000)

    # # # # X = np.array(d[['position', 'trailing_management', 'trailing_period', 'trailing_7']])
    logistic.fit(X_train, y_train)

    with open('../open_quarterly_random_forest.pkl', 'wb') as f:
        pickle.dump(logistic, f)


    plot_confusion_matrix(logistic, X_train, y_train, normalize='true', cmap='Blues')
    plt.show()
    plot_confusion_matrix(logistic, X_test, y_test, normalize='true', cmap='Blues')
    plt.show()

    explainer = shap.TreeExplainer(logistic)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


def close_train_quarterly():
    df = pd.read_csv('quarterlyClassification.csv', index_col=0)
    df = df[df['y_close'] != 'None']

    df['true_class'] = np.where(df['input_location'] == df['true_location_open'], 1, 0)
    # df['regex_class'] = np.where(df['regex_open'] == df['true_location_open'], 1, 0)

    X = df[list(df.columns)]
    y = df['y_close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)

    # cm = confusion_matrix(y_true=y, y_pred=X['regex_open'])
    cm = confusion_matrix(y_true=y_train, y_pred=X_train['regex_close'], normalize='true')
    f = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.show()

    cm = confusion_matrix(y_true=y_test, y_pred=X_test['regex_close'], normalize='true')
    f = sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()

    # logistic = GradientBoostingClassifier(max_depth=12, random_state=699)
    # # logistic = DecisionTreeClassifier(class_weight='balanced')
    # logistic = LogisticRegression(class_weight='balanced', random_state=69)



    # X = df[['position', 'trailing_management', 'trailing_period', 'trailing_3', 'trailing_newline',
    #         'leading_newline', 'total_size', 'leading_tab', 'leading_spaces', 'trailing_financial', 'regex_close',
    #         'leading_see', 'leading_text', 'leading_double_newline', 'is_uppercase',
    #         'trailing_omission', 'leading_table_of_contents', 'leading_newline_count', 'trailing_analysis', 'trailing_quantitative']]

    X = df[['position', 'trailing_period', 'trailing_3', 'leading_newline', 'total_size', 'leading_tab', 'leading_spaces', 'regex_close',
            'trailing_quantitative']]


    y = df['y_close'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=699, shuffle=True)
    logistic = RandomForestClassifier(class_weight='balanced', max_depth=5, random_state=699, n_estimators=1000)

    logistic.fit(X_train, y_train)

    with open('../close_quarterly_random_forest.pkl', 'wb') as f:
        pickle.dump(logistic, f)


    plot_confusion_matrix(logistic, X_train, y_train, normalize='true', cmap='Blues')
    plt.show()
    plot_confusion_matrix(logistic, X_test, y_test, normalize='true', cmap='Blues')
    plt.show()

    explainer = shap.TreeExplainer(logistic)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    #
    # plot_roc_curve(logistic, X_test, y_test)
    # plt.show()
    # plot_roc_curve(logistic, X_train, y_train)
    # plt.show()
    #
    # print(classification_report(y_test, logistic.predict(X_test)))

    # print(logistic.predict_proba(X_test)[:, 1])
    # positive = df[df['y_close'] == 0]
    #
    # df = pd.DataFrame(list(logistic.predict_proba(positive[['position', 'trailing_period', 'trailing_3', 'leading_newline', 'total_size', 'leading_tab', 'leading_spaces', 'trailing_financial', 'regex_close',
    #         'trailing_quantitative']])[:, 1]), columns=['probability'])
    # df['probability'].hist(bins=100)
    # plt.show()


def test_file(file_directory, open_model, close_model):


if __name__ == "__main__":

    # close_training()

    # test()
    # import pickle
    # with open('opening_random_forest.pkl', 'rb') as f:
    #     clf = pickle.load(f)
    #
    # clf.predict(X_train)

    # get_supervised('QuarterlySupervised.csv')
    # test_quarterly()
    close_train_quarterly()
    # open_train_quarterly()