import os
import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import shap

PATH = "D:\\Python\\Projects\\FinTech\\SEC-Analytics\\Data\\10-K Sample"
# PATH = "C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample"
REGEX_10K = r"(Item[\s]+?7\.[\s\S]*?)(Item[\s]+?8\.)"

pd.options.display.width = 0
# pd.set_printoptions(max_rows=200, max_columns=10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)

def get_index(raw_text, text_positions, is_open = True):
    if is_open:
        response_text = f'Is this the true opening label?:\n Nothing: Proceed 1\n 0: Yes\n 1: Go back 1\n 5: Proceed 5\n'
    else:
        response_text = f'Is this the true closing label?:\n Nothing: Proceed 1\n 0: Yes\n 1: Go back 1\n 5: Proceed 5\n'

    i = 0
    while i < len(text_positions):
        position = text_positions[i]
        print('\n' * 100)
        print(f'{raw_text[max(0, position - 300):position]}$%$%$%%$%${raw_text[position:position + 4]}$%$%$%%$%${raw_text[position + 4:min(len(raw_text), position + 300)]}')
        response = input(response_text)

        if response == '':
            i += 1
        elif response == '5':
            i += 5
        elif response == '1':
            i -= 1
        elif response == '0':
            if input('Are you sure? Yes: 0') == '0':
                return i

    return None

def get_supervised_old():
    with open('supervised.csv', mode='a', encoding='utf-8') as file:
        _, _, file_names = next(
            os.walk(PATH))

        for file_name in file_names:
            file_path = f'{PATH}\\{file_name}'
            file_text = open(file_path).read()
            indices = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
            opening_index = get_index(file_text, indices, is_open=True)
            closing_index = get_index(file_text, indices, is_open=False)
            file.write(f'{file_name},{opening_index},{closing_index}\n')

    file.close()

def get_supervised(new_file_name):
    already_done = open('supervised3.csv').read()

    _, _, file_names = next(
        os.walk(PATH))
            # f'C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample'))

    for file_name in file_names:
        if file_name in already_done:
            print('file already done!')
            continue
        file_path = f'{PATH}\\{file_name}'
        file_text = open(file_path).read()
        match = re.findall(REGEX_10K, file_text, re.IGNORECASE)

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


def data_mine(X_train, X_test, y_train, y_test):
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="rbf", C=0.025, probability=True),
        # NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        # QuadraticDiscriminantAnalysis()
    ]

    # Logging for Visual Comparison
    log_cols = ["Classifier", "Accuracy", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    for clf in classifiers:
        clf.fit(X_train, y_train)
        name = clf.__class__.__name__

        print("=" * 30)
        print(name)

        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        train_predictions = clf.predict_proba(X_test)
        ll = log_loss(y_test, train_predictions)
        print("Log Loss: {}".format(ll))

        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)

        plot_roc_curve(clf, X_test, y_test)
        plt.show()

    print("=" * 30)


def test():
    df = pd.read_csv('supervised4.csv')
    # print(df)
    df = df[df['open'] != 'None']

    data = pd.DataFrame(columns=['position', 'near_management', 'trailing_period', 'trailing_7', 'y_open', 'y_close',
                                 'trailing_newline', 'leading_newline', 'total_size', 'leading_tab', 'leading_spaces',
                                 'regex_open', 'regex_close'])
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
        match = re.findall(REGEX_10K, file_text, re.IGNORECASE)

        for index, item in enumerate(items):

            if 'management' in file_text[item - 100: item + 100].lower():
                near_management = 1
            else:
                near_management = 0

            if '.' in file_text[item: item + 15]:
                trailing_period = 1
            else:
                trailing_period = 0

            if '7' in file_text[item: item + 15]:
                trailing_7 = 1
            else:
                trailing_7 = 0

            if '\n' in file_text[item: item + 15]:
                trailing_newline = 1
            else:
                trailing_newline = 0

            if '\n' in file_text[item - 5: item]:
                leading_newline = 1
            else:
                leading_newline = 0

            if '\t' in file_text[item - 5: item]:
                leading_tab = 1
            else:
                leading_tab = 0

            if '  ' in file_text[item - 5: item]:
                leading_spaces = 1
            else:
                leading_spaces = 0

            regex_open = 0
            regex_close = 0

            try:
                mda = match[-1][0]
                if file_text[item: item + len(mda)] == mda:
                    regex_open = 1
                    regex_close = 1
            except IndexError:
                e = 1

            data = data.append(
                {
                    'position': item / total_text_length,
                    'y_open': y_open[index],
                    'near_management': near_management,
                    'y_close': y_close[index],
                    'trailing_period': trailing_period,
                    'trailing_7': trailing_7,
                    'trailing_newline': trailing_newline,
                    'leading_newline': leading_newline,
                    'total_size': total_text_length,
                    'leading_tab': leading_tab,
                    'leading_spaces': leading_spaces,
                    'regex_open': regex_open,
                    'regex_close': regex_close
                }, ignore_index=True)

    logistic = RandomForestClassifier(class_weight='balanced', max_depth=8, random_state=69)
    # # logistic = DecisionTreeClassifier(class_weight='balanced')
    # # logistic = LogisticRegression(class_weight='balanced', random_state=69)

    X = data[['position', 'near_management', 'trailing_period', 'trailing_7', 'trailing_newline', 'leading_newline',
              'total_size', 'leading_tab', 'leading_spaces', 'regex_open']]
    # X = data[['regex_open']]

    y = data['y_open']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69, shuffle=True)
    # #
    # # # X = np.array(d[['position', 'near_management', 'trailing_period', 'trailing_7']])
    logistic.fit(X_train, y_train)
    #
    plot_confusion_matrix(logistic, X_train, y_train)
    plt.show()
    plot_confusion_matrix(logistic, X_test, y_test)
    plt.show()
    plot_roc_curve(logistic, X_train, y_train)
    plt.show()
    plot_roc_curve(logistic, X_test, y_test)
    plt.show()

    # explainer = shap.TreeExplainer(logistic)
    # shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X)

    data['predicted'] = logistic.predict(X)
    data['prob'] = [probs[1] for probs in logistic.predict_proba(X)]
    print(data)

    for index, row in data[~(data['predicted'] == data['y_open'])].iterrows():
        os.startfile(f'{PATH}\\{row.file}')
        print(row['predicted'], row['y_open'])
        input('continue?')

    # shap.plots.beeswarm(shap_values)

    # data_mine(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    # get_supervised('supervised4.csv')
    test()

