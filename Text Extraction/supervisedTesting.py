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

PATH = "D:\\Python\\Projects\\FinTech\\SEC-Analytics\\Data\\10-K Sample"
# PATH = "C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample"
REGEX_10K = r"(Item[\s]+?7\.[\s\S]*?)(Item[\s]+?8\.)"

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
    _, _, file_names = next(
        os.walk(PATH))
            # f'C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Data\\10-K Sample'))

    for file_name in file_names:
        file_path = f'{PATH}\\{file_name}'
        file_text = open(file_path).read()
        match = re.findall(REGEX_10K, file_text, re.IGNORECASE)

        items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]

        opening_index = None
        closing_index = None
        try:
            mda = match[-1][0]

            for index, item in enumerate(items):
                # Check for beginning index
                if file_text[item: item + len(mda)] == mda:
                    opening_index = index
                    closing_index = min(range(len(items)), key=lambda i: abs(items[i] - (item + len(mda))))


            os.startfile(file_path)
            output_file = open('output.txt', 'w')
            output_file.write(mda)
            os.startfile('output.txt')
            output_file.close()
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

def test():
    df = pd.read_csv('supervised2.csv')
    # print(df)
    df = df[df['open'] != 'None']

    data = pd.DataFrame(columns=['position', 'near_management', 'trailing_period', 'trailing_7', 'y_open', 'y_close',
                                 'trailing_newline'])
    # X_open = pd.DataFrame()
    y_open = []
    # X_close = pd.DataFrame()
    y_close = []
    for index, row in df.iterrows():
        file_text = open(f'{PATH}\\{row.file}').read()
        items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]
        close_items = [0] * len(items)
        close_items[int(row.close)] = 1
        open_items = [0] * len(items)
        open_items[int(row.close)] = 1

        y_close += close_items
        y_open += open_items

        total_text_length = len(file_text)

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

            data = data.append(
                {
                    'position': item / total_text_length,
                    'y_open': y_open[index],
                    'near_management': near_management,
                    'y_close': y_close[index],
                    'trailing_period': trailing_period,
                    'trailing_7': trailing_7,
                    'trailing_newline': trailing_newline
                }, ignore_index=True)

    logistic = RandomForestClassifier(class_weight='balanced', max_depth=5, random_state=69)
    # logistic = DecisionTreeClassifier(class_weight='balanced')
    # logistic = LogisticRegression(class_weight='balanced', random_state=69)
    X = data[['position', 'near_management', 'trailing_period', 'trailing_7', 'trailing_newline']]
    y = data['y_open']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

    # X = np.array(d[['position', 'near_management', 'trailing_period', 'trailing_7']])
    logistic.fit(X_train, y_train)

    plot_confusion_matrix(logistic, X_train, y_train)
    plt.show()
    plot_confusion_matrix(logistic, X_test, y_test)
    plt.show()
    plot_roc_curve(logistic, X_train, y_train)
    plt.show()
    plot_roc_curve(logistic, X_test, y_test)
    plt.show()

if __name__ == "__main__":
    get_supervised('supervised3.csv')

