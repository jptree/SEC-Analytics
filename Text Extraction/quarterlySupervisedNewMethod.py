import os
import pickle
import re
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        print('Open: ', open_index)
        print('Close: ', close_index)

    except ValueError:
        return ''

    if open_index > close_index:
        return ''
    else:
        # return file_text[items[open_index]: items[close_index]]
        return file_text[open_indices[open_index]: close_indices[close_index]].replace('\n', '\\n')


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

        mda_text = get_mda(OPEN_CLASSIFIER, CLOSE_CLASSIFIER, file_text)

        modified_file = open('IsThisTheSection.txt', 'w')
        modified_file.write(mda_text.replace('\\n', '\n'))
        modified_file.close()

        output_file = open('FileInQuestion.txt', 'w')
        output_file.write(file_text.replace('\\n', '\n'))
        output_file.close()

        os.startfile('FileInQuestion.txt')
        os.startfile('IsThisTheSection.txt')

        is_correct = input('<ENTER> if the section correct.')

        writer = csv.writer(open(new_file_name, 'a', newline=''))
        if is_correct == '':
            writer.writerow([file_name, mda_text])
        else:
            output_file = open('FileInQuestion.txt', 'r').read().replace('\n', '\\n')
            writer.writerow([file_name, output_file])


def create_classification(file):
    df = pd.read_csv(file, header=None, names=['file', 'mda'])
    df = df[df['mda'] != '-9']

    files = list(df['file'])
    mdas = list(df['mda'])

    data = pd.DataFrame(columns=['file', 'open_target', 'close_target', 'trailing_management', 'trailing_discussion',
                                 'trailing_item', 'trailing_2', 'trailing_3', 'trailing_4', 'trailing_controls',
                                 'trailing_procedures', 'trailing_quantitative', 'trailing_qualitative'])
    vectorizer = TfidfVectorizer()
    for i, file_name in enumerate(files):
        file_text = open(f'{PATH}\\{file_name}').read()
        newlines = [m.start() for m in re.finditer('discussion', file_text, re.IGNORECASE)]

        mda_text = mdas[i].replace('\\n', '\n')
        mda_open = mda_text[:500]
        mda_close = mda_text[-500:]

        total_text_length = len(mda_text)

        for index in newlines:
            index_text_open = file_text[index:index + 500]
            index_text_close = file_text[index - 500:index]


            vectorizer.fit([mda_open, index_text_open])
            open_similarity = cosine_similarity(vectorizer.transform([mda_open]), vectorizer.transform([index_text_open]))[0][0]

            vectorizer.fit([mda_close, index_text_close])
            close_similarity = cosine_similarity(vectorizer.transform([mda_close]), vectorizer.transform([index_text_close]))[0][0]

            open_target = 0
            close_target = 0

            if open_similarity > 0.9:
                open_target = 1
            if close_similarity > 0.9:
                close_target = 1

            trailing_text = mda_text[index:index + 100].lower()
            # trailing_text_300 = mda_text[index:index + 300].lower()

            trailing_management = 1 if 'management' in trailing_text else 0
            trailing_discussion = 1 if 'discussion' in trailing_text else 0
            trailing_item = 1 if 'item' in trailing_text else 0
            trailing_2 = 1 if '2' in trailing_text else 0
            trailing_3 = 1 if '3' in trailing_text else 0
            trailing_4 = 1 if '4' in trailing_text else 0
            trailing_controls = 1 if 'controls' in trailing_text else 0
            trailing_procedures = 1 if 'procedures' in trailing_text else 0
            trailing_quantitative = 1 if 'quantitative' in trailing_text else 0
            trailing_qualitative = 1 if 'qualitative' in trailing_text else 0

            data = data.append(
                {
                    'file': file,
                    'open_target': open_target,
                    'close_target': close_target,
                    'position': index / total_text_length,
                    'text_size': total_text_length,
                    'trailing_management': trailing_management,
                    'trailing_discussion': trailing_discussion,
                    'trailing_item': trailing_item,
                    'trailing_2': trailing_2,
                    'trailing_3': trailing_3,
                    'trailing_4': trailing_4,
                    'trailing_controls': trailing_controls,
                    'trailing_procedures': trailing_procedures,
                    'trailing_quantitative': trailing_quantitative,
                    'trailing_qualitative': trailing_qualitative
                }, ignore_index=True)



if __name__ == "__main__":
    # get_supervised_mda('NewExtractionSupervised.csv')
    p = 'D:\\SEC Filing Data\\10-X_C_1993-2000\\1997\\QTR1\\19970102_10-Q_edgar_data_58636_0000058636-97-000001_1.txt'
    file_text = open(p, 'r').read()
    mda = get_mda(OPEN_CLASSIFIER, CLOSE_CLASSIFIER, file_text)
    modified_file_text = ''
    for index, s in enumerate(re.split(r'item', file_text, flags=re.IGNORECASE)):
        modified_file_text += f'{s} item $%$ {index} $%$'

    output_file = open('FileInQuestion.txt', 'w')
    output_file.write(modified_file_text)
    output_file.close()

    os.startfile('FileInQuestion.txt')

    # create_classification('NewExtractionSupervised.csv')


