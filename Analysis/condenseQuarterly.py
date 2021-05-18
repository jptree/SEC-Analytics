import re
import os
import csv
import time
import pickle
import multiprocessing
import pathlib
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble.forest import RandomForestClassifier

PATH = '\\'.join(str(pathlib.Path().absolute()).split('\\')[:-1])
DATA_PATH= ''
# REGEX_10K = r"(Item[\s]+?7\.[\s\S]*?)(Item[\s]+?8\.)"
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
    except ValueError:
        return ''

    if open_index > close_index:
        return ''
    else:
        # return file_text[items[open_index]: items[close_index]]
        return file_text[open_indices[open_index]: close_indices[close_index]].replace('\n', '\\n')


def mp_worker(args):
    file_name = str(args)
    clf_open = OPEN_CLASSIFIER
    clf_close = CLOSE_CLASSIFIER

    file_text = open(file_name).read()

    mda = get_mda(clf_open, clf_close, file_text)
    cik = file_name.split('_')[6]
    date = file_name.split('_')[2][-8:]

    return file_name, date, cik, mda


def mp_handler(file_names, n_pools, output_dir):
    p = multiprocessing.Pool(n_pools)

    start = time.time()
    writer = csv.writer(open(output_dir, 'a', newline=''))
    counter = 0
    for result in p.imap(mp_worker, file_names):
        counter += 1
        print(f'\rPercentage Complete: {round((counter / len(file_names)) * 100, 2)}%', end="", flush=True)
        writer.writerow([result[0],result[1],result[2],result[3]])
    print('\n')
    # with open(output_dir, 'w') as f:
    #     for result in p.imap(mp_worker, file_names):
    #         f.write(f'{result[0]},{result[1]},{result[2]},{result[3]}\n')
    # f.close()



    end = time.time()
    print(f'Multiple Threads: {round(end - start, 2)} seconds')

if __name__=='__main__':
    path_data = 'D:\\SEC Filing Data\\10-X_C_2006-2010'
    _, years, _ = next(os.walk(path_data))


    for year in years:
        _, quarters, _ = next(os.walk(f'{path_data}\\{year}'))
        for quarter in quarters:
            print(f'Working on {quarter} of {year}...')
            output_directory = f'Extracted\\Quarterly\\10-Q_{year}_{quarter}.csv'
            all_directories = []
            _, _, directories = next(os.walk(f'{path_data}\\{year}\\{quarter}'))
            for directory in directories:
                if '_10-Q_' in directory:
                    all_directories += [f'{path_data}\\{year}\\{quarter}\\' + directory]

            # print(all_directories)
            mp_handler(all_directories, 4, output_directory)


    # print(OPEN_CLASSIFIER.feature_importances_)

    # import pandas as pd
    # print(pd.read_csv('Extracted/Quarterly/10-Q_2006_QTR1.csv'))