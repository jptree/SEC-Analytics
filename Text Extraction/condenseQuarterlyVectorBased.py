import re
import os
import csv
import time
import pickle
import multiprocessing
import os
from gensim.models import KeyedVectors
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pickle


OPEN_CLASSIFIER = pickle.load(open('../rf_quarterly_open_wv.pkl', 'rb'))
CLOSE_CLASSIFIER = pickle.load(open('../rf_quarterly_close_wv.pkl', 'rb'))
WV = KeyedVectors.load("mda.wordvectors", mmap='r')


def get_mda(file_text, clf_open, clf_close, wv, add_extra=False):
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

    try:
        index = open_probabilities.index(max(open_probabilities))
    except ValueError:
        return ''
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

        if np.isnan(features).any():
            continue
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


def mp_worker(args):
    file_name = str(args)
    clf_open = OPEN_CLASSIFIER
    clf_close = CLOSE_CLASSIFIER

    file_text = open(file_name).read()

    mda = get_mda(file_text, clf_open, clf_close, WV).replace('\n', '\\n')
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
        writer.writerow([result[0], result[1], result[2], result[3]])

    print('\n')



    end = time.time()
    print(f'Multiple Threads: {round(end - start, 2)} seconds')

    p.close()


if __name__ == '__main__':
    path_data_list = ['D:\\SEC Filing Data\\10-X_C_2006-2010',
                      'D:\\SEC Filing Data\\10-X_C_2011-2015',
                      'D:\\SEC Filing Data\\10-X_C_2016-2018']

    for path_data in path_data_list:
        _, years, _ = next(os.walk(path_data))

        for year in years:
            if int(year) < 2017:
                continue
            _, quarters, _ = next(os.walk(f'{path_data}\\{year}'))
            for quarter in quarters:
                if int(year) == 2017 and (quarter == 'QTR1' or quarter == 'QTR2'):
                    continue
                print(f'Working on {quarter} of {year}...')
                output_directory = f'Extracted\\Quarterly\\10-Q_{year}_{quarter}.csv'
                all_directories = []
                _, _, directories = next(os.walk(f'{path_data}\\{year}\\{quarter}'))
                for directory in directories:
                    if '_10-Q_' in directory:
                        all_directories += [f'{path_data}\\{year}\\{quarter}\\' + directory]

                mp_handler(all_directories, 4, output_directory)