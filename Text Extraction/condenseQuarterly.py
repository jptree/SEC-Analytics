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
DATA_PATH = ''


OPEN_INDEPENDENT_VARIABLES = ['position', 'text_size', 'trailing_analysis', 'trailing_discussion', 'leading_item',
            'leading_newline_count', 'leading_item_count', 'n_management', 'trailing_uppercase',
            'next_newline_distance', 'trailing_newline_count', 'leading_2', 'trailing_continue', 'trailing_item_count']

CLOSE_INDEPENDENT_VARIABLES = ['position', 'text_size', 'trailing_qualitative', 'trailing_procedures', 'leading_item',
                'leading_newline_count', 'leading_item_count', 'n_index', 'trailing_uppercase',
                'next_newline_distance', 'trailing_newline_count', 'leading_3', 'leading_4',
                'trailing_continue', 'trailing_item_count', 'isQuantitativeInDocument', 'isControlsInDocument',
                'trailing_word_count', 'next_double_newline_distance', 'leading_word_count']


OPEN_CLASSIFIER = pickle.load(open('../rf_quarterly_open.pkl', 'rb'))
CLOSE_CLASSIFIER = pickle.load(open('../rf_quarterly_close.pkl', 'rb'))


def get_open_text_index(clf_open, file_text):
    open_probabilities = []
    managements = [m.start() for m in re.finditer('management', file_text, re.IGNORECASE)]
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

        data = {
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
        }

        open_features = []
        for f in OPEN_INDEPENDENT_VARIABLES:
            open_features.append(data[f])

        open_probabilities.append(clf_open.predict([open_features]))
    return managements[open_probabilities.index(max(open_probabilities))]


def get_close_text_index(clf_close, file_text):

    quantitatives = [m.start() for m in re.finditer('quantitative', file_text, re.IGNORECASE)]
    controls = [m.start() for m in re.finditer('controls', file_text, re.IGNORECASE)]
    isQuantitativeInDocument = 1 if len(re.findall(r'quantitative[\s\S]+qualitative', file_text, re.IGNORECASE)) > 0 else 0
    isControlsInDocument = 1 if len(re.findall(r'controls[\s\S]+procedures', file_text, re.IGNORECASE)) > 0 else 0

    total_text_length = len(file_text)
    close_probabilities = []
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

        data = {
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
            }

        close_features = []
        for f in CLOSE_INDEPENDENT_VARIABLES:
            close_features.append(data[f])

        close_probabilities.append(clf_close.predict([close_features]))

    try:
        quantitative_max = close_probabilities.index(max(close_probabilities))
    except ValueError:
        quantitative_max = None

    close_probabilities = []
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

        data = {
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
            }
        close_features = []
        for f in CLOSE_INDEPENDENT_VARIABLES:
            close_features.append(data[f])

        close_probabilities.append(clf_close.predict([close_features]))

    try:
        controls_max = close_probabilities.index(max(close_probabilities))
    except ValueError:
        controls_max = None


    if quantitative_max and controls_max:

        if quantitative_max > controls_max:
            return quantitatives[quantitative_max]
        else:
            return controls[controls_max]

    elif quantitative_max and not controls_max:
        return quantitatives[quantitative_max]
    elif controls_max and not quantitative_max:
        return controls[controls_max]
    else:
        return total_text_length


def mp_worker(args):
    file_name = str(args)
    clf_open = OPEN_CLASSIFIER
    clf_close = CLOSE_CLASSIFIER

    file_text = open(file_name).read()

    open_index = get_open_text_index(clf_open, file_text)
    close_index = get_close_text_index(clf_close, file_text)
    mda = file_text[open_index:close_index]
    cik = file_name.split('_')[6]
    date = file_name.split('_')[2][-8:]

    return file_name, date, cik, mda


def mp_handler(file_names, n_pools, output_dir):
    p = multiprocessing.Pool(n_pools)

    start = time.time()
    # writer = csv.writer(open(output_dir, 'a', newline=''))
    counter = 0
    for result in p.imap(mp_worker, file_names):
        counter += 1
        print(f'\rPercentage Complete: {round((counter / len(file_names)) * 100, 2)}%', end="", flush=True)
        # writer.writerow([result[0],result[1],result[2],result[3]])

        output_file = open('IsThisTheSection.txt', 'w')

        output_file.write(result[3])
        output_file.close()

        os.startfile('IsThisTheSection.txt')
        input('Continue?')
    print('\n')



    end = time.time()
    print(f'Multiple Threads: {round(end - start, 2)} seconds')


if __name__=='__main__':
    path_data = 'D:\\SEC Filing Data\\10-X_C_1993-2000'
    _, years, _ = next(os.walk(path_data))


    for year in years:
        if int(year) < 1996:
            continue
        _, quarters, _ = next(os.walk(f'{path_data}\\{year}'))
        for quarter in quarters:
            if quarter == 'QTR1' and year == '1996':
                continue
            print(f'Working on {quarter} of {year}...')
            output_directory = f'Extracted\\New\\Quarterly\\10-Q_{year}_{quarter}.csv'
            all_directories = []
            _, _, directories = next(os.walk(f'{path_data}\\{year}\\{quarter}'))
            for directory in directories:
                if '_10-Q_' in directory:
                    all_directories += [f'{path_data}\\{year}\\{quarter}\\' + directory]

            # print(all_directories)
            mp_handler(all_directories, 1, output_directory)

