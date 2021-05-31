import re
import os
import time
import pickle
import multiprocessing
import pathlib
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble.forest import RandomForestClassifier

PATH = '\\'.join(str(pathlib.Path().absolute()).split('\\')[:-1])
DATA_PATH= ''
REGEX_10K = r"(Item[\s]+?7\.[\s\S]*?)(Item[\s]+?8\.)"
OPEN_INDEPENDENT_VARIABLES = ['position', 'trailing_management', 'trailing_period', 'trailing_7', 'trailing_newline',
                              'leading_newline', 'total_size', 'leading_tab', 'leading_spaces', 'trailing_financial',
                              'trailing_7A', 'regex_open','leading_see', 'leading_text', 'leading_double_newline',
                              'is_uppercase', 'trailing_omission', 'leading_table_of_contents', 'leading_newline_count',
                              'trailing_analysis']
CLOSE_INDEPENDENT_VARIABLES = ['position', 'trailing_management', 'trailing_period', 'trailing_7', 'trailing_8',
                               'trailing_newline', 'leading_newline', 'total_size', 'leading_tab', 'leading_spaces',
                               'trailing_financial', 'trailing_7A', 'regex_close', 'leading_see', 'leading_text',
                               'leading_double_newline', 'is_uppercase', 'trailing_omission',
                               'leading_table_of_contents', 'leading_newline_count', 'trailing_analysis']

OPEN_CLASSIFIER = pickle.load(open('../opening_random_forest.pkl', 'rb'))
CLOSE_CLASSIFIER = pickle.load(open('../closing_random_forest.pkl', 'rb'))

def get_mda(clf_open, clf_close, file_text):

    items = [m.start() for m in re.finditer('item', file_text, re.IGNORECASE)]

    total_text_length = len(file_text)
    match = re.findall(REGEX_10K, file_text, re.IGNORECASE)

    open_probabilities = []
    close_probabilities = []

    for index, item in enumerate(items):

        if 'management' in file_text[item: item + 50].lower():
            trailing_management = 1
        else:
            trailing_management = 0

        if 'analysis' in file_text[item: item + 80].lower():
            trailing_analysis = 1
        else:
            trailing_analysis = 0

        if '.' in file_text[item: item + 20]:
            trailing_period = 1
        else:
            trailing_period = 0

        if '7' in file_text[item: item + 20]:
            trailing_7 = 1
        else:
            trailing_7 = 0

        if '8' in file_text[item: item + 20]:
            trailing_8 = 1
        else:
            trailing_8 = 0

        if '\n' in file_text[item: item + 100]:
            trailing_newline = 1
        else:
            trailing_newline = 0

        if '7a' in file_text[item: item + 15].lower():
            trailing_7A = 1
        else:
            trailing_7A = 0

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

        if (len(file_text[item - 40: item].split(' ')) > 2) and ((sum(
                [len(w) for w in file_text[item - 40: item].split(' ')]) / len(
            file_text[item - 40: item].split(' '))) > 2):
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
            'trailing_7': trailing_7,
            'trailing_newline': trailing_newline,
            'leading_newline': leading_newline,
            'total_size': total_text_length,
            'leading_tab': leading_tab,
            'leading_spaces': leading_spaces,
            'regex_open': regex_open,
            'regex_close': regex_close,
            'trailing_financial': trailing_financial,
            'input_location': index,
            'trailing_7A': trailing_7A,
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
            'trailing_8': trailing_8
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

    try:
        open_index = open_probabilities.index(max(open_probabilities))
        close_index = close_probabilities.index(max(close_probabilities))
    except ValueError:
        return ''

    if open_index > close_index:
        return ''
    else:
        # return file_text[items[open_index]: items[close_index]]
        return file_text[items[open_index]: items[close_index]].replace('\n', '\\n').replace(',', '$%$')


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

    with open(output_dir, 'w') as f:
        for result in p.imap(mp_worker, file_names):
            f.write(f'{result[0]},{result[1]},{result[2]},{result[3]}\n')
    f.close()

    end = time.time()
    print(f'Multiple Threads: {round(end - start, 2)} seconds')

if __name__=='__main__':
    path_data = 'D:\\SEC Filing Data\\10-X_C_2006-2010'
    # # a, years, _ = next(os.walk(f'{path_data}\\2006\\QTR2'))
    _, years, _ = next(os.walk(path_data))
    # # print(a)
    #

    for year in years:
        _, quarters, _ = next(os.walk(f'{path_data}\\{year}'))
        if year == '2006':
            continue
        for quarter in quarters:
            if quarter != 'QTR1':
                continue
            print(f'Working on {quarter} of {year}...')
            output_directory = f'Extracted\\10-K_{year}_{quarter}.csv'
            all_directories = []
            _, _, directories = next(os.walk(f'{path_data}\\{year}\\{quarter}'))
            for directory in directories:
                if '_10-K_' in directory:
                    all_directories += [f'{path_data}\\{year}\\{quarter}\\' + directory]

            # print(all_directories)
            mp_handler(all_directories, 4, output_directory)


    # print(OPEN_CLASSIFIER.feature_importances_)








