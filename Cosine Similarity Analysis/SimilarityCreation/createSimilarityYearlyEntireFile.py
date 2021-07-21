import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
import multiprocessing
import csv
import time
import numpy as np

# https://poseidon01.ssrn.com/delivery.php?ID=803090114088005098001115102064020103016052085015079029126119079027126070064113116030006052013006037046016118069065014088116005048011066061044005098018125101099122071079013098081117007004069103122123112093100065074127113008077064093107098018100126111&EXT=pdf&INDEX=TRUE


def compute_similarity(document1, document2):
    vectorizer = CountVectorizer(binary=True)
    vectorizer.fit([document1, document2])
    return cosine_similarity(vectorizer.transform([document1]), vectorizer.transform([document2]))[0][0]


def compute_jaccard(document1, document2):
    vectorizer = CountVectorizer()
    vectorizer.fit([document1, document2])
    return jaccard_score(np.array(vectorizer.transform([document1]).toarray()[0]), vectorizer.transform([document2]).toarray()[0])


def mp_worker(args):
    current_file = args[0]
    previous_file = args[1]
    date = current_file.split('\\')[5].split('_')[0]
    cik = current_file.split('\\')[5].split('_')[4]

    current_file = open(current_file).read()
    previous_file = open(previous_file).read()


    return [date, cik, compute_similarity(current_file, previous_file)]


def mp_handler(current_files, previous_files, n_pools, output_dir):
    p = multiprocessing.Pool(n_pools)
    n_cik = len(current_files)
    work = zip(current_files, previous_files)
    # writer = csv.writer(open(output_dir, 'a', newline=''))

    with open(output_dir, 'a', newline='') as f:
        writer = csv.writer(f)

        counter = 0
        for result in p.imap(mp_worker, work):
            counter += 1
            if result is not None:
                writer.writerow(result)
            print(f'\rPercentage Complete: {round((counter / n_cik) * 100, 2)}%', end="", flush=True)
        print('\n')

    p.close()


def get_previous_filing(cik, files_t_1):
    if cik in files_t_1:
        return files_t_1[cik]
    else:
        return None


if __name__ == "__main__":
    path_data_list = [
        'D:\\SEC Filing Data\\10-X_C_1993-2000',
        'D:\\SEC Filing Data\\10-X_C_2001-2005',
        'D:\\SEC Filing Data\\10-X_C_2006-2010',
        'D:\\SEC Filing Data\\10-X_C_2011-2015',
        'D:\\SEC Filing Data\\10-X_C_2016-2018'
    ]

    work = []
    for path_data in path_data_list:
        _, years, _ = next(os.walk(path_data))
        for year in years:
            _, quarters, _ = next(os.walk(f'{path_data}\\{year}'))
            for quarter in quarters:
                work.append(f'{path_data}\\{year}\\{quarter}')

    files_t_1 = None
    for w_i, w in enumerate(work):
        quarter = w.split('\\')[4]
        year = w.split('\\')[3]

        previous_year = str(int(year) - 1)

        if w_i - 4 < 0:
            continue
        else:
            w_previous = work[w_i - 4]


        # if int(year) < 2002:
        #     continue
        #
        # if int(year) == 2002 and quarter in ['QTR1', 'QTR2', 'QTR3']:
        #     continue

        print(f'Working on {quarter} of {year}...')
        output_directory = f'../EntireFilePreviousYear\\Cosine\\CountVectorizerBinary\\Annual\\10-K_{year}_{quarter}.csv'
        files_t = {}
        file_names = []
        _, _, directories = next(os.walk(w))
        for directory in directories:
            if '_10-K_' in directory:
                cik = directory.split('_')[4]
                files_t[cik] = f'{w}\\{directory}'
                file_names.append(directory)

        files_t_1 = {}
        _, _, directories = next(os.walk(w_previous))
        for directory in directories:
            if '_10-K_' in directory:
                cik = directory.split('_')[4]
                files_t_1[cik] = f'{w_previous}\\{directory}'

        current_files = []
        previous_files = []
        for file_name in file_names:
            cik = file_name.split('_')[4]
            previous_file = get_previous_filing(cik, files_t_1)

            if previous_file:
                previous_files.append(previous_file)
                current_files.append(f'{w}\\{file_name}')

        mp_handler(current_files, previous_files, 6, output_directory)

