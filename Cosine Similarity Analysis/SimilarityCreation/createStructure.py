import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
import csv

pd.set_option('display.max_columns', 10)
pd.options.display.width = 0

STARTING_YEAR = 2006
# STARTING_YEAR = 1993
STARTING_QUARTER = 1
ENDING_YEAR = 2010
# ENDING_YEAR = 2018
ENDING_QUARTER = 4
PATH_TO_EXTRACTED = '../../Old/Analysis/Extracted/Quarterly'
OUTPUT_DIRECTORY = 'test.csv'


def get_quarter_mda(desired_year, desired_quarter, path_to_extracted):
    _, _, file_names = next(os.walk(path_to_extracted))
    for file_name in file_names:
        if (f'QTR{desired_quarter}'in file_name) and (str(desired_year) in file_name):
            return pd.read_csv(f'{path_to_extracted}\\{file_name}', header=None, names=['path', 'date', 'cik', 'mda'])


def compute_similarity(document1, document2):
    vectorizer = TfidfVectorizer()
    vectorizer.fit([document1, document2])
    return cosine_similarity(vectorizer.transform([document1]), vectorizer.transform([document2]))[0][0]


def mp_worker(args):
    document1 = args[0]
    document2 = args[1]
    cik = args[2]
    date = args[3]

    similarity = compute_similarity(document1, document2)

    return similarity, cik, date


def mp_handler(current_mda, previous_mda, cik, date, n_pools, output_dir):
    p = multiprocessing.Pool(n_pools)
    args = list(zip(current_mda, previous_mda, cik, date))

    writer = csv.writer(open(output_dir, 'a', newline=''))
    counter = 0
    for result in p.imap(mp_worker, args):
        counter += 1
        print(f'\rPercentage Complete: {round((counter / len(args)) * 100, 2)}%', end="", flush=True)
        writer.writerow([result[0], result[1], result[2]])
    print('\n')


if __name__ == "__main__":
    for year in range(STARTING_YEAR, ENDING_YEAR + 1):

        for quarter in range(1, 4 + 1):
            if (year == STARTING_YEAR) and ((quarter == 1) or (quarter == 2)):
                continue

            df_one = get_quarter_mda(year, [1, 2, 3, 4][quarter - 2], PATH_TO_EXTRACTED)[['cik', 'mda']]
            df_one = df_one.set_index('cik')

            df_two = get_quarter_mda(year, [1, 2, 3, 4][quarter - 3], PATH_TO_EXTRACTED)[['cik', 'mda']]
            df_two = df_two.rename(columns={'mda': 'mda_two'})
            df_two = df_two.set_index('cik')

            df_current = get_quarter_mda(year, quarter, PATH_TO_EXTRACTED)

            df_all = df_current.join(df_one, on='cik', how='left', lsuffix='_current', rsuffix='_one')
            df_all = df_all.join(df_two, on='cik', how='left')
            df_all['mda_previous'] = df_all['mda_one']
            # df_all.loc[df_all['mda_previous'].isnull(), 'mda_previous'] = df_all['mda_two']
            df_all['mda_previous'].fillna(df_all['mda_two'], inplace=True)

            df_all = df_all[(~df_all['mda_current'].isna()) & (~df_all['mda_previous'].isna())]


            current_mda = list(df_all['mda_current'])
            previous_mda = list(df_all['mda_previous'])
            dates = list(df_all['date'])
            ciks = list(df_all['cik'])

            print(f'Working on Quarter {quarter} of {year}')
            mp_handler(current_mda, previous_mda, ciks, dates, 1, OUTPUT_DIRECTORY)

