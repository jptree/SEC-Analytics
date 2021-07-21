import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
import csv
import time

STARTING_YEAR = 1994
# STARTING_YEAR = 1993
STARTING_QUARTER = 3
ENDING_YEAR = 2018
ENDING_QUARTER = 4
PATH_TO_EXTRACTED = '../../Text Extraction/Extracted/Quarterly'
OUTPUT_DIRECTORY = 'similarity.csv'

pd.set_option('display.max_columns', 10)


def get_quarter_mda(desired_year, desired_quarter, path_to_extracted):
    _, _, file_names = next(os.walk(path_to_extracted))
    for file_name in file_names:
        if (f'QTR{desired_quarter}' in file_name) and (str(desired_year) in file_name):
            return pd.read_csv(f'{path_to_extracted}\\{file_name}', header=None, names=['path', 'date', 'cik', 'mda'])


def compute_similarity(document1, document2):
    vectorizer = TfidfVectorizer()
    vectorizer.fit([document1, document2])
    return cosine_similarity(vectorizer.transform([document1]), vectorizer.transform([document2]))[0][0]


def mp_worker(args):
    year = args[0]
    quarter = args[1]


    result = get_similarity_df(quarter, year)
    result['end_of_month'] = result['end_of_month'].astype(str)

    return result


def mp_handler(work_list, n_pools, output_dir):
    p = multiprocessing.Pool(n_pools)

    writer = csv.writer(open(output_dir, 'a', newline=''))
    counter = 0
    for result in p.imap(mp_worker, work_list):
        counter += 1
        if result is not None:
            writer.writerows(result.values.tolist())
            # print(result.values.tolist()[:5])
        print(f'\rPercentage Complete: {round((counter / len(work_list)) * 100, 2)}%', end="", flush=True)
    print('\n')

    p.close()

def get_previous_filing(text_t_1, text_t_2):
    if (pd.isna(text_t_1)) and (not pd.isna(text_t_2)):
        return text_t_2
    elif (not pd.isna(text_t_1)) and (pd.isna(text_t_2)):
        return text_t_1
    elif (not pd.isna(text_t_1)) and (not pd.isna(text_t_2)):
        return text_t_1
    else:
        return None


def get_similarity_df(current_quarter, current_year):
    quarter_t = current_quarter
    year = current_year
    quarter_t_1 = range(4)[current_quarter - 1 - 1] + 1
    year_t_1 = year - 1 if current_quarter == 1 else year
    quarter_t_2 = range(4)[current_quarter - 2 - 1] + 1
    year_t_2 = year - 1 if current_quarter < 2 else year

    # if year == STARTING_YEAR and quarter_t == 1:
    #     return None

    if year == STARTING_YEAR and quarter_t == 2:
        df_t = get_quarter_mda(year, quarter_t, '../../Text Extraction/Extracted/Quarterly')
        df_t_1 = get_quarter_mda(year_t_1, quarter_t_1, '../../Text Extraction/Extracted/Quarterly')

        if not (df_t or df_t_1):
            return None

        # TODO: Drop if length of MDA is super small.... this would indicate an error

        df_t = df_t[~df_t['mda'].isna()]
        df_t = df_t[df_t['mda'] != '']
        df_t = df_t[['date', 'cik', 'mda']]

        df_t_1 = df_t_1[~df_t_1['mda'].isna()]
        df_t_1 = df_t_1[df_t_1['mda'] != '']
        df_t_1 = df_t_1[['cik', 'mda']]

        df_all = pd.merge(df_t, df_t_1, how='left', on='cik', suffixes=['_t', '_t_1'])
        df_all = df_all[~df_all['mda_t'].isna()]
        df_all = df_all[~df_all['mda_t_1'].isna()]

        df_all['similarity'] = df_all.apply(lambda x: compute_similarity(x['mda_t'], x['mda_t_1']), axis=1)
        df_all['end_of_month'] = df_all['date'].to_period('M').to_timestamp('M')


        return df_all[['cik', 'end_of_month', 'similarity']]

    else:
        df_t = get_quarter_mda(year, quarter_t, '../../Text Extraction/Extracted/Quarterly')
        df_t_1 = get_quarter_mda(year_t_1, quarter_t_1, '../../Text Extraction/Extracted/Quarterly')
        df_t_2 = get_quarter_mda(year_t_2, quarter_t_2, '../../Text Extraction/Extracted/Quarterly')


        df_t = df_t[~df_t['mda'].isna()]
        df_t = df_t[df_t['mda'] != '']
        df_t = df_t[['date', 'cik', 'mda']]

        df_t_1 = df_t_1[~df_t_1['mda'].isna()]
        df_t_1 = df_t_1[df_t_1['mda'] != '']
        df_t_1 = df_t_1[['cik', 'mda']]

        df_t_2 = df_t_2[~df_t_2['mda'].isna()]
        df_t_2 = df_t_2[df_t_2['mda'] != '']
        df_t_2 = df_t_2[['cik', 'mda']]


        df_all = pd.merge(df_t, df_t_1, how='left', on='cik', suffixes=['_t', '_t_1'])

        # Get rid of any columns that have no current MD&A text
        df_all = df_all[~df_all['mda_t'].isna()]

        df_all = pd.merge(df_all, df_t_2, how='left', on='cik')
        df_all = df_all.rename(columns={'mda': 'mda_t_2'})

        df_all['previous_mda'] = df_all.apply(lambda x: get_previous_filing(x['mda_t_1'], x['mda_t_2']), axis=1)
        df_all = df_all[~df_all['previous_mda'].isna()]

        df_all['similarity'] = df_all.apply(lambda x: compute_similarity(x['mda_t'], x['previous_mda']), axis=1)
        df_all['end_of_month'] = pd.to_datetime(df_all['date'], format='%Y%m%d').dt.to_period('M').dt.to_timestamp('M')

        return df_all[['cik', 'end_of_month', 'similarity']]



def main():

    list_of_work = []
    for year in range(STARTING_YEAR, ENDING_YEAR + 1):
        for q in range(4):
            if year == STARTING_YEAR and (q + 1 < STARTING_QUARTER):
                continue
            if year == ENDING_YEAR and q + 1 == ENDING_QUARTER:
                break
            list_of_work.append((year, q + 1))

    start = time.time()
    list_of_work.append((2018, 4))
    print(list_of_work)
    mp_handler(list_of_work, 6, OUTPUT_DIRECTORY)
    print(f'Multiple Threads: {round(time.time() - start, 2)} seconds')

if __name__ == "__main__":
    main()
    # print(get_similarity_df(3, 1994))