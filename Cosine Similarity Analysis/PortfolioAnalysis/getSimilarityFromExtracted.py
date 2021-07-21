import pandas as pd
import os
import multiprocessing
import csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


# OUTPUT_FOLDER = '../MDA/Cosine/CountVectorizerBinary/Quarterly'


def get_previous(cik, df_p):
    try:
        previous_text = list(df_p.loc[df_p['cik'] == cik]['mda'])[0]
        return previous_text
    except IndexError:
        return None


def compute_similarity(document1, document2, similarity_type, vectorizer_type, binary=False):
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(binary=binary)
    elif vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(binary=binary)
    else:
        vectorizer = CountVectorizer(binary=binary)

    vectorizer.fit([document1, document2])

    if similarity_type == 'cosine':
        return cosine_similarity(vectorizer.transform([document1]), vectorizer.transform([document2]))[0][0]
    elif similarity_type == 'jaccard':
        return jaccard_score(vectorizer.transform([document1]).toarray().tolist()[0], vectorizer.transform([document2]).toarray().tolist()[0])
    else:
        return cosine_similarity(vectorizer.transform([document1]), vectorizer.transform([document2]))[0][0]


def mp_worker(args):
    current_text = args[0]
    previous_text = args[1]
    date = args[2]
    cik = args[3]
    similarity_type = args[4]
    vectorizer_type = args[5]
    binary = args[6]

    return [date, cik, compute_similarity(current_text, previous_text, similarity_type, vectorizer_type, binary)]


def mp_handler(work, n_pools, output_dir):
    p = multiprocessing.Pool(n_pools)
    n_work = len(work)

    with open(output_dir, 'a', newline='') as f:
        writer = csv.writer(f)

        counter = 0
        for result in p.imap(mp_worker, work):
            counter += 1
            if result is not None:
                writer.writerow(result)
            print(f'\rPercentage Complete: {round((counter / n_work) * 100, 2)}%', end="", flush=True)
        print('\n')

    p.close()


def get_yearly_change(directory, similarity_type, vectorizer_type, binary, output_dir):
    _, _, dirs = next(os.walk(directory))
    for i, d in enumerate(dirs):
        if i - 4 < 0:
            continue

        quarter = d.split('_')[2][:4]
        year = d.split('_')[1]

        print(f'Working on {quarter} of {year}...')

        previous_dir = dirs[i - 4]
        df_c = pd.read_csv(f'{directory}\\{d}', header=None, names=['path', 'date', 'cik', 'mda'])
        df_p = pd.read_csv(f'{directory}\\{previous_dir}', header=None, names=['path', 'date', 'cik', 'mda'])

        df_c['previous_mda'] = df_c['cik'].apply(lambda x: get_previous(x, df_p))
        df_c = df_c[~(df_c['mda'].isna()) & ~(df_c['previous_mda'].isna())]

        work = list(zip(df_c['mda'],
                        df_c['previous_mda'],
                        df_c['date'],
                        df_c['cik'],
                        [similarity_type] * len(df_c['mda']),
                        [vectorizer_type] * len(df_c['mda']),
                        [binary] * len(df_c['mda'])))

        output_directory = f'{output_dir}/10-K_{year}_{quarter}.csv'
        mp_handler(work, 6, output_directory)



if __name__ == "__main__":
    get_yearly_change('C:\\Users\\jpetr\\PycharmProjects\\SEC-Analytics\\Text Extraction\\Extracted\\Quarterly',
                      'cosine', 'count', False, '../MDA/cosine/CountVectorizer/Quarterly')