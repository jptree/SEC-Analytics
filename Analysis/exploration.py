from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pathlib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import re
import string


# TODO: Consider looking at random samples of bankruptcy subset and non-bankrupt subset and make comparisions of frequency of bigrams.
#   This could lead to feature creation!


PATH = '\\'.join(str(pathlib.Path().absolute()).split('\\')[:-1])


def text_stuff(df):
    # vectorizer = TfidfVectorizer()
    # df = pd.read_csv('results.csv')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df = df[~df['mda'].isna()]
    df['mda'] = df['mda'].apply(lambda x: str(x).replace('\\n', '\n').replace('$%$', ','))

    def apply_stuff(x):
        x = re.sub(r'[0-9]+', '', x)
        x = x.translate(str.maketrans('', '', string.punctuation))
        x = str(x).split(' ')
        try:
           return ' '.join([lemmatizer.lemmatize(w) for w in x if not w in stop_words])
        except ValueError:
            return ''


    df['mda'] = df['mda'].apply(lambda x: apply_stuff(x))
    corpus = list(df['mda'])
    count = CountVectorizer(ngram_range=(2, 2))
    count.fit(corpus)

    l = LatentDirichletAllocation(n_components=20)
    l.fit(corpus)

    # X = count.transform(corpus)
    #
    # transformer = TfidfTransformer()
    # X = transformer.fit_transform(X)
    #
    # tfidf = pd.DataFrame(data={'n_gram': list(count.vocabulary_.keys()), 'frequency': list(count.vocabulary_.values()), 'tfidf':transformer.idf_})
    # print(tfidf.sort_values(by='tfidf', ascending=False).to_csv('test2.csv'))

def lda(df):
    l = LatentDirichletAllocation(n_components=20)
    l.fit(df['mda'])
    print('')

if __name__ == "__main__":


    bankruptcy = pd.read_csv('bankruptcy_cik.csv', index_col=0)
    _, _, files = next(os.walk(f'{PATH}\\Analysis\\Extracted'))
    mda = pd.DataFrame(columns=['date', 'cik', 'mda'])
    for file in files:
        extracted = pd.read_csv(f'{PATH}\\Analysis\\Extracted\\{file}', header=None, names=['path', 'date', 'cik', 'mda'])
        mda = mda.append(extracted[['date', 'cik', 'mda']], ignore_index=True)
    #     print('a')
    mda['merge_date'] = mda['date'].apply(lambda x: f'{str(x)[:4]}-{str(x)[4:6]}')
    all = pd.merge(mda, bankruptcy, left_on=['merge_date', 'cik'], right_on=['date_month', 'cik'])
    # print()

    data = all[all['bankruptcyWithin12Months'] == 1]
    lda(data)
    # text_stuff(all[all['bankruptcyWithin12Months'] == 1])