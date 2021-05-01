from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if __name__ == "__main__":
    # vectorizer = TfidfVectorizer()
    df = pd.read_csv('results.csv')
    df = df[~df['text'].isna()]
    df['text'] = df['text'].apply(lambda x: str(x).replace('\\n', '\n').replace('$%$', ','))
    corpus = list(df['text'])



    count = CountVectorizer(ngram_range=(2, 2))
    count.fit(corpus)

    X = count.transform(corpus)

    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)

    tfidf = pd.DataFrame(data={'n_gram': list(count.vocabulary_.keys()), 'frequency': list(count.vocabulary_.values()), 'tfidf':transformer.idf_})
    print(tfidf.sort_values(by='tfidf', ascending=False))





    # for index, row in df.iterrows():
    #     for w in weird:
    #         if w in row['text']:
    #             file = open('s.txt', 'w')
    #             file.write(row['text'])
    #             file.close()
    #             print(w, print(row['filing']))
    #             input('continue?')