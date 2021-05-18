import pandas as pd
import os
import re
import nltk
import multiprocessing
import gensim
import seaborn as sns
import matplotlib.pyplot as plt


def get_raw_mda():
    bankruptcy = pd.read_csv('bankruptcy_cik.csv', index_col=0)
    # Get annual files
    _, _, files = next(os.walk(f'Extracted'))
    mda = pd.DataFrame(columns=['date', 'cik', 'mda'])
    for file in files:
        extracted = pd.read_csv(f'Extracted\\{file}', header=None,
                                names=['path', 'date', 'cik', 'mda'])
        mda = mda.append(extracted[['date', 'cik', 'mda']], ignore_index=True)

    # Get quarterly files
    _, _, files = next(os.walk(f'Extracted\\Quarterly'))
    for file in files:
        extracted = pd.read_csv(f'Extracted\\Quarterly\\{file}', header=None,
                                names=['path', 'date', 'cik', 'mda'])
        mda = mda.append(extracted[['date', 'cik', 'mda']], ignore_index=True)

    mda['merge_date'] = mda['date'].apply(lambda x: f'{str(x)[:4]}-{str(x)[4:6]}')
    all = pd.merge(mda, bankruptcy, left_on=['merge_date', 'cik'], right_on=['date_month', 'cik'])
    return all


def clean_raw_text(text, stopwords, lemmatizer):

    # Remove punctuation, then strip the text of other spacing
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    list_text = text.split()

    # Remove stop words
    if stopwords is not None:
        list_text = [word for word in list_text if word not in stopwords]

    # Lemmatize words
    list_text = [lemmatizer.lemmatize(word) for word in list_text]

    # Convert list of tokens to text
    text = ' '.join(list_text)

    return text


def mp_worker(args):
    text = args[0]
    nltk_stopwords = args[1]
    nltk_lemmatizer = args[2]

    clean_mda = clean_raw_text(text, nltk_stopwords, nltk_lemmatizer)

    return clean_mda


def mp_handler(text, lemmatizer, stopwords, n_pools):
    p = multiprocessing.Pool(n_pools)
    args = list(zip(text, [stopwords] * len(text), [lemmatizer] * len(text)))
    counter = 0
    cleaned_documents = []
    for result in p.imap(mp_worker, args):
        counter += 1
        cleaned_documents.append(result)
        print(f'\rPercentage Complete: {round((counter / len(args)) * 100, 2)}%', end="", flush=True)
    print('\n')

    return cleaned_documents

def lda_stuff(corpus):
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i + 2]) for i in range(0,
                                                                 len(lst_words), 2)]
        lst_corpus.append(lst_grams)

    id2word = gensim.corpora.Dictionary(lst_corpus)
    dic_corpus = [id2word.doc2bow(word) for word in lst_corpus]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=dic_corpus, id2word=id2word, num_topics=50, random_state=123,
                                                update_every=1, chunksize=100, passes=10, alpha='auto',
                                                per_word_topics=True)

    ## output
    lst_dics = []
    for i in range(0, 50):
        lst_tuples = lda_model.get_topic_terms(i)
        for tupla in lst_tuples:
            lst_dics.append({"topic": i, "id": tupla[0],
                             "word": id2word[tupla[0]],
                             "weight": tupla[1]})
    dtf_topics = pd.DataFrame(lst_dics,
                              columns=['topic', 'id', 'word', 'weight'])
    dtf_topics.to_csv('topics.csv')
    ## plot
    # fig, ax = plt.subplots()
    # sns.barplot(y="word", x="weight", hue="topic", data=dtf_topics, dodge=False, ax=ax).set_title('Main Topics')
    # ax.set(ylabel="", xlabel="Word Importance")
    # plt.show()


def surrounding_word_corpus(word, corpus, n_words):

    corpus_around_word = []
    for document in corpus:
        docs = document.split(' ')
        for index, doc in enumerate(docs):
            if doc == word:
                context = []
                for i in range(n_words * 2):
                    try:
                        context.append(docs[index - n_words + i])
                    except IndexError:
                        pass

                corpus_around_word.append(' '.join(context))

    return corpus_around_word





if __name__ == "__main__":
    # Obtain the extracted MD&A sections in dataframe format
    data = get_raw_mda()
    # _, _, file_path = next(os.walk('../Data/10-K Sample'))
    # data = [open(f'../Data/10-K Sample/{file}').read() for file in file_path]

    # Clean the raw text
    nltk_stopwords = nltk.corpus.stopwords.words("english")
    nltk_lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    # corpus = mp_handler(data, nltk_lemmatizer, nltk_stopwords, 2)
    corpus = mp_handler(list(data['mda']), nltk_lemmatizer, nltk_stopwords, 4)

    # financial_stopwords = ['january', 'februrary', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
    #                        'november', 'december', 'year', 'fiscal']

    lda_stuff(corpus)

    # print(nltk_stopwords)



