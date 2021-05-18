from time import time
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pathlib
import os
import re
import string

# from sklearn.datasets import fetch_20newsgroups
PATH = '\\'.join(str(pathlib.Path().absolute()).split('\\')[:-1])



def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(60, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 10})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=10)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

bankruptcy = pd.read_csv(f'{PATH}\\SEC_Analytics\\Analysis\\bankruptcy_cik.csv', index_col=0)
_, _, files = next(os.walk(f'{PATH}\\SEC_Analytics\\Analysis\\Extracted'))
mda = pd.DataFrame(columns=['date', 'cik', 'mda'])
for file in files:
    extracted = pd.read_csv(f'{PATH}\\Analysis\\Extracted\\{file}', header=None, names=['path', 'date', 'cik', 'mda'])
    mda = mda.append(extracted[['date', 'cik', 'mda']], ignore_index=True)

_, _, files = next(os.walk(f'{PATH}\\SEC_Analytics\\Analysis\\Extracted\\Quarterly'))
# mda = pd.DataFrame(columns=['date', 'cik', 'mda'])
for file in files:
    extracted = pd.read_csv(f'{PATH}\\Analysis\\Extracted\\{file}', header=None, names=['path', 'date', 'cik', 'mda'])
    mda = mda.append(extracted[['date', 'cik', 'mda']], ignore_index=True)



mda['merge_date'] = mda['date'].apply(lambda x: f'{str(x)[:4]}-{str(x)[4:6]}')
all = pd.merge(mda, bankruptcy, left_on=['merge_date', 'cik'], right_on=['date_month', 'cik'])
# print()

data_samples = all[all['bankruptcyWithin12Months'] == 1]


lemmatizer = WordNetLemmatizer()

financial_stopwords = {'million', 'year', 'fiscal', 'january', 'february', 'march', 'april', 'may', 'june', 'july',
                       'august', 'september', 'october', 'november', 'december'}

def apply_stuff(x):
    x = re.sub(r'[0-9]+', '', x)
    x = x.translate(str.maketrans('', '', string.punctuation))
    x = set(str(x).split(' '))
    try:
        return ' '.join([lemmatizer.lemmatize(w) for w in x if str(x).lower() not in financial_stopwords])
    except ValueError:
        return ''


data_samples = data_samples[~data_samples['mda'].isna()]
data_samples['mda'] = data_samples['mda'].apply(lambda x: str(x).replace('\\n', '\n').replace('$%$', ','))
data_samples['mda'] = data_samples['mda'].apply(lambda x: apply_stuff(x))
data_samples = data_samples['mda']






n_samples = 1000
n_features = 2000
n_components = 10
n_top_words = 10

print("Loading dataset...")
t0 = time()

print("done in %0.3fs." % (time() - t0))

# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                   stop_words='english', ngram_range=(1, 2))
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english', ngram_range=(1, 2))
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))
print()

# Fit the NMF model
print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))


tfidf_feature_names = tfidf_vectorizer.get_feature_names()
plot_top_words(nmf, tfidf_feature_names, n_top_words,
               'Topics in NMF model (Frobenius norm)')

# Fit the NMF model
print('\n' * 2, "Fitting the NMF model (generalized Kullback-Leibler "
      "divergence) with tf-idf features, n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
t0 = time()
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
plot_top_words(nmf, tfidf_feature_names, n_top_words,
               'Topics in NMF model (generalized Kullback-Leibler divergence)')

print('\n' * 2, "Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

tf_feature_names = tf_vectorizer.get_feature_names()
plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')