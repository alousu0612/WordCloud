import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

from wordcloud import WordCloud

import os
import numpy as np
import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import pickle


# create tf, idf function


def tf(term, doc_tokens):
    return doc_tokens.count(term) / len(doc_tokens)


def numDocsContaining(word, token_doclist):
    return sum([doc_token.count(word) > 0 for doc_token in token_doclist])


def idf(word, token_doclist):
    return math.log10(len(token_doclist) / numDocsContaining(word, token_doclist))


def compute_tfidf(doc_tokens, bag_words_idf):
    return {word: tf(word, doc_tokens) * bag_words_idf[word] for word in set(doc_tokens) if word in bag_words_idf.keys()}


def title_words(data):
    title_bag_words = set()
    for txt in data:
        title_bag_words.update(text_preprocessing(txt[:-4]))

    with open('./data/title_bag_words.pickle', 'wb') as handle:
        pickle.dump(title_bag_words, handle, protocol=pickle.HIGHEST_PROTOCOL)


def text_preprocessing(text, stopwords={}):
    '''
    For single text preprocessing
    Tokenization, removing stopwords and punctuations, word_count
    '''

    tokens = nltk.word_tokenize(text.lower())

    stopwords = set(nltk.corpus.stopwords.words('english')).union(stopwords)

    with open('./data/title_bag_words.pickle', 'rb') as handle:
        title_words = pickle.load(handle)

    words = set(nltk.corpus.words.words()).intersection(title_words)

    require = {'NN', 'NNS', 'JJ'}

    filtered_tokens = [word for word, pos in nltk.pos_tag(
        tokens) if word not in stopwords if word in words if len(word) > 2 if pos in require]

    return filtered_tokens


def data_preprocessing(data, tokenize=True):
    # all_tokens, bag of words
    tokens = []
    for text in data:
        if tokenize:
            tokens += text
        else:
            tokens += text_preprocessing(text)

    bag_of_words = set(tokens)

    return tokens, bag_of_words


def doc_text(datapath):
    doc_all = {}
    for filename in os.listdir(datapath):
        if filename.split('.')[1] == 'txt':
            text = open(datapath+filename, encoding='utf-8').read()
            doc_all[filename[:-4]] = text_preprocessing(text)

    # return doc_all: {doc: tokens}, tokens_pos: bag of words with pos_tag
    return doc_all


def query(que, bag_words_idf, doc_all_tfidf):
    # query
    query_tokens = text_preprocessing(que)
    if 'image' in query_tokens:
        query_tokens.remove('image')

    doc_all_tfidf['query'] = compute_tfidf(query_tokens, bag_words_idf)

    tfidf_df = pd.DataFrame(doc_all_tfidf).transpose().fillna(0)

    cosine_sim = {doc: cosine_similarity([tfidf_df.loc[doc], tfidf_df.loc['query']])[0, 1]
                  for doc in tfidf_df.index if doc != 'query'}

    return dict(sorted(cosine_sim.items(), key=lambda item: item[1], reverse=True)[:10])


def plot_query(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(range(len(data)), list(data.values()), align='center', alpha=0.5)
    plt.yticks(range(len(data)), list(data.keys()))
    ax.set_xlabel('Smimilarity score')
    ax.set_ylabel('Papers')
    fig.savefig('./pic/query.png', bbox_inches='tight', pad_inches=0)


def wrdcld(data):

    count = {word: freq for word, freq in Counter(data).most_common() if len(word) > 5}

    _mask = Image.open("./pic/mask.png")
    mask = np.array(_mask)

    wordcloud = WordCloud(max_words=300, max_font_size=300,
                          prefer_horizontal=1.0,
                          relative_scaling=0.5,
                          background_color="white", mask=mask)
    wordcloud.generate_from_frequencies(count)

    return wordcloud
