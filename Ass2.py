# import logging
# import numpy as np
# import sys
# from time import time
# import matplotlib.pyplot as plt
import pandas as pd
#
# # from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
# from sklearn.pipeline import Pipeline
# from sklearn.utils.extmath import density
# from sklearn import metrics


def load_data(path):
    labels = []
    docs = []
    with open(path) as f:
        for line in f:
            splitted_line = line.rstrip('\n').rstrip('\r').split(maxsplit=1)
            labels.append(splitted_line[0])
            docs.append(splitted_line[1])

    return labels, docs


def stemming_tokenizer(str_input):
    porter_stemmer = PorterStemmer()
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]

    return words


def get_stemmed_stopwords():
    porter_stemmer = PorterStemmer()
    nltk.download('stopwords')
    stop_words = [porter_stemmer.stem(word) for word in stopwords.words("english")]

    return stop_words


def print_stats(train_labels, train_docs, test_labels, test_docs):

    #Create a dataframe containing both train and test sets.
    all_labels = train_labels.copy() + test_labels.copy()
    all_docs = train_docs.copy() + test_docs.copy()
    list_of_tuples = list(zip(all_labels, all_docs))

    df = pd.DataFrame(list_of_tuples, columns=['Label', 'Document'])

    grouped_df = df.groupby(['Label']).size().reset_index(name='Num of Docs')
    print('-' * 60)
    print('-' * 60)
    print("Number of categories: " + str(len(list(grouped_df.Label))))
    print('-' * 60)
    print('-' * 60)
    print("Number of documents per category:")
    print(grouped_df)


test_path = "test_hw2.txt"
train_path = "train_hw2.txt"

train_labels, train_docs = load_data(train_path)
test_labels, test_docs = load_data(test_path)

stop_words = get_stemmed_stopwords()
vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True, tokenizer=stemming_tokenizer)
X = vectorizer.fit_transform(train_docs)

print_stats(train_labels, train_docs, test_labels, test_docs)

