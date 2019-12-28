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
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from time import time
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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


# found in stackOverflow in the following link:
# https://stackoverflow.com/questions/53218931/how-to-unnest-explode-a-column-in-a-pandas-dataframe/53218939#53218939
import numpy as np

def unnesting(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


def print_stats(train_labels, train_docs, test_labels, test_docs):
    # Create a dataframe containing both train and test sets.
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

    tokenize_docs = []
    for doc in all_docs:
        tokenize_docs.append(word_tokenize(doc))
    list2 = list(zip(all_labels, tokenize_docs))
    df2 = pd.DataFrame(list2, columns=['Label', 'Word'])
    df3 = unnesting(df2, ['Word']).groupby(['Label', 'Word']).size().reset_index(name='Occurences')
    print ("-" * 60)

    for name, group in df3.groupby('Label'):
        print("Category:" + name)
        print(group.nlargest(10, 'Occurences'))
        print ("-" * 60)



test_path = "test_hw2.txt"
train_path = "train_hw2.txt"

train_labels, train_docs = load_data("train_hw2.txt")
test_labels, test_docs = load_data("test_hw2.txt")

stop_words = get_stemmed_stopwords()
vectorizer = TfidfVectorizer(stop_words=stop_words, lowercase=True, tokenizer=stemming_tokenizer)
print (train_labels)
X_Train = vectorizer.fit_transform(train_docs)
X_Test = vectorizer.transform(test_docs)

Y_Train = train_labels
Y_Test = test_labels

print_stats(train_labels, train_docs, test_labels, test_docs)



# Classify - using machine learning methods : SVM, Naive-Bayes, Random Forest
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_Train, Y_Train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_Test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(Y_Test, pred)
    print("accuracy:   %0.3f" % score)

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


for clf, name in (
        (SGDClassifier(), "SVM"),
        (MultinomialNB(), "Naive Bayes"),
        (RandomForestClassifier(), "Random Forest")):
    print('=' * 80)
    print(name)
    benchmark(clf)



# this part deals with optimizing the model


# optimizing SVM
nb_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', SGDClassifier())])
parameters = {'vect__max_df': (0.3, 0.5), 'clf__alpha': (0.0001, 0.001)}
gs_clf = GridSearchCV(nb_clf, parameters, n_jobs=1)
gs_clf = gs_clf.fit(train_docs, train_labels)
print('Best score: ', gs_clf.best_score_)
print('Best params: ', gs_clf.best_params_)

best_model = gs_clf.best_estimator_
best_score = gs_clf.best_score_

# optimizing Naive Bayes
nb_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])
parameters = {'vect__max_df': (0.3, 0.5), 'clf__alpha': (0.0001, 0.001)}
gs_clf = GridSearchCV(nb_clf, parameters, n_jobs=1)
gs_clf = gs_clf.fit(train_docs, train_labels)
print('Best score: ', gs_clf.best_score_)
print('Best params: ', gs_clf.best_params_)

if best_score < gs_clf.best_score_:
    best_model = gs_clf.best_estimator_
    best_score = gs_clf.best_score_

# optimizing Rand Forest
nb_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', RandomForestClassifier())])
parameters = {'vect__max_df': (0.3, 0.5), 'clf__criterion': ('gini', 'entropy'), 'clf__min_samples_leaf': (1, 10)}
gs_clf = GridSearchCV(nb_clf, parameters, n_jobs=1)
gs_clf = gs_clf.fit(train_docs, train_labels)
print('Best score: ', gs_clf.best_score_)
print('Best params: ', gs_clf.best_params_)
if best_score < gs_clf.best_score_:
    best_model = gs_clf.best_estimator_
    best_score = gs_clf.best_score_
