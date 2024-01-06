"""
Used to shuffle Narrative data originally.
IMDB data came upackaged from Kaggle. 
"""

from copy import deepcopy

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.metrics import homogeneity_score, f1_score, accuracy_score
from sklearn.metrics import completeness_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import random
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
#df = pd.read_csv('instrument_clean.csv',
#                     names=['', "text_raw", "instrument", "text_clean", "text_freq"], skiprows=[0])
#df = df.sample(frac=1).reset_index(drop=True)
#df.to_csv('instrument_clean.csv')


def fitness(X, y, true_k, max_iter, n_init, X_test, y_test, edit_column):
    '''
    model = KMeans(n_clusters=true_k, init='k-means++', algorithm='lloyd', max_iter=max_iter, n_init=n_init)
    model.fit(np.asarray(X))
    predicted_labels = model.predict(X_test[:, edit_column])

    return adjusted_rand_score(y_test, predicted_labels)

    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(np.asarray(X), y)
    predicted_labels = classifier_linear.predict(np.asarray(X_test[:, edit_column].todense()))
    return f1_score(y_test, predicted_labels, average="weighted")'''

    model = GaussianNB()
    model.fit(np.asarray(X), y)
    predicted_labels = model.predict(np.asarray(X_test[:, edit_column].todense()))
    return f1_score(y_test, predicted_labels, average="weighted")

def make_population(pop_size):
    df = pd.read_csv('instrument_clean.csv',
                     names=['', "text_raw", "instrument", "text_clean", "text_freq"], nrows = 200)
    #df = pd.read_csv('IMDB_clean.csv',
    #                 names=['', "review", "sentiment", "text_clean"], nrows=500)

    df_test = pd.read_csv('instrument_clean.csv',
                     names=['', "text_raw", "instrument", "text_clean", "text_freq"], skiprows=200)
    #df_test = pd.read_csv('IMDB_clean.csv',
    #                      names=['', "review", "sentiment", "text_clean"], skiprows=501, nrows=100)

    y = df["instrument"]
    #vectorizer = TfidfVectorizer()
    vectorizer = CountVectorizer()
    normal_X = vectorizer.fit_transform(df['text_clean'].fillna(' '))
    normal_X = Normalizer().fit_transform(normal_X)
    normal_X = normal_X.todense()

    X_test = vectorizer.transform(df_test['text_clean'].fillna(' '))
    X_test = Normalizer().transform(X_test)
    y_test = df_test["instrument"]

    population = []
    for i in range(pop_size):
        edit_column = []
        prob = 1.0
        for i in range(normal_X.shape[1]):
            if random.uniform(0, 1) < prob:
                edit_column.append(i)
        new_X = normal_X[:, edit_column]
        n_init = 21
        max_iter = 133
        population.append([new_X, edit_column, max_iter, n_init])

    return population, y, normal_X, X_test, y_test

pop, y, norm, testx, testy = make_population(1)
print(fitness(pop[0][0], y, 11, 437, 45, testx, testy, pop[0][1]))
