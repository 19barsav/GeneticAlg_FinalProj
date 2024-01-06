"""
Savvy B.
CSE 848 - Evolutional Computation
Final Project Code

Feature Finder GA
To change data, see make_population()
    NOTE: run Narrative data with true_k of 11, and IMDB
    with true_k of 2. Do this in GA()
To change model, see fitness()
To graph KMean clusters, see graph_clusters()

Current Models: Gaussian Naive Bayes, Linear SVM, KMeans++
"""


from copy import deepcopy

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import homogeneity_score, f1_score, accuracy_score
from sklearn.metrics import rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import completeness_score

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB

from sklearn.manifold import TSNE

from sklearn import svm



def fitness(X, y, true_k, max_iter, n_init, X_test, y_test, edit_column):
    """
    A measure-based score (double) for the GA to maximize. Leave models
    commented, except for the one you want to run. Current models are
    KMeans, SVM, and Gaussian Naive Bayes.

    More models may be added here.

    :param X: training text data, parent member
    :param y: training true labels/classifications
            (not necessary for current fitness functions)
    :param true_k: number of clusters to make
    :param max_iter: max iterations, parent member
    :param n_init: n-initializations, parent member
    :param X_test: testing text data
    :param y_test: testing true labels/classifications
    :param edit_column: indexes of X (parent member)
    :return: returns double - fitness score
    """

    model = KMeans(n_clusters=true_k, init='k-means++', algorithm='lloyd', max_iter=max_iter, n_init=n_init)
    model.fit(np.asarray(X))
    predicted_labels = model.predict(X_test[:, edit_column])

    return adjusted_rand_score(y_test, predicted_labels)
    '''
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(np.asarray(X), y)
    predicted_labels = classifier_linear.predict(np.asarray(X_test[:, edit_column].todense()))
    return f1_score(y_test, predicted_labels, average="weighted") + accuracy_score(y_test, predicted_labels)
  
    model = GaussianNB()
    model.fit(np.asarray(X), y)
    predicted_labels = model.predict(np.asarray(X_test[:, edit_column].todense()))
    return f1_score(y_test, predicted_labels, average="weighted") + accuracy_score(y_test, predicted_labels)
  '''



def make_population(pop_size, vec_choice):
    """
    Created the population members of the GA and reads in all training and test data.
    Uncomment whichever data source is wanted. If IMDB is wanted, uncomment the lower set
    of calls. For Narrative, uncomment the upper.

    :param pop_size: population size
    :param vec_choice: vectorizer choice
            Choice 1: TFIDF
            Chocie 2: Count
    :return: returns a list of lists
            Form: [features, indexes, max_iter, n-it]
            In addition, it returns true labels of training set, the list of all features,
            and the test data + labels
    """
    df = pd.read_csv('instrument_clean.csv',
                     names=['', "text_raw", "instrument", "text_clean", "text_freq"], nrows = 200)

    #df = pd.read_csv('IMDB_clean.csv',
    #                       names=['', "review", "sentiment", "text_clean"], nrows=500)

    df_test = pd.read_csv('instrument_clean.csv',
                     names=['', "text_raw", "instrument", "text_clean", "text_freq"], skiprows=200)
    #df_test = pd.read_csv('IMDB_clean.csv',
    #                      names=['', "review", "sentiment", "text_clean"], skiprows=501, nrows=200)

    y_test = df_test["instrument"]
    #y_test = df_test["sentiment"]

    y = df["instrument"]
    #y = df["sentiment"]

    if vec_choice == 1:
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()
    normal_X = vectorizer.fit_transform(df['text_clean'].fillna(' '))
    normal_X = Normalizer().fit_transform(normal_X)
    normal_X = normal_X.todense()
    print(normal_X.shape[1])
    X_test = vectorizer.transform(df_test['text_clean'].fillna(' '))
    X_test = Normalizer().transform(X_test)


    population = []
    for i in range(pop_size):
        edit_column = []
        prob = random.uniform(0.1, 1)
        #prob = 1
        for i in range(normal_X.shape[1]):
            if random.uniform(0, 1) < prob:
                edit_column.append(i)
        new_X = normal_X[:, edit_column]
        n_init = random.randint(1, 50)
        max_iter = random.randint(50, 500)
        population.append([new_X, edit_column, max_iter, n_init])

    return population, y, normal_X, X_test, y_test


def tournament_selection(population, fitnesses, tournament_size):
    '''
    Randomly selects population members and selects member with best fitness

    :param population: the entire list of population members
    :param fitnesses: calculates fitness for each pop. member
    :param tournament_size: number of members to be randomly selected
    :return: returns the population member [[],[],x,x]
    '''
    selected_indices = []
    for x in range(tournament_size):
        random_index = random.randint(0, len(population) - 1)
        selected_indices.append(random_index)
    best_index = max(selected_indices, key=lambda i: fitnesses[i])
    return population[best_index]


def crossover_features(parent1, parent2, normal_X):
    """
    One=point crossover for lists - takes care of duplicates - uses sets
    :param parent1: the first population member to be crossed
    :param parent2: the second populatioin member to be crossed
    :param normal_X: entire, original feature list
    :return: 2 lists, each a new population member
    """
    index_cross = random.randint(0, min(len(parent1[1]), len(parent2[1])))
    child1_indexes = list(set(parent1[1][0:index_cross]).union(parent2[1][index_cross:]))
    child2_indexes = list(set(parent2[1][0:index_cross]).union(parent1[1][index_cross:]))
    child1 = normal_X[:, child1_indexes]
    child2 = normal_X[:, child2_indexes]
    return [child1, child1_indexes, parent1[2], parent1[3]], [child2, child2_indexes, parent2[2], parent2[3]]


def crossover_max_iter(parent1, parent2, crossover_rate=1.0):
    """
    Blended crossover for max_iteration.
    Also done with rate of 0.80.
    :param parent1: First population member
    :param parent2: Second Population Member
    :param crossover_rate: Default to 1.0, but works well with 0.8.
    :return: nothing
    """
    if np.random.random() < crossover_rate:
        u = random.uniform(0, 1)
        k = random.uniform(0, 1)
        if parent1[2] < parent2[2]:
            child1 = (1 - u) * parent1[2] + u * parent2[2]
            child2 = (1 - k) * parent1[2] + k * parent2[2]
        else:
            child1 = (1 - u) * parent2[2] + u * parent1[2]
            child2 = (1 - k) * parent2[2] + k * parent1[2]
        parent1[2] = int(child1)
        parent2[2] = int(child2)


def crossover_n_init(parent1, parent2, crossover_rate=1.0):
    """
    Blended crossover for n_init.
    Also done with rate of 0.80.
    :param parent1: First population member
    :param parent2: Second Population Member
    :param crossover_rate: Default to 1.0, but works well with 0.8.
    :return: nothing
    """
    if np.random.random() < crossover_rate:
        u = random.uniform(0, 1)
        k = random.uniform(0, 1)
        if parent1[3] < parent2[3]:
            child1 = (1 - u) * parent1[3] + u * parent2[3]
            child2 = (1 - k) * parent1[3] + k * parent2[3]
        else:
            child1 = (1 - u) * parent2[3] + u * parent1[3]
            child2 = (1 - k) * parent2[3] + k * parent1[3]
        parent1[3] = int(child1)
        parent2[3] = int(child2)


def mutation_max_iter(parent, mutation_rate=0.2):
    """
    Max_iter mutation = random number in range
    :param parent: population member
    :param mutation_rate: default to 0.2, likelihood of mutation
    :return: nothing
    """
    if random.uniform(0, 1) < mutation_rate:
        parent[2] = random.randint(50, 500)


def mutation_n_init(parent, mutation_rate=0.2):
    """
    n_init mutation = random number in range
    :param parent: population member
    :param mutation_rate: default to 0.2, likelihood of mutation
    :return: nothing
    """
    if random.uniform(0, 1) < mutation_rate:
        parent[3] = random.randint(1, 50)


def mutation_features(parent, normal_X, mutation_rate=0.2):
    """
    Mutates the feature list by giving chance for missing indexes
            to be added and existing indexes ot be removed
    :param parent: population member
    :param normal_X: entire, original feature list
    :param mutation_rate: default to 0.2, likelihood of mutation
    :return: nothing
    """
    full_column = [i for i in range(normal_X.shape[1])]
    missing_columns = list(set(full_column) - set(parent[1]))
    new_edit_column = [i for i in parent[1] if random.uniform(0, 1) <= 1 - mutation_rate]
    new_edit_column2 = [i for i in missing_columns if random.uniform(0, 1) <= mutation_rate]
    new_edit_column = list(set(new_edit_column2).union(new_edit_column))
    new_X = normal_X[:, new_edit_column]
    parent[0] = new_X
    parent[1] = new_edit_column


def graph_clusters(X, y, true_k, max_iter, n_init):
    """
    Graphs the k-mean clusters. Currently not implemented, commented in GA().
    Will take forever with more data.
    :param X: text data
    :param y: text labels
    :param true_k: number of clusters
    :param max_iter: maximum iterations
    :param n_init: n-initializations
    :return: nothing, shows graph
    """
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), y, test_size=0.33, random_state=42)
    model = KMeans(n_clusters=true_k, init='k-means++', algorithm='lloyd', max_iter=max_iter, n_init=n_init,
                   random_state=0)
    model.fit_transform(X_train)
    pca_model = PCA(n_components=3)
    X_train = pca_model.fit_transform(X_train)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=model.labels_)
    plt.show()


def plot_program_evolution(fitnesses, generations):
    """
    Plots the evolution of generations. Works for any two lists of the same size (of numbers).
    :param fitnesses: list of top fitnesses per generation
    :param generations: list of generations - numbers are [1-size of generations]
    :return: nothing, shows plot
    """
    a, b = np.polyfit(np.array([i for i in range(generations)]), np.array(fitnesses), 1)
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(generations)], fitnesses, label="Evolved Fitnesses")
    plt.plot(np.array([i for i in range(generations)]), a * np.array([i for i in range(generations)]) + b)
    plt.legend()
    plt.ticklabel_format(style='plain')
    plt.title(f"Evolution of Best Program over {generations} Generations")
    plt.show()


def GA(vec_choice):
    """
    The genetic algorithm.
    Can adjust population and generation sizes.
    :param vec_choice: chooses between vectorizers, noted in main project note
    :return: best fitness found, double
    """
    pop_size = 100
    num_generations = 100
    pop, y, normal_x, X_test, y_test = make_population(pop_size, vec_choice)
    best_fitnesses = []
    best_parents = []
    for generation in range(num_generations):
        new_pop = []
        print(f"Generation {generation + 1}")
        fitnesses = [fitness(parent[0], y, 11, parent[2], parent[3], X_test, y_test, parent[1]) for parent in pop]
        print(f"Best fitness {max(fitnesses)}")
        best_fitnesses.append(max(fitnesses))
        top = max(enumerate(fitnesses), key=lambda x: x[1])[0]
        best_parents.append(deepcopy(pop[top]))
        for i in range(pop_size):
            print(i) if i % 10 == 0 else None
            best = tournament_selection(pop, fitnesses, 3)
            best2 = tournament_selection(pop, fitnesses, 3)
            c1, c2 = crossover_features(best, best2, normal_x)
            # blended crossover method
            crossover_max_iter(c1, c2)
            crossover_n_init(c1, c2)
            mutation_max_iter(c1)
            mutation_max_iter(c2)
            mutation_n_init(c1)
            mutation_n_init(c2)
            mutation_features(c1, normal_x)
            mutation_features(c2, normal_x)
            new_pop.append(c1)
            new_pop.append(c2)
        #Elitism
        new_pop.append(deepcopy(best_parents[max(enumerate(best_fitnesses), key=lambda x: x[1])[0]]))
        pop = new_pop

    best_f = max(best_fitnesses)
    best_cluster = best_parents[max(enumerate(best_fitnesses), key=lambda x: x[1])[0]]

    print(best_f)
    print(best_cluster[1])
    print(len(best_cluster[1]))
    print(len(normal_x))
    print(best_cluster[2])
    print(best_cluster[3])
    plot_program_evolution(best_fitnesses, num_generations)
    #graph_clusters(best_cluster[0], y, 2, best_cluster[2], best_cluster[3])
    return best_fitnesses[-1]

def main():
    GA(2)


main()
