
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from numpy import genfromtxt

def metric_intern(estimator, data):
    estimator.fit(data)
    return metrics.silhouette_score(data, estimator.labels_,
                             metric='euclidean')

def metric_extern(estimator, data, class_origin):
    estimator.fit(data)
    return metrics.adjusted_mutual_info_score(class_origin,  estimator.labels_)


def main():
    fileName = "/home/nathana/AM/exercicio4//cluster-data.csv"
    fileNameClass = "/home/nathana/AM/exercicio4//cluster-class.csv"

    data = genfromtxt(fileName, delimiter=',')[1:]
    dataClass = genfromtxt(fileNameClass, delimiter=',')[1:]

    fileData = np.array([d for d in data])
    fileClass = np.array([float(d) for d in dataClass])

    # k a se escolher
    k_init =  2
    k_end = 10

    # Rode o kmeans nos dados, com numero de restarts = 5
    n_init = 5

    metrics= []
    for k in xrange(k_init, k_end + 1):
        intern = metric_intern(KMeans(init='random', n_clusters=k, n_init=n_init),
                  data=fileData)
        extern = metric_extern(KMeans(init='random', n_clusters=k, n_init=n_init),
                  data=fileData,  class_origin = fileClass)

        metrics.append({'k': k, 'extern': extern , 'intern': intern})



    # 2 independent random clusterings with equal cluster number





    # Random labeling with varying n_clusters against ground class labels
    # with fixed number of clusters

    plots = []
    names = []
    for m in metrics:

        scores = m['intern']
        plots.append(plt.errorbar(
            scores , m['k'], m['k']))

    plt.ylim(ymin=-0.05, ymax=1.05)
    plt.legend(plots, names)
    plt.show()

if __name__ == '__main__':
    main()
