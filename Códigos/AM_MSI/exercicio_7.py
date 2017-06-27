import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

### Nome: Nathana Facion
### Exercicio 7 - Aprendizado de maquina

def graphMeanStd(fileName, imageName):
    dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S")
    timeseries = pd.read_csv(fileName, index_col='timestamp', date_parser=dateparse)
    rolmean = pd.rolling_mean(timeseries, window=50)
    rolstd = pd.rolling_std(timeseries, window=50)
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Media')
    std = plt.plot(rolstd, color='black', label='Desvio Padrao')
    plt.legend(loc='best')
    plt.title(imageName)
    plt.savefig(imageName + ".png")
    plt.show()


def graphDBSCAN(X,labels,core_samples_mask):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    X = np.array(X)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        print 'cor',col,'k',k
        xy = X[class_member_mask & core_samples_mask]
        #print 'xy',xy
        plt.plot(xy[:,0], xy[:,1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)

    plt.title('Estimated number of clusters:')
    plt.show()

def metrics(X,labels):
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)

def AlgorDBSCAN(fileName,eps,min_samples ):
    # Pre - processamento
    X = process(fileName)
    stscaler = StandardScaler().fit(X)
    data = stscaler.transform(X)

    # Aplicando DBSCan
    dbsc = DBSCAN(eps = eps,min_samples = min_samples, metric = 'chebyshev' ).fit(data)
    labels = dbsc.labels_

    # Descobrindo estimativa de cluster
    metrics(X,labels)

    # Criando grafico
    print 'labels:',labels
    core_samples = np.zeros_like(labels, dtype=bool)
    core_samples[dbsc.core_sample_indices_] = True
    graphDBSCAN(X,labels,core_samples)


# Pre processa dados
def process(fileName):
    data = open(fileName)
    dataCsv = pd.read_csv(data, sep=',', header=None)[1:]
    deleteLine = []
    for i in range(dataCsv.shape[0]):
        col = dataCsv[0][i+1]
        try:
            dataCsv[0][i + 1] = time.mktime(datetime.datetime.strptime(col, "%Y-%m-%d %H:%M:%S").timetuple())
        except ValueError as e:
            print('Erro de data: dado ignorado linha', i+1)
            deleteLine.append(i)
            continue

    dataCsv = dataCsv.drop(dataCsv.index[deleteLine])
    return dataCsv

def graphKNN(waves, step):
    plt.figure()
    n_graph_rows = 3
    n_graph_cols = 3
    graph_n = 1
    wave_n = 0
    print waves
    print 'waves =>',waves
    for _ in range(n_graph_rows):
        for _ in range(n_graph_cols):
            axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            axes.set_ylim([-100, 150])
            plt.plot(pd.rolling_mean(waves[wave_n], window=100))
            plt.plot(waves[wave_n])
            graph_n += 1
            wave_n += step
            print wave_n

    # fix subplot sizes so that everything fits
    plt.tight_layout()
    plt.show()


def AlgorKNN(fileSerie1):
    dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S")
    data = pd.read_csv(fileSerie1, index_col='timestamp', date_parser=dateparse)

    ekg_data = data
    segment_len = 1200
    slide_len = 100
    segments = []
    for start_pos in range(0, len(ekg_data), slide_len):
        end_pos = start_pos + segment_len
        segment = np.copy(ekg_data[start_pos:end_pos])
        if len(segment) != segment_len:
            continue
        segments.append(segment)

    windowed_segments = []
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads) ** 2

    for segment in segments:
        windowed_segment = np.copy(segment) * window
        windowed_segments.append(windowed_segment)

    graphKNN(segments, step=3)

def main():
    # leitura da serie 1
    fileSerie1 = "//home//nathana//AM//exercicio7//serie1.csv"
    #dataCsv = process(fileSerie1)
    #AlgorDBSCAN(fileSerie1,0.5,5)
    #graphMeanStd(fileSerie1, 'Serie1')
    #AlgorKNN(fileSerie1)

    # leitura da serie 2
    fileSerie2 = "//home//nathana//AM//exercicio7//serie2.csv"
    #dataCsv = process(fileSerie2)
    #AlgorDBSCAN(dataCsv,0.5,5)
    #graphMeanStd(fileSerie2, 'Serie2')
    #AlgorKNN(fileSerie2)

    # leitura da serie 3
    fileSerie3 = "//home//nathana//AM//exercicio7//serie3.csv"
    #process(fileSerie3)
    AlgorDBSCAN(fileSerie3,0.8,50)
    #graphMeanStd(fileSerie3, 'Serie3')
    #AlgorKNN(fileSerie3)

    # leitura da serie 4
    fileSerie4 = "//home//nathana//AM//exercicio7//serie4.csv"
    #process(fileSerie4)
    #graphMeanStd(fileSerie4, 'Serie4')

    # leitura da serie 5
    fileSerie5 = "//home//nathana//AM//exercicio7//serie5.csv"
    #process(fileSerie5)
    #graphMeanStd(fileSerie5, 'Serie5')

if __name__ == '__main__':
    main()