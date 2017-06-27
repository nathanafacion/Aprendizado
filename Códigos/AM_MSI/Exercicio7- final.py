import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import gridspec

### Nome: Nathana Facion
### Exercicio 7 - Aprendizado de maquina

# Cria o grafico com media e desvio padrao.
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
    plt.tight_layout()
    #plt.show()

# soma os valores das ondas
def wavesSum(values):
    sum = 0
    for v in values:
        if  not np.isnan(v):
            sum += v
    return sum

# Detecta janela de anomalia.
def wavesGraph(imageName ,waves, data, type, step):
    plt.figure()
    n_graph_rows = 3
    n_graph_cols = 3
    graph_n = 1
    wave_n = 0
    greater_sum = [0 if type == 'Maior' else sys.maxint]
    greater_wave = 0
    for _ in range(n_graph_rows):
        for _ in range(n_graph_cols):
            axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            axes.set_ylim([-100, 150])
            mean = pd.rolling_mean(waves[wave_n], window=100)
            std = pd.rolling_std(waves[wave_n], window=100)
            plt.plot(mean, label='Media')
            plt.plot(std, label='Desvio Padrao')
            sum = wavesSum(waves[wave_n])
            if type == 'Maior':
                if sum > greater_sum:
                    greater_sum = sum
                    greater_wave = wave_n
            else:
                if sum < greater_sum:
                    greater_sum = sum
                    greater_wave = wave_n
            plt.plot(waves[wave_n], label='Original')
            graph_n += 1
            wave_n += step

    plt.tight_layout()
    #plt.show()

    # Plota as diferentes janelas criadas para analisar
    plt.savefig(imageName + "-Fases.png")
    plt.clf()

    # debug
    # plt.plot(waves[greater_wave])
    # plt.savefig("Teste" + ".png")
    #plt.show()

    oneClass(imageName,waves[greater_wave],data)


# Essa funcao faz com que o arquivo seja dividido em segmentos
# esses segmentos serão usados para detectarmos a janela de anomalia
def waveWindows(fileSerie1, imageName, window, inters, type):
    dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S")
    data = pd.read_csv(fileSerie1, index_col='timestamp', date_parser=dateparse)
    dataReturn  = data

    # Duas questoes a serem tratadas sao:
    #  qual o valor de N
    segment_len = window
    # e quanto de interssecao entre os trechos.
    slide_len = inters

    segments = []
    for start_pos in range(0, len(data), slide_len):
        end_pos = start_pos + segment_len
        segment = np.copy(data[start_pos:end_pos])
        if len(segment) != segment_len:
            continue
        segments.append(segment)

    windowed_segments = []
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads) ** 2

    for segment in segments:
        windowed_segment = np.copy(segment) * window
        windowed_segments.append(windowed_segment)

    wavesGraph(imageName, segments, dataReturn, type, step=3)

# Algoritmo usado para detectar anomalias no trecho : SVM One Class - visto em sala de aula
def oneClass (imageName, X_outliers, X_train):

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # Divide o grafico em duas telas
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    ax1 = plt.subplot(gs[0])
    ax1.plot(X_train)
    ax1.set_title('Serie')

    ax2 = plt.subplot(gs[1])
    ax2.plot(X_outliers,color='green', markersize=5)
    ax2.set_title('Anomalia')

    fig.text(0.5, 0.04, "Numero de erros no treino: %d/4033 ; Numero de erros na janela de  anomalia: %d/1200" % (n_error_train, n_error_outliers), ha='center', va='center')

    plt.savefig(imageName + ".png")
    plt.axis('tight')
    plt.clf()
    #plt.show()

def main():
    # leitura da serie 1 ------------------------------------
    fileSerie1 = "//home//nathana//AM//exercicio7//serie1.csv"

    # Plota medias e desvio padrao
    graphMeanStd(fileSerie1, 'Serie1')

    # Encontra janela com anomalia
    waveWindows(fileSerie1, 'Serie1- window',1200,100,'Maior')

    # leitura da serie 2 ------------------------------------
    fileSerie2 = "//home//nathana//AM//exercicio7//serie2.csv"

    # Plota medias e desvio padrao
    graphMeanStd(fileSerie2, 'Serie2')

    # Encontra janela com anomalia
    waveWindows(fileSerie2,'Serie2- window',1200,100, 'Menor')

    # leitura da serie 3 ------------------------------------
    fileSerie3 = "//home//nathana//AM//exercicio7//serie3.csv"

    # Plota medias e desvio padrao
    graphMeanStd(fileSerie3, 'Serie3')
    waveWindows(fileSerie3,'Serie3- window',1200,100, 'Menor')

    # leitura da serie 4 ------------------------------------
    fileSerie4 = "//home//nathana//AM//exercicio7//serie4.csv"

    # Plota medias e desvio padrao
    graphMeanStd(fileSerie4, 'Serie4')

    # Encontra janela com anomalia
    waveWindows(fileSerie4,'Serie4- window',1200,100,'Maior')

    # leitura da serie 5 ------------------------------------
    fileSerie5 = "//home//nathana//AM//exercicio7//serie5.csv"

    # Plota medias e desvio padrao
    graphMeanStd(fileSerie5, 'Serie5')

    # Encontra janela com anomalia
    waveWindows(fileSerie5,'Serie5- window',1200,100,'Maior')

if __name__ == '__main__':
    main()