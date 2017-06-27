import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import statsmodels.tsa.stattools as tsa
from statsmodels.tsa.stattools import adfuller
from sklearn.cluster import KMeans

def process(fileName):
    data = open(fileName)
    dataCsv = pd.read_csv(data)
    return dataCsv

def graphKNN(waves, step):
    plt.figure()
    n_graph_rows = 3
    n_graph_cols = 3
    graph_n = 1
    wave_n = 0
    for _ in range(n_graph_rows):
        for _ in range(n_graph_cols):
            axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
            axes.set_ylim([-100, 150])
            plt.plot(waves[wave_n])
            graph_n += 1
            wave_n += step
    # fix subplot sizes so that everything fits
    plt.tight_layout()
    plt.show()

fileSerie1 = "//home//nathana//AM//exercicio7//serie1.csv"
fileSerie2 = "//home//nathana//AM//exercicio7//serie2.csv"
data = process(fileSerie1)
print data.head()
print 'size:'
print len(data)
print '\n Data Types:'
print data.dtypes
print data.index

dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S")
data = pd.read_csv(fileSerie1, index_col='timestamp',date_parser=dateparse)
print data.head()
print 'min',data.max()

ekg_data = data
segment_len = 1200
slide_len = 100

segments = []
for start_pos in range(0, len(ekg_data), slide_len):
    end_pos = start_pos + segment_len
    # make a copy so changes to 'segments' doesn't modify the original ekg_data
    segment = np.copy(ekg_data[start_pos:end_pos])
    # if we're at the end and we've got a truncated segment, drop it
    if len(segment) != segment_len:
        continue
    segments.append(segment)

windowed_segments = []
window_rads = np.linspace(0, np.pi, segment_len)
window = np.sin(window_rads)**2

for segment in segments:
    windowed_segment = np.copy(segment) * window
    windowed_segments.append(windowed_segment)

plot_waves(segments, step=3)
plt.show()

cluster = KMeans(n_clusters=150)
for i in windowed_segments:
    print i
#X = np.array(pd.DataFrame.from_records(windowed_segments))
#cluster.fit(X)
#print type(cluster.cluster_centers_)
#plot_waves(cluster.cluster_centers_, step=15)
#plot_waves(cluster.cluster_centers_, step = 3)
#plt.show()
#plt.scatter(ekg_data[:, 0], ekg_data[:, 1], c=TwoDim)
#plot_waves(cluster.cluster_centers_, step=15)

print("Produced %d waveform segments" % len(segments))
