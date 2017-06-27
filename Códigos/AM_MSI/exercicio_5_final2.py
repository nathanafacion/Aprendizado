import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR , SVC
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import collections
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import urllib2

### Nome: Nathana Facion
### Exercicio 5 - Aprendizado de maquina

def meanFinal(acfinal, n_folds):
    return float(acfinal / n_folds)

def accuracy(matrix):
    return float(matrix[0][0] + matrix[1][1]) / float(matrix.sum())


def GBM_C ():
    parameters = { 'learning_rate' : [0.1], 'n_estimators' : [100], 'loss': ['deviance'],  'max_depth': [5]
        ,  'min_samples_split': [30], 'min_samples_leaf' : [62], 'max_features': ['auto'], 'warm_start' : [True],'min_weight_fraction_leaf':[0.2]}
    return GridSearchCV(GradientBoostingClassifier(), parameters, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)


def  SVR_In ():
    parameters = {'C': [2 ** (0)],'gamma': [2 ** (-10)],'epsilon': [2 ** (-5)],'coef0': [2 ** (0)],'kernel':['rbf'],'degree' : range(1,20,1) }
    return GridSearchCV(SVR(), parameters,cv=3,  scoring='neg_mean_absolute_error', n_jobs=1)


def SGD():
    parameters = {'penalty': ['l2'],'loss': ['epsilon_insensitive'],'alpha':[0.1],'epsilon':[0.5],'average':[2],'power_t': [0.1],'learning_rate':['invscaling'],
                      'l1_ratio':[0.1], 'fit_intercept':[True],'shuffle':[True],'n_iter':[100],'eta0': [0.005],'random_state':[5]}
    return GridSearchCV(linear_model.SGDRegressor(), parameters, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)

def HG():
    parameters = {'alpha': [0.6],'epsilon': [1],'max_iter':[60],'warm_start':[True]}
    return GridSearchCV(linear_model.HuberRegressor(), parameters, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)

def GBM():
    parameters = { 'learning_rate' : [0.1], 'alpha' :[0.1], 'n_estimators' : [100], 'loss': ['lad'],  'max_depth': [5]
        ,  'min_samples_split': [30], 'min_samples_leaf' : [62], 'subsample' : [0.2], 'random_state' :[10], 'max_features': ['auto'], 'warm_start' : [True],'min_weight_fraction_leaf':[0.2]}
    return GridSearchCV(GradientBoostingRegressor(), parameters, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)


def kfoldExterno(X,Y,algorithm):
    n_folds = 2
    external_skf = StratifiedKFold(n_folds)
    acxFinal = 0
    for training_index, test_index in external_skf.split(X,Y):
        X_train, X_test = X[training_index], X[test_index]
        Y_train, Y_test = Y[training_index], Y[test_index]
        if algorithm == 'SVR':
            model = SVR_In()
        elif algorithm == 'GBM_C':  # Gradient Boosting Machine
            model = GBM_C()
        elif algorithm == 'SVR':
            model = SVR_In()
        elif algorithm == 'SGD':  ### Stochastic Gradient Descent
            model = SGD()
        elif algorithm == 'HG':  ### Huber Regressor
            model = HG()
        elif algorithm == 'GBM':
            model = GBM()

        #print('Algorithm:',algorithm)
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)

        #print(model.cv_results_, 'melhor parametro', model.best_params_, model.best_score_)

        ac = mean_absolute_error(Y_test, predicted)
        #print('ac',ac)
        acxFinal = ac + acxFinal

    return meanFinal(acxFinal, n_folds)


def buildDataSet():
    fileName = "//home//nathana//train.csv"
    data = open(fileName)
    alphabet = (list(string.lowercase) + list(string.uppercase))
    dataCsv = pd.read_csv(data, sep=',', header=None)
    dataCsv.columns = [letter for letter in alphabet][0:33]
    dataCsvNotColumn = dataCsv[alphabet[1:33]]  # desconsiderar primeira linha
    dataDictionary = dataCsvNotColumn.T.to_dict().values()
    dicVectoririzer = DictVectorizer()
    classRegression = dataCsv[alphabet[0]]
    X = dicVectoririzer.fit_transform(dataDictionary).toarray()
    X = preprocessing.scale(X)
    Y = np.array(classRegression)

    count_elements = collections.Counter(Y)
    print 'Quanta cada uma das regressoes se repete',count_elements

    # Leitura dos dados de teste
    data_test = urllib2.urlopen("http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/test.csv")
    df_test = pd.read_csv(data_test, sep=',', header=None)
    df_test.columns = [letter for letter in alphabet][1:33]
    df_dic_test = df_test.T.to_dict().values()
    X_test = dicVectoririzer.transform(df_dic_test).toarray()

    return X, Y,X_test

# Realiza pre processamento
def preProcessingData():

    X , Y,X_test = buildDataSet()

    #deleteColumns = []
    ## Removendo colunas com NaN
    #for i in range(X.shape[1]):  # numero de colunas
     #   col = X[:, i]
     #   totalNaN = np.count_nonzero(np.isnan(col))
     #   percNaN = (float(totalNaN) / float(col.size))
     #   count_elements =collections.Counter(col)
     #   print  i,')',count_elements, 'size:', len(count_elements)
     #   deleteColumns.append(i) if (percNaN > 0.0) else None

    #X = np.delete(X, deleteColumns, axis=1)

    X = np.array(X)
    Y = np.array(Y)

    #normalizer =  Normalizer()
    #normalizer.fit(X)
    #X = normalizer.transform(X)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    #maxAbsolute = MaxAbsScaler()
    #maxAbsolute.fit(X)
    #X = maxAbsolute.transform(X)

    pca = PCA(n_components=88)
    #pca = SparsePCA(n_components = 88)
    #pca = FactorAnalysis(n_components = 150)
    #pca.fit(X)
    X = pca.fit_transform(X)
    #print(X.shape[1])
    #print(X.shape[0])
    X_test =  pca.fit_transform(X_test)

    print('preprocessamento...')
    return X, Y,X_test


def main():
    X, Y, X_test = preProcessingData()


    count_elements = collections.Counter(Y)
    print  'COL)', count_elements, 'size:', len(count_elements)


    cont2 = 0
    for i in count_elements:
        cont = 0
        if count_elements[i] < 200:
            for j in Y:
                #print j
                if i == Y[cont]:
                    Y[cont] = 1
                    cont2 = cont2 + 1
                cont = cont + 1

    print 'Numeros de dados que mudaram de classe:',cont2
    count_elements = collections.Counter(Y)
    print  'Quantidade de dados por valor de regressao', count_elements, 'size:', len(count_elements)


    ### Gradient Boosting Machine Regressor
    #mean_gbm = kfoldExterno(X, Y, 'GBM')
    #print('GBM Regressor:', mean_gbm)

    ### Gradient Boosting Machine Classificator
    #mean_gbm = kfoldExterno(X, Y, 'GBM_C')
    #print('GBM C:', mean_gbm)

    ### Stochastic Gradient Descent
    #mean_sgd = kfoldExterno( X, Y, 'SGD')
    #print('SGD:', mean_sgd)

    ### Huber Regressor
    #mean_hg = kfoldExterno(X, Y, 'HG')
    #print('HG:', mean_hg)

    #### SVM Regressor
    #mean_svr = kfoldExterno( X, Y, 'SVR')
    #print('SVR:', mean_svr)

    model = SVR_In()
    print len(X)
    print len(Y)

    model.fit(X, Y)
    print len(X_test)
    predicted = model.predict(X_test)
    # Salva a predicao do melhor algoritmo eu um arquivo .txt
    np.savetxt('predicao-' + "SVR" + '.txt', predicted)


if __name__ == '__main__':
    main()