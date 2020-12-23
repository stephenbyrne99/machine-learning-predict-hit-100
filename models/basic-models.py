import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def create_test_space(size):
    # create test space
    X_test_space = []
    grid = np.linspace(size * -1,size)
    for i in grid:
        for j in grid:
            X_test_space.append([i,j])
    X_test_space = np.array(X_test_space)
    return X_test_space

def main():
    df = pd.read_csv ('../final_lyrics_dataset.csv')
    y = df.iloc[:,-1]
    X = df.drop('isTop100',axis=1)
    X = X.drop('Title',axis=1)
    X = X.drop('spotify_id',axis=1)

    X['Artist'] = X['Artist'].astype('category')
    X['artist_cat'] = X['Artist'].cat.codes
    X = X.drop('Artist',axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # Baseline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('\n \n')
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X,y)
    y_pred_dummy = dummy_clf.predict(X_test)
    print('Dummy classifier ', dummy_clf.score(X_test, y_test)) # retruns mean accuracy 
    true_positive = dummy_clf.score(X_test, y_test)
    false_positive = 1 - true_positive
    print('TP' , true_positive)
    print('FP' , false_positive)
    tn, fp, fn, tp = confusion_matrix(y_test,y_pred_dummy).ravel()
    print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    print(X)
    
    # lasso
    c_values = [0.001,0.01,0.1,1,10,100,1000]
    for ci in c_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        model_logistic_c_i = LogisticRegression(penalty="l2",C=ci)
        model_logistic_c_i .fit(X_train,y_train.values.ravel())
        print('\n \n')
        print('Lasso C=',ci,'Coef : ' , model_logistic_c_i .coef_)
        print('Lasso C=',ci,' Intercept : ' , model_logistic_c_i .intercept_)
        y_pred_i = model_logistic_c_i.predict(X_test)
        print('score for ' + str(ci) + ' : ' + str(model_logistic_c_i.score(X_test,y_test.values.ravel())))
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred_i).ravel()
        print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    # logistic 
    c_values = [0.001,0.01,0.1,1,10,100,1000]
    for ci in c_values:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        model_logistic_c_i = LogisticRegression(C=ci)
        model_logistic_c_i .fit(X_train,y_train.values.ravel())
        print('\n \n')
        print('logistic C=',ci,'Coef : ' , model_logistic_c_i .coef_)
        print('logistic C=',ci,' Intercept : ' , model_logistic_c_i .intercept_)
        y_pred_i = model_logistic_c_i.predict(X_test)
        print('score for ' + str(ci) + ' : ' + str(model_logistic_c_i.score(X_test,y_test.values.ravel())))
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred_i).ravel()
        print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    # kNN
    k_neighbours = [1,2,3,4,5,10,15,25,50,100]
    for ni in k_neighbours:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        model_kNN_ni = KNeighborsClassifier(n_neighbors=ni,weights="uniform").fit(X_train,y_train.values.ravel())
        y_pred_i = model_kNN_ni.predict(X_test)
        print('\n \n')
        print('kNN n =' + str(ni))
        print('score for ' + str(ni) + ' : ' + str(model_kNN_ni.score(X_test,y_test.values.ravel())))
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred_i).ravel()
        print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    # kNeighbours Regressor - doesnt work
    # gammas = [0,1,5,10,25]
    # for gamma in gammas:
    #     def gaussian_kernel(distances):
    #         weights = np.exp(-gamma*(distances**2))
    #         return weights/np.sum(weights)

    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    #     model_i = KNeighborsRegressor(n_neighbors=len(X_train),weights=gaussian_kernel).fit(X_train, y_train)
    #     y_pred_i = model_i.predict(X_test)
    #     print('\n \n')
    #     print('kNeighbours Regressor gamma =' + str(gamma))
    #     print('score for ' + str(gamma) + ' : ' + str(model_i.score(X,y.values.ravel())))
    #     tn, fp, fn, tp = confusion_matrix(y_test,y_pred_i).ravel()
    #     print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    
   

if __name__ == "__main__":
    main()
