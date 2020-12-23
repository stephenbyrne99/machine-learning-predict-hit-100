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


dataset2 = "../final_lyrics_dataset.csv"
dataset1 = "../featured_taggedv_2.1.csv"

df = pd.read_csv (dataset2)


# adjust graph values so legibly
plt.rc('font', size=16)
plt.rcParams['figure.constrained_layout.use'] = True


y = df.iloc[:,-1]
X = df.drop('isTop100',axis=1)
X = X.drop('Title',axis=1)
X = X.drop('spotify_id',axis=1)


# option to switch between one hot encoding and categorical codes
hot = False

if (hot):
    #making one hot encodes
    ohe = OneHotEncoder()
    res = ohe.fit_transform(X[["Artist"]])
    X = X.drop("Artist", axis=1)

    #converting the one hot encodes so they will work for xgboost
    res_arr = res.toarray() 
    res_arr = res_arr.astype(int) #float codes to int codes

    X["aCode"] = pd.Series(res_arr.tolist())
    # print(X.dtypes)

    # https://stackoverflow.com/questions/56088264/trouble-training-xgboost-on-categorical-column
    # turning binary into decimal
    lbl = LabelEncoder()
    X['aCode'] = lbl.fit_transform(X['aCode'].astype(str))
else:
    X['Artist'] = X['Artist'].astype('category')
    X['artist_cat'] = X['Artist'].cat.codes
    X = X.drop('Artist',axis=1)


X = X.drop('has_lyrics',axis=1)
X = X.drop('flesch_readability',axis=1)


min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)



print(X)
print(y)


#########################################################################
# Logistic Model w/L1 penalty
#########################################################################

from sklearn import linear_model

c_values = [0.001,0.01,0.1,1,10,100,1000]
std_devs = []
means = []
for c in c_values:
    kf = KFold(n_splits=5, shuffle=True)
    mse_estimates = []
    for train, test in kf.split(X,y):
        clf = linear_model.LogisticRegression(penalty="l1",C=c,solver='saga',max_iter=10000)
        clf.fit(X[train],y[train].values.ravel())
        y_pred_i = clf.predict(X[test])
        mse_estimates.append(mean_squared_error(y_pred_i, y[test]))
       # print('\n \n \n ')
       # print('Logistic w/L2 score for ' + str(ci) + ' : ' + str(model_logistic_c_i.score(X[test],y[test].values.ravel())))
        tn, fp, fn, tp = confusion_matrix(y[test],y_pred_i).ravel()
       # print('Logistic w/L2 tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    means.append(np.mean(mse_estimates))
    std_devs.append(np.std(mse_estimates))

plt.errorbar(np.log10(c_values),means,yerr=std_devs,linewidth=3,capsize=5)
plt.title('Logistic w/l1 penalty - dif values c')
plt.xlabel('Log(C) values'); 
plt.ylabel('Mean/STD dev value')
plt.show()

#########################################################################
# kNN
#########################################################################

neighbours = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35]
std_devs = []
means = []
for ni in neighbours:
    kf = KFold(n_splits=5, shuffle=True)
    mse_estimates = []
    for train, test in kf.split(X,y):
        model_kNN_ni = KNeighborsClassifier(n_neighbors=ni,weights="uniform").fit(X[train],y[train].values.ravel())
        y_pred_i = model_kNN_ni.predict(X[test])
        mse_estimates.append(mean_squared_error(y_pred_i, y[test]))
        print('\n \n \n ')
        print('KNN score for ' + str(ni) + ' : ' + str(model_kNN_ni.score(X[test],y[test].values.ravel())))
        tn, fp, fn, tp = confusion_matrix(y[test],y_pred_i).ravel()
        print('kNN tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    means.append(np.mean(mse_estimates))
    std_devs.append(np.std(mse_estimates))


plt.errorbar(neighbours,means,yerr=std_devs,linewidth=3,capsize=5)
plt.title('kNN for Different Neighbours')
plt.xlabel('Neigbours'); 
plt.ylabel('Mean/STD dev value')
plt.show()


#########################################################################
# Random Forest
#########################################################################
from sklearn.ensemble import RandomForestClassifier

estimators = [1,5,10,25,30,35,40,45,50]
std_devs = []
means = []
for ni in estimators:
    kf = KFold(n_splits=5, shuffle=True)
    mse_estimates = []
    for train, test in kf.split(X,y):
        model_forest_ni =RandomForestClassifier(n_estimators=ni).fit(X[train],y[train])
        y_pred_i = model_forest_ni.predict(X[test])
        mse_estimates.append(mean_squared_error(y_pred_i, y[test]))
        print('\n \n \n ')
        print('random forest score for ' + str(ni) + ' : ' + str(model_forest_ni.score(X[test],y[test].values.ravel())))
        tn, fp, fn, tp = confusion_matrix(y[test],y_pred_i).ravel()
        print('random forest tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    means.append(np.mean(mse_estimates))
    std_devs.append(np.std(mse_estimates))


plt.errorbar(estimators,means,yerr=std_devs,linewidth=3,capsize=5)
plt.title('Random Forest for Different Estimators')
plt.xlabel('Estimators'); 
plt.ylabel('Mean/STD dev value')
plt.show()


# print('\n \n \n USING THE ABOVE CROSS VALIDATION')
# print('\n \n \n results for best models with optimised paramters')

# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc


#########################################################################
# Baseline
#########################################################################

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

# probs = dummy_clf.predict_proba(X_test)
# probs = probs[:,1]
# fper,tper,thresholds = roc_curve(y_test,probs)
# plt.plot(fper, tper, color='orange', label='ROC')
# plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend()
# plt.show()


#########################################################################
# Logistic model no penalty
#########################################################################

# print('\n \n')
# model_logistic = LogisticRegression()
# model_logistic.fit(X_train,y_train)
# print('Logistic no pen ', model_logistic.score(X_test, y_test)) # retruns mean accuracy 
# true_positive = model_logistic.score(X_test, y_test)
# false_positive = 1 - true_positive
# print('TP' , true_positive)
# print('FP' , false_positive)
# y_pred = model_logistic.predict(X_test)
# tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
# print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

# fpr,tpr, threshold = roc_curve(y_test,model_logistic.decision_function(X_test))
# roc_auc = auc(fpr, tpr)
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.plot(1,1, label='Most Frequent',marker='o')
# plt.plot(0.5,0.5, label='Random',marker='o')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for Logistic no pen')
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.show()


#########################################################################
# Logistic model with penalty
#########################################################################


print('\n \n')
model_knn = linear_model.LogisticRegression(penalty="l1",C=1,solver='saga',max_iter=10000)
model_knn.fit(X_train,y_train.values.ravel())
print('lasso ', model_knn.score(X_test, y_test)) # retruns mean accuracy 
true_positive = model_knn.score(X_test, y_test)
false_positive = 1 - true_positive
print('TP' , true_positive)
print('FP' , false_positive)
y_pred = model_knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

probs = model_knn.predict_proba(X_test)
probs = probs[:,1]
fper,tper,thresholds = roc_curve(y_test,probs)
plt.plot(fper, tper, color='b', label='AUC = %0.2f' % true_positive)
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.plot(1,1, label='Most Frequent',marker='o')
plt.plot(0.5,0.5, label='Random',marker='o')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for logistic l1 c=1')
plt.legend(loc = 'lower right')
plt.show()

#########################################################################
# kNN - 15
#########################################################################

print('\n \n')
model_knn = KNeighborsClassifier(n_neighbors=15,weights="uniform").fit(X_train,y_train.values.ravel())
print('kNN ', model_knn.score(X_test, y_test)) # retruns mean accuracy 
true_positive = model_knn.score(X_test, y_test)
false_positive = 1 - true_positive
print('TP' , true_positive)
print('FP' , false_positive)
y_pred = model_knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

probs = model_knn.predict_proba(X_test)
probs = probs[:,1]
fper,tper,thresholds = roc_curve(y_test,probs)
plt.plot(fper, tper, color='b', label='AUC = %0.2f' % true_positive)
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.plot(1,1, label='Most Frequent',marker='o')
plt.plot(0.5,0.5, label='Random',marker='o')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for kNN n=15')
plt.legend(loc = 'lower right')
plt.show()



#########################################################################
# Random forest - 45
#########################################################################

print('\n \n')
model_rf = RandomForestClassifier(n_estimators=45).fit(X_train,y_train)
print('random forest ', model_rf.score(X_test, y_test)) # retruns mean accuracy 
true_positive = model_rf.score(X_test, y_test)
false_positive = 1 - true_positive
print('TP' , true_positive)
print('FP' , false_positive)
y_pred = model_rf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print('tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

probs = model_rf.predict_proba(X_test)
probs = probs[:,1]
fper,tper,thresholds = roc_curve(y_test,probs)
# plt.plot(fper, tper, color='orange', label='ROC')
plt.plot(fper, tper, color='b', label='AUC = %0.2f' % true_positive)
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(1,1, label='Most Frequent',marker='o')
plt.plot(0.5,0.5, label='Random',marker='o')
plt.title('ROC Curve for rf n=45')
plt.legend(loc = 'lower right')
plt.show()






