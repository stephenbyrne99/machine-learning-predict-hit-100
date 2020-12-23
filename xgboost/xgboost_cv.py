from xgboost import plot_tree, XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

# from sklearn.model_selection import TimeSeriesSplit , GridSearchCV, RandomizedSearchCV

df = pd.read_csv('final_lyrics_dataset.csv')
print(df.iloc[0:].to_json)

result = df.to_json(orient="split")

# df = df.drop('Artist name',axis=1)
y = df.iloc[:,-1]
df = df.drop('Title',axis=1)
df = df.drop('spotify_id',axis=1)
df = df.drop('isTop100',axis=1)
df = df.drop("flesch_readability", axis=1)
X = df

min_max_scaler = MinMaxScaler()
df["key"] = min_max_scaler.fit_transform(np.array(df[["key"]]))
df["gunning_fog"] = min_max_scaler.fit_transform(np.array(df[["gunning_fog"]]))
df["flesch_ease"] = min_max_scaler.fit_transform(np.array(df[["flesch_ease"]]))

#making one hot encodes
ohe = OneHotEncoder()
res = ohe.fit_transform(X[["Artist"]])
X = X.drop("Artist", axis=1)

#converting the one hot encodes so they will work for xgboost
res_arr = res.toarray() 
res_arr = res_arr.astype(int) #float codes to int codes

X["aCode"] = pd.Series(res_arr.tolist())
print(type(X))

# print(X_all)

# https://stackoverflow.com/questions/56088264/trouble-training-xgboost-on-categorical-column
# turning binary into decimal
lbl = LabelEncoder()
X['aCode'] = lbl.fit_transform(X['aCode'].astype(str))

# the line below is used for dropping the our artist code ("aCode" / One-Hot Code). To see if it adds accuracy 
# X_all = X_all.drop("aCode", axis=1)
print(X.dtypes)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

c_values = [0.000001,0.00001, 0.0001, 0.001,0.01,0.1,1,10,100,1000, 10000]
std_devs = []
means = []
for ci in c_values:
    kf = KFold(n_splits=5)
    mse_estimates = []
    for train, test in kf.split(X,y):
        xgbc = XGBClassifier(use_label_encoder=False, verbose=0, gamma=0, eta=0.01, max_depth=3, reg_lambda=ci)

        xgbc.fit(X.iloc[train],y.iloc[train].values.ravel())
        y_pred_i = xgbc.predict(X.iloc[test])
        mse_estimates.append(mean_squared_error(y_pred_i, y.iloc[test]))
       # print('\n \n \n ')
       # print('Logistic w/L2 score for ' + str(ci) + ' : ' + str(model_logistic_c_i.score(X[test],y[test].values.ravel())))
        tn, fp, fn, tp = confusion_matrix(y.iloc[test],y_pred_i).ravel()
       # print('Logistic w/L2 tp=' + str(tp) + ' tn=' + str(tn) + ' fp=' + str(fp) + ' fn=' + str(fn))

    means.append(np.mean(mse_estimates))
    std_devs.append(np.std(mse_estimates))

plt.errorbar(np.log10(c_values),means,yerr=std_devs,linewidth=3,capsize=5)
plt.title('XGB with l2 penalty with different values of C')
plt.xlabel('Log(C) values') 
plt.ylabel('Mean/STD dev value')
plt.show()




