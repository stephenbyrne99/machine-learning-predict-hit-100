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
#####################################################################
#####################################################################
#                XGB Model fixing
#####################################################################
####################################################################
df = pd.read_csv('final_lyrics_dataset.csv')
print(df.iloc[0:].to_json)

result = df.to_json(orient="split")

# df = df.drop('Artist name',axis=1)
X_labels = np.array(df["isTop100"])
df = df.drop('Title',axis=1)
df = df.drop('spotify_id',axis=1)
df = df.drop('isTop100',axis=1)
X_all = df

min_max_scaler = MinMaxScaler()
df["key"] = min_max_scaler.fit_transform(np.array(df[["key"]]))
df["loudness"] = min_max_scaler.fit_transform(np.array(df[["loudness"]]))
df["gunning_fog"] = min_max_scaler.fit_transform(np.array(df[["gunning_fog"]]))
df["flesch_ease"] = min_max_scaler.fit_transform(np.array(df[["flesch_ease"]]))

#making one hot encodes
ohe = OneHotEncoder()
res = ohe.fit_transform(X_all[["Artist"]])
X_all = X_all.drop("Artist", axis=1)

#converting the one hot encodes so they will work for xgboost
res_arr = res.toarray() 
res_arr = res_arr.astype(int) #float codes to int codes

X_all["aCode"] = pd.Series(res_arr.tolist())
print(type(X_all))

# print(X_all)

# https://stackoverflow.com/questions/56088264/trouble-training-xgboost-on-categorical-column
# turning binary into decimal
lbl = LabelEncoder()
X_all['aCode'] = lbl.fit_transform(X_all['aCode'].astype(str))

# the line below is used for dropping the our artist code ("aCode" / One-Hot Code). To see if it adds accuracy 
# X_all = X_all.drop("aCode", axis=1)
print(X_all.dtypes)

#####################################################################
#####################################################################
#                Using XGB
#####################################################################
####################################################################

X_train, X_test, y_train, y_test = train_test_split(X_all, X_labels, test_size=0.2, random_state=123)
xgbc = XGBClassifier(use_label_encoder=False, col__sample_bytree=0.1, gamma=0, eta=0.01, max_depth=3, clf__n_estimators=50, reg_lambda=0.000001)

print(X_all.head())

xgbc.fit(X_train, y_train)
print(xgbc)

fig, ax = plt.subplots(figsize=(30, 30))
# plot_tree(xgbc, num_trees=4, ax=ax)
# plt.savefig("xboost_tree.pdf")
# plt.show()


y_pred = xgbc.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# - cross validataion
scores = cross_val_score(xgbc, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

y_pred = xgbc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

fig, ax = plt.subplots(figsize=(30, 30))
plt.show()

#####################################################################
#####################################################################
#                XGB Grid Search
#####################################################################
####################################################################

# from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.metrics import accuracy_score, make_scorer# Define our search space for grid search
# search_space = [
#   {
#     #'clf__n_estimators': [50, 100, 150, 200, 250],
#     'clf__n_estimators': [50, 100, 150, 200, 250, 10],
#     'clf__learning_rate': [0.01, 0.1, 0.2, 0.05, 0.001],
#     #'clf__max_depth': range(2, 20),
#     'clf__max_depth': range(2, 10),
#     'clf__colsample_bytree': [i/10.0 for i in range(1, 10)],
#     'clf__gamma': [i/10.0 for i in range(10)],
#     'fs__score_func': [chi2],
#     'fs__k': [5],
#   }
# ]# Define cross validation
# kfold = KFold(n_splits=5, shuffle=True, random_state=55)# AUC and accuracy as score
# scoring = {'AUC':'roc_auc', 'Accuracy':make_scorer(accuracy_score)}# Define grid search
# grid = GridSearchCV(
#   xgbc,
#   param_grid=search_space,
#   cv=kfold,
#   scoring=scoring,
#   refit='AUC',
#   verbose=1,
#   n_jobs=-1
# )# Fit grid search
# model = grid.fit(X_train, y_train)

# predict = model.predict(X_test)
# print('Best AUC Score: {}'.format(model.best_score_))
# print('Accuracy: {}'.format(accuracy_score(y_test, predict)))
# print(confusion_matrix(y_test,predict))
# print(model.best_params_)

#####################################################################
#####################################################################
#                XGB AOC
#####################################################################
####################################################################
# for plotting auc
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, threshold = roc_curve(y_test, xgbc.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot(1,1, label='Most Frequent',marker='o')
plt.plot(0.5,0.5, label='Random',marker='o')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.show()