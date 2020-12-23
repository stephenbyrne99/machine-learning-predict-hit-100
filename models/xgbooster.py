from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.model_selection import TimeSeriesSplit , GridSearchCV, RandomizedSearchCV

df = pd.read_csv ('featured_taggedv_2.1.csv')
# df = df.drop('Artist name',axis=1)
X_labels = np.array(df["isTop100"])
df = df.drop('Title',axis=1)
df = df.drop('spotify_id',axis=1)
df = df.drop('isTop100',axis=1)
X_all = df

#making one hot encodes
ohe = OneHotEncoder()
res = ohe.fit_transform(X_all[["Artist"]])
X_all = X_all.drop("Artist", axis=1)

#converting the one hot encodes so they will work for xgboost
res_arr = res.toarray() 
res_arr = res_arr.astype(int) #float codes to int codes

X_all["aCode"] = pd.Series(res_arr.tolist())
print(X_all.dtypes)

# https://stackoverflow.com/questions/56088264/trouble-training-xgboost-on-categorical-column
# turning binary into decimal
lbl = LabelEncoder()
X_all['aCode'] = lbl.fit_transform(X_all['aCode'].astype(str))

# the line below is used for dropping the our artist code ("aCode" / One-Hot Code). To see if it adds accuracy 
# X_all = X_all.drop("aCode", axis=1)
print(X_all.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X_all, X_labels, test_size=0.2, random_state=123)
xgbc = XGBClassifier(use_label_encoder=False)

print(X_all.head())
# xgbc.fit(X_train, y_train)
# print(xgbc)

# y_pred = xgbc.predict(X_test)
# predictions = [round(value) for value in y_pred]
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# # - cross validataion
# scores = cross_val_score(xgbc, X_train, y_train, cv=5)
# print("Mean cross-validation score: %.2f" % scores.mean())

# y_pred = xgbc.predict(X_test)
# cm = confusion_matrix(y_test,y_pred)
# print(cm)

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer# Define our search space for grid search
search_space = [
  {
    # 'clf__n_estimators': [100, 150],
    'clf__n_estimators': [50, 100, 150, 200],
    'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
    # 'clf__learning_rate': [0.01, 0.1],
    'clf__max_depth': range(3, 10),
    # 'clf__max_depth': range(3, 8),
    'clf__colsample_bytree': [i/10.0 for i in range(1, 3)],
    'clf__gamma': [i/10.0 for i in range(3)],
    'fs__score_func': [chi2],
    'fs__k': [5],
  }
]# Define cross validation
kfold = KFold(n_splits=5, random_state=42)# AUC and accuracy as score
scoring = {'AUC':'roc_auc', 'Accuracy':make_scorer(accuracy_score)}# Define grid search
grid = GridSearchCV(
  xgbc,
  param_grid=search_space,
  cv=kfold,
  scoring=scoring,
  refit='AUC',
  verbose=1,
  n_jobs=-1
)# Fit grid search
model = grid.fit(X_train, y_train)

predict = model.predict(X_test)
print('Best AUC Score: {}'.format(model.best_score_))
print('Accuracy: {}'.format(accuracy_score(y_test, predict)))
print(confusion_matrix(y_test,predict))
print(model.best_params_)