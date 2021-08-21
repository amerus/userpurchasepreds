#!/usr/bin/env python
# coding: utf-8

EXPORT_FILE="predictions.csv"
TRAINING_DATA="data_train.csv"
TEST_DATA="data_test.csv"
PROFILE_FEATURES="features.csv"

import sklearn.externals
import joblib
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd
from sklearn.model_selection import train_test_split

data_test = pd.read_csv(TEST_DATA, index_col=0)
data_train = pd.read_csv(TRAINING_DATA, index_col=0)
features = dd.read_csv(PROFILE_FEATURES, sep="\t").set_index("Unnamed: 0")

test_ids = data_test.id.to_list()
train_ids = data_train.id.to_list()
train_and_test = list(set(train_ids + test_ids))
feat_comm = features.loc[features["id"].isin(train_and_test), :]

data_train_dd = dd.from_pandas(data_train, npartitions=1)
X = data_train_dd.merge(feat_comm, on=['id','buy_time'])
X = X.compute().reset_index()
y = X['target']
X = X.drop('target', axis=1)

train_idx, test_idx = train_test_split(X.index, random_state=42, test_size=0.3)
y_train, y_test = y.loc[train_idx], y.loc[test_idx]
X_train, X_test = X.loc[train_idx], X.loc[test_idx]

scaler = StandardScaler()
all_features = X_test.columns.tolist()
X_nunique = X_test.apply(lambda x: x.nunique(dropna=False))
const_feat = X_nunique[X_nunique == 1].index.tolist()
selected = list(set(all_features) - set(const_feat))
bin_feat = set(X.loc[:,selected].columns[((X.loc[:,selected].min()==0) &                                              
              (X.loc[:,selected].max()==1) &            
              (X.loc[:,selected].isnull().sum()==0))])

selected = list(set(selected) - bin_feat)
X_train_scaled = scaler.fit_transform(X_train.loc[:,selected])
X_test_scaled = scaler.transform(X_test.loc[:,selected])


model = CatBoostClassifier(n_estimators=2000, max_depth=8, eval_metric='Precision',    random_state=42, class_weights=[0.1, 10], task_type="CPU")


def eval_cat_model(model, X_train, y_train, X_valid, y_valid, feature_names):
    from pprint import pprint
    model.fit(X_train, y_train, verbose=False, eval_set=(X_valid, y_valid), plot=True)
    #model.load_model("catboost.model")
    model.save_model("catboost.model")
    y_pred = model.predict(X_valid)

    pprint(model.get_all_params())
    print("-------------------------------------------")
    print("F1 Score: {}".format(np.round(f1_score(y_valid, y_pred, average='macro'),3)))
    print("-------------------------------------------")
    print(classification_report(y_valid, y_pred > 0.5))
    print("-------------------------------------------")
    print(pd.DataFrame(model.feature_importances_, columns=["weight"],     index=[feature_names]).sort_values(by="weight", ascending=False).head(20)) 
    
    return model



trained_model = eval_cat_model(model, X_train_scaled, y_train, X_test_scaled, y_test, X_train.loc[:,selected].columns)

X_final_test = dd.from_pandas(data_test, npartitions=1).merge(feat_comm, on=['id','buy_time'])
X_final_test = X_final_test.compute().reset_index()

y_final_pred = trained_model.predict(X_final_test)

pd.DataFrame(y_final_pred)[0].value_counts()

def save_predictions(model, y_pred, X_test):
    target_df = pd.DataFrame(y_pred, columns=['target'])
    X_to_file = X_test.iloc[:,:4].reset_index()
    X_to_file = X_to_file.drop(columns=["level_0"])
    X_comp = pd.concat([X_to_file, target_df], axis=1).drop(columns=["index"])
    X_comp.to_csv(EXPORT_FILE, index=False)

save_predictions(trained_model, y_final_pred, X_final_test)

