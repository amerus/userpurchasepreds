#!/usr/bin/env python
# coding: utf-8

EXPORT_FILE="predictions.csv"
TRAINING_DATA="data_train.csv"
TEST_DATA="data_test.csv"
PROFILE_FEATURES="features.csv"
MODEL_NAME="catboost.model"

import sklearn.externals
import joblib
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from dask_ml.preprocessing import StandardScaler
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split


data_test = pd.read_csv(TEST_DATA, index_col=0)
data_train = pd.read_csv(TRAINING_DATA, index_col=0)
features = dd.read_csv(PROFILE_FEATURES, sep="\t").set_index("Unnamed: 0")

test_ids = data_test.id.to_list()
train_ids = data_train.id.to_list()
train_and_test = list(set(train_ids + test_ids))
feat_comm = features.loc[features["id"].isin(train_and_test), :]

X_final_test = dd.from_pandas(data_test, npartitions=1).merge(feat_comm, on=['id','buy_time'], how="left")
X_final_test = X_final_test.compute().reset_index()

for item in X_final_test.columns.tolist():
    X_final_test[item] = X_final_test[item].fillna(0)

import gc
del features
gc.collect()
features = pd.DataFrame()

scaler = StandardScaler()
X_final_scaled = pd.concat([X_final_test.iloc[:,:4], scaler.fit_transform(X_final_test.iloc[:,4:])], axis=1)
model = CatBoostClassifier(n_estimators=1400, max_depth=6, eval_metric='Precision',    random_state=42, class_weights=[0.1, 10], task_type="CPU")
trained_model = model.load_model(MODEL_NAME)

y_final_pred = trained_model.predict(X_final_scaled)
pd.DataFrame(y_final_pred)[0].value_counts()

def save_predictions(model, y_pred, X_test):
    target_df = pd.DataFrame(y_pred, columns=['target'])
    X_to_file = X_test.iloc[:,:4].reset_index()
    X_to_file = X_to_file.drop(columns=["level_0"])
    X_comp = pd.concat([X_to_file, target_df], axis=1).drop(columns=["index"])
    X_comp.to_csv(EXPORT_FILE, index=False)
    
save_predictions(trained_model, y_final_pred, X_final_test)

