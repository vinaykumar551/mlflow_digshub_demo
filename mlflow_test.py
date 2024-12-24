from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris

data = load_iris()

X = data.data[data.target != 2]
y = data.target[data.target != 2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
params={
    "C":56,
    "penalty":'l2',
    "solver":'liblinear',
    "random_state":71
}
logreg = LogisticRegression(**params)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

results = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_pred_proba)
}

import mlflow
# mlflow.set_experiment("sample_experiment")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics({
        'Accuracy':results['Accuracy'],
        'Precision':results['Precision'],
        'Recall':results['Recall'],
        'F1 Score':results['F1 Score'],
        'ROC AUC':results['ROC AUC']
    })
    mlflow.sklearn.log_model(logreg,'Log Reg')
model_name="sample_model2"
# run_id=input("Enter run id: ")
# model_uri=f"runs:/{run_id}/{model_name}"
# mlflow.register_model(
#     model_uri,model_name
# )
# model_version=2
# model_uri2=f"models:/{model_name}@challenger"
# load_model=mlflow.xgboost.load_model(model_uri2)
# y_pred=load_model.predict(X_test)
# print(y_pred)
dev_model_uri=f"models:/{model_name}@challenger"
prod_model='dest_prof'
client=mlflow.MlflowClient()
client.copy_model_version(src_model_uri=dev_model_uri,dst_name=prod_model)

