
import comet_ml
import optuna

comet_ml.login(project_name="comet-sklearn-example")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

random_state = 42
def evaluate(y_test, y_pred):
    return {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
experiment = comet_ml.start()
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=random_state
)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
with experiment.train():
    metrics = evaluate(y_train, y_train_pred)
    experiment.log_metrics(metrics)

y_test_pred = clf.predict(X_test)

with experiment.test():
    metrics = evaluate(y_test, y_test_pred)
    experiment.log_metrics(metrics)

experiment.end()