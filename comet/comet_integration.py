"""
This script integrates Comet ML and Optuna to optimize a Random Forest Classifier
on the scikit-learn Breast Cancer dataset. It performs the following steps:

1. Initializes a Comet ML experiment for logging.
2. Loads the Breast Cancer dataset and splits it into training and testing sets.
3. Defines an evaluation function using F1-score, precision, and recall.
4. Implements an Optuna objective function to optimize hyperparameters
   (n_estimators and max_depth) for the Random Forest model.
5. Uses Optuna to run multiple trials and identify the best hyperparameters.
6. Trains the final Random Forest model using the best-found hyperparameters.
7. Logs training and testing metrics to Comet ML.

You can run this example as follows:
    $ python comet_integration.py
"""

import optuna
from optuna_integration.comet import CometCallback

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def objective(trial):
    """Objective function for optimizing a RandomForestClassifier using Optuna."""
    data = load_iris()
    x_train, x_valid, y_train, y_valid = train_test_split(
        data["data"], data["target"], random_state=42
    )
    params = {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_valid)
    score = accuracy_score(y_valid, pred)

    return score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    comet_callback = CometCallback(study, project_name="comet-optuna-sklearn-example")

    study.optimize(objective, n_trials=20, callbacks=[comet_callback])

    print(f"Number of finished trials: {len(study.trials)}\n")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}\n")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
