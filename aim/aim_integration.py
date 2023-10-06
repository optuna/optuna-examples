"""
Optuna example that optimizes a classifier configuration for the Iris dataset using
scikit-learn and records hyperparameters and metrics using aim.

In this example we optimize random forest classifier for the Iris dataset. All
hyperparameters and metrics will be logged to aim via integration callback.

You can run this example as follows:
    $ python aim_integration.py

Results and plots will be available in aim ui once script finishes:
    $ aim up

and view the optimization results at http://127.0.0.1:43800.
"""

import optuna

from aim.optuna import AimCallback
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def objective(trial):
    data = load_iris()
    x_train, x_valid, y_train, y_valid = train_test_split(data["data"], data["target"])

    params = {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_valid)
    score = accuracy_score(y_valid, pred)

    return score


if __name__ == "__main__":
    aim_callback = AimCallback(experiment_name="optuna_aim-example")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[aim_callback])

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
