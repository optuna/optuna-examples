"""
Optuna example that optimizes a classifier configuration for the Iris dataset using
scikit-learn and records hyperparameters and metrics using Weights & Biases.

In this example we optimize random forest classifier for the Iris dataset. All
hyperparameters and metrics will be logged to Weights & Biases via integration callback.

Before running this example, please make sure to create and login into wandb account:
https://docs.wandb.ai/quickstart#1-set-up-wandb

You can run this example as follows:
    $ python wandb_integration.py

Results and plots will be available in Weights & Biases UI once script finishes.
"""

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

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
    wandb_kwargs = {"project": "optuna-wandb-example"}
    wandbc = WeightsAndBiasesCallback(metric_name="accuracy", wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[wandbc])

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
