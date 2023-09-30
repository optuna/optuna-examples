"""
Optuna example showcasing the new Optuna Terminator feature.

In this example, we utilize the Optuna Terminator for hyperparameter
optimization on a LightGBM using the wine dataset.
The Terminator automatically stops the optimization process based
on the potential for further improvement.

To run this example:

    $ python lightgbm_terminator.py

"""

import optuna
from optuna.terminator.callback import TerminatorCallback
from optuna.terminator.erroreval import report_cross_validation_scores

import lightgbm as lgb
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def objective(trial):
    X, y = load_wine(return_X_y=True)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    clf = lgb.LGBMClassifier(**params)

    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
    report_cross_validation_scores(trial, scores)

    return scores.mean()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, callbacks=[TerminatorCallback()])

    print(f"The number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value} (params: {study.best_params})")
