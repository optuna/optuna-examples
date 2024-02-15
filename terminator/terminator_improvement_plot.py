"""
Optuna example showcasing the new Optuna Terminator feature.

This code visualizes Optuna Terminator using the 'plot_terminator_improvement'
function. It graphically depicts cross-validation errors and terminator
improvement reported during optimization.

The example optimizes hyperparameters of a RandomForestClassifier using
the wine dataset.

To run this example:

    $ python terminator_improvement_plot.py

"""

import optuna
from optuna.terminator.erroreval import report_cross_validation_scores
from optuna.visualization import plot_terminator_improvement

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def objective(trial):
    X, y = load_wine(return_X_y=True)

    clf = RandomForestClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 32),
        min_samples_split=trial.suggest_float("min_samples_split", 0, 1),
        criterion=trial.suggest_categorical("criterion", ("gini", "entropy")),
    )

    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
    report_cross_validation_scores(trial, scores)
    return scores.mean()


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=50)

    fig = plot_terminator_improvement(study, plot_error=True)
    fig.show()
