"""
Optuna example that optimizes a classifier configuration using OptunaSearchCV.

This example is the same as `sklearn/sklearn_optuna_search_cv_simple.py` except
that you leave termination of the study up to the terminator callback.
"""

import optuna
from optuna.terminator import TerminatorCallback

from sklearn.datasets import load_iris
from sklearn.svm import SVC


if __name__ == "__main__":
    clf = SVC(gamma="auto")

    param_distributions = {
        "C": optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
        "degree": optuna.distributions.IntDistribution(1, 5),
    }
    terminator = TerminatorCallback()

    optuna_search = optuna.integration.OptunaSearchCV(
        clf, param_distributions, n_trials=100, timeout=600, verbose=2, callbacks=[terminator]
    )

    X, y = load_iris(return_X_y=True)
    optuna_search.fit(X, y)

    print("Best trial:")
    trial = optuna_search.study_.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
