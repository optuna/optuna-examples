"""
Optuna example that demonstrates a pruner for XGBoost.cv.

In this example, we optimize the validation auc of cancer detection using XGBoost.
We optimize both the choice of booster model and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stops unpromising trials,
and a terminator observes studies and stops optimizations with insignificant changes.

You can run this example as follows:
    $ python xgboost_cv_terminator.py

"""

import optuna
from optuna.terminator import report_cross_validation_scores
from optuna.terminator import TerminatorCallback

import sklearn.datasets
import xgboost as xgb


def objective(trial):
    train_x, train_y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    dtrain = xgb.DMatrix(train_x, label=train_y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
    param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
    param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
    param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    history = xgb.cv(param, dtrain, num_boost_round=100)

    report_cross_validation_scores(trial, history["test-auc-mean"])

    mean_auc = history["test-auc-mean"].values[-1]
    return mean_auc


if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    terminator = TerminatorCallback()
    study = optuna.create_study(pruner=pruner, direction="maximize", callbacks=[terminator])
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
