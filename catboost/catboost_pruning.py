"""
Optuna example that demonstrates a pruner for CatBoost.

In this example, we optimize the validation accuracy of cancer detection using CatBoost.
We optimize both the choice of booster models and their hyperparameters. Throughout
training of models, a pruner observes intermediate results and stop unpromising trials.

You can run this example as follows:
    $ python catboost_pruning.py

"""

from typing import Any

import numpy as np
import optuna

import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class CatBoostPruningCallback(object):
    """Callback for catboost to prune unpromising trials.

    You must call `check_pruned` after training manually unlike other pruning callbacks.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        metric:
            An evaluation metric for pruning, e.g., ``Logloss`` and ``AUC``.
            Please refer to
            `CatBoost reference
            <https://catboost.ai/docs/references/eval-metric__supported-metrics.html>`_
            for further details.
        valid_name:
            The name of the target validation.
            If you set only one ``eval_set``, ``validation`` is used.
            If you set multiple datasets as ``eval_set``, the index number of ``eval_set`` must be
            included in the valid_name, e.g., ``validation_0`` or ``validation_1``
            when ``eval_set`` contains two datasets.
    """

    def __init__(
        self, trial: optuna.trial.Trial, metric: str, valid_name: str = "validation"
    ) -> None:
        self._trial = trial
        self._metric = metric
        self._valid_name = valid_name
        self._pruned = False
        self._message = ""

    def after_iteration(self, info: Any) -> bool:
        """Run after each iteration.

        Args:
            info:
                A ``simpleNamespace`` containing iteraion, ``validation_name``, ``metric_name``
                and history of losses.
                For example ``namespace(iteration=2, metrics= {
                'learn': {'Logloss': [0.6, 0.5]},
                'validation': {'Logloss': [0.7, 0.6], 'AUC': [0.8, 0.9]}
                })``.

        Returns:
            A boolean value. If :obj:`True`, the trial should be pruned.
            Otherwise, the trial should continue.
        """
        step = info.iteration - 1
        if self._valid_name not in info.metrics:
            raise ValueError(
                'The entry associated with the validation name "{}" '
                "is not found in the evaluation result list {}.".format(self._valid_name, info)
            )
        metrics = info.metrics[self._valid_name]
        if self._metric not in metrics:
            raise ValueError(
                'The entry associated with the metric name "{}" '
                "is not found in the evaluation result list {}.".format(self._metric, info)
            )
        current_score = metrics[self._metric][-1]
        self._trial.report(current_score, step=step)
        if self._trial.should_prune():
            self._message = "Trial was pruned at iteration {}.".format(step)
            self._pruned = True
            return False
        return True

    def check_pruned(self) -> None:
        """Check whether pruend."""
        if self._pruned:
            raise optuna.TrialPruned(self._message)


def objective(trial: optuna.Trial) -> float:
    data, target = load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "used_ram_limit": "3gb",
        "eval_metric": "Accuracy",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = cb.CatBoostClassifier(**param)

    pruning_callback = CatBoostPruningCallback(trial, "Accuracy", "validation")
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(valid_y, pred_labels)

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
