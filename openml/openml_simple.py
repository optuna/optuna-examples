"""
====================================================
Hyperparameter Optimization Benchmark with OpenML
====================================================

In this tutorial, we walk through how to conduct hyperparameter optimization experiments using
OpenML and OptunaHub.
"""

############################################################################
# Please make sure to install the dependencies with:
# ``pip install -r requirements.txt``
# Then we import all the necessary modules.

# License: MIT License
import logging

import optuna

import openml
from openml.extensions.sklearn import cat
from openml.extensions.sklearn import cont
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.Logger(name="Experiment Logger", level=1)

# Set your openml api key if you want to upload your results to OpenML (eg:
# https://openml.org/search?type=run&sort=date) . To get one, simply make an
# account (you don't need one for anything else, just to upload your results),
# go to your profile and select the API-KEY.
# Or log in, and navigate to https://www.openml.org/auth/api-key
openml.config.apikey = ""

############################################################################
# Prepare for preprocessors and an OpenML task
# ============================================

# OpenML contains several key concepts which it needs to make machine learning research shareable.
# A machine learning experiment consists of one or several runs, which describe the performance of
# an algorithm (called a flow in OpenML), its hyperparameter settings (called a setup) on a task.
# A Task is the combination of a dataset, a split and an evaluation metric We choose a dataset from
# OpenML, (https://www.openml.org/d/1464) and a subsequent task (https://www.openml.org/t/10101) To
# make your own dataset and task, please refer to
# https://openml.github.io/openml-python/main/examples/30_extended/create_upload_tutorial.html

# https://www.openml.org/search?type=study&study_type=task&id=218
task_id = 10101
seed = 42
categorical_preproc = (
    "categorical",
    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
    cat,
)
numerical_preproc = ("numerical", SimpleImputer(strategy="median"), cont)
preproc = ColumnTransformer([categorical_preproc, numerical_preproc])

############################################################################
# Define a pipeline for the hyperparameter optimization (this is standark for Optuna)
# =====================================================

# Optuna explanation
# we follow the `Optuna <https://github.com/optuna/optuna/>`__ search space design.

# OpenML runs
# We can simply pass the parametrized classifier to `run_model_on_task` to obtain the performance
# of the pipeline
# on the specified OpenML task.
# Do you want to share your results along with an easily reproducible pipeline, you can set an API
# key and just upload your results.
# You can find more examples on https://www.openml.org/


def objective(trial: optuna.Trial) -> Pipeline:
    clf = RandomForestClassifier(
        max_depth=trial.suggest_int("max_depth", 2, 32, log=True),
        min_samples_leaf=trial.suggest_float("min_samples_leaf", 0.0, 1.0),
        random_state=seed,
    )
    pipe = Pipeline(steps=[("preproc", preproc), ("model", clf)])
    logger.log(1, f"Running pipeline - {pipe}")
    run = openml.runs.run_model_on_task(pipe, task=task_id, avoid_duplicate_runs=False)

    logger.log(1, f"Model has been trained - {run}")
    if openml.config.apikey != "":
        try:
            run.publish()

            logger.log(1, f"Run was uploaded to - {run.openml_url}")
        except Exception as e:
            logger.log(1, f"Could not publish run - {e}")
    else:
        logger.log(
            0,
            "If you want to publish your results to OpenML, please set an apikey",
        )
    accuracy = max(run.fold_evaluations["predictive_accuracy"][0].values())
    logger.log(0, f"Accuracy {accuracy}")

    return accuracy


############################################################################
# Optimize the pipeline
# =====================
study = optuna.create_study(direction="maximize")
logger.log(0, f"Study {study}")
study.optimize(objective, n_trials=15)

############################################################################
# Visualize the optimization history
# ==================================
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
