import logging
from typing import Callable
import warnings

import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold


logger = logging.getLogger("trainkit")


def train_valid_splits_by_counts(train_count, valid_count):
    """Generate indices for training and validation sets based on specified counts.

    This function creates a list of tuples, where each tuple contains
    two numpy arrays: the first array contains indices for the training set,
    and the second array contains indices for the validation set.

    Parameters
    ----------
    train_count: int
        The row count of the training set.
    valid_count: int
        The row count of the validation set.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        A list containing one tuple of two numpy arrays, where the first array
        is the training indices and the second array is the validation indices.
    """
    return [
        (
            np.arange(train_count),
            np.arange(train_count, train_count + valid_count),
        )
    ]


def get_splits(X, split_count=2):
    """Generate training and validation indices for K-Fold cross-validation.

    A list of tuples is created in which each tuple contains two arrays:
    the first array contains indices for the training set, and
    the second array contains indices for the validation set.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The data used for creating splits.
    split_count : int, optional
        The number of splits/folds to create. Default is 2.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of tuples containing the training indices and validation indices.
    """
    kf = KFold(n_splits=split_count, shuffle=True, random_state=42)
    return [(train_idx, valid_idx) for train_idx, valid_idx in kf.split(X)]


def compute_loss(
    model_params: dict[str, any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple],
    selected_feature_names: list[str] | None = None,
    loss_fn: Callable[[pd.Series, pd.Series], float] = root_mean_squared_error,
) -> float:
    """Compute the loss for a given model configuration and dataset.

    Parameters
    ----------
    model_params : dict[str, any]
        Parameters for the model to be trained.
    X : pd.DataFrame
        The input features dataframe containing corresponding values
        for both training and validation sets.
    y : pd.Series
        The target variable series containing corresponding values
        for both training and validation sets.
    splits : list[tuple]
        List of tuples of indices for training and validation sets,
        used to split X and y into respective training and validation subsets.
    selected_feature_names : list[str], optional
        List of feature names to be used. If None, all features in X are used.
    loss_fn : function, optional
        The loss function to be used. Defaults to root_mean_squared_error.

    Returns
    -------
    float
        The average loss computed over all the validation splits.
    """
    loss = 0
    iteration = 0
    if not selected_feature_names:
        selected_feature_names = list(X.columns)

    for train_idx, valid_idx in splits:
        iteration += 1
        X_train_split = X.iloc[train_idx][selected_feature_names]
        y_train_split = y.iloc[train_idx]
        X_valid_split = X.iloc[valid_idx][selected_feature_names]
        y_valid_split = y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train_split, label=y_train_split)
        valid_data = lgb.Dataset(X_valid_split, label=y_valid_split, reference=train_data)
        with warnings.catch_warnings():  # Python 3.11: (action="ignore"):
            warnings.simplefilter("ignore")
            gbm = lgb.train(
                model_params,
                train_data,
                valid_sets=[valid_data],
            )

        pred = gbm.predict(X_valid_split, num_iteration=gbm.best_iteration)
        iter_loss = loss_fn(y_valid_split, pred)
        loss += iter_loss
        if iteration > 1:
            logger.debug(
                "[compute_loss] RMSE: %07.3f ( ~%07.3f)",
                iter_loss,
                loss / iteration,
            )

    return loss / len(splits)
