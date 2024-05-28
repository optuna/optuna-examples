#!/usr/bin/env python3

import logging

from feature_selection import feature_removal_cv

from sklearn.datasets import fetch_california_housing


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

df = fetch_california_housing(as_frame=True).frame

print("\nBasic info on the dataset:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display the columns
print("\nColumn names:")
print(df.columns.to_list())

# Show basic stats on the 'passengers' column
print("\nBasic statistics on the 'MedHouseVal' column:")
print(df.MedHouseVal.describe())

feature_removal_cv(
    model_params={
        "objective": "regression",
        "metric": "rmse",
        "data_random_seed": 42,
        "num_boost_round": 1000,
        "early_stopping_rounds": 10,
        "learning_rate": 0.12599281729053988,
        "force_row_wise": True,
        "verbose": -1,
        "verbose_eval": False,
        "num_leaves": 631,
        "max_depth": 7,
        "min_child_samples": 65,
        "colsample_bytree": 0.8430078242019065,
        "reg_alpha": 0.06636017620531826,
        "reg_lambda": 0.057077523364489346,
    },
    X=df.drop(columns=["MedHouseVal"]),
    y=df.MedHouseVal,
    split_count=5,
    trial_count=1000,
)
