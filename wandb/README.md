# Tracking optimization process with Weights & Biases

Optuna example that optimizes a classifier configuration for the Iris dataset usinng
scikit-learn and records hyperparameters and metrics using Weights & Biases.

In this example we optimize random forest classifier for the Iris dataset. All
hyperparameters and metrics will be logged to Weights & Biases via integration callback.

Before running this example, please make sure to [create and login](https://docs.wandb.ai/quickstart#1-set-up-wandb) into wandb account,
or switch to offline mode with:
```
$ wandb offline
```

You can run this example as follows:
```
$ python wandb_simple.py
```

Results and plots will be available in Weights & Biases UI once script finishes.