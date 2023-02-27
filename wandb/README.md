# Tracking optimization process with Weights & Biases

![wandbui](https://user-images.githubusercontent.com/37713008/133468378-3eae55e3-dd07-4e87-b3cf-f60c06fd061d.png)

Optuna example that optimizes a classifier configuration for the Iris dataset using
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
$ python wandb_integration.py
```

Results and plots will be available in Weights & Biases UI once script finishes.

See also our [Medium post](https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893) for additional use-case.
