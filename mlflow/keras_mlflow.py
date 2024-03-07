"""
Optuna example that optimizes a neural network regressor for the
wine quality dataset using Keras and records hyperparameters and metrics using MLflow.

In this example, we optimize the learning rate and momentum of
stochastic gradient descent optimizer to minimize the validation mean squared error
for the wine quality regression.

You can run this example as follows:
    $ python keras_mlflow.py

After the script finishes, run the MLflow UI:
    $ mlflow ui

and view the optimization results at http://127.0.0.1:5000.
"""

import optuna
from optuna.integration.mlflow import MLflowCallback

from keras.backend import clear_session
from keras.layers import Dense
from keras.layers import Input
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TEST_SIZE = 0.25
BATCHSIZE = 16
EPOCHS = 100


def standardize(data):
    return StandardScaler().fit_transform(data)


def create_model(num_features, trial):
    model = Sequential()
    model.add(Input(shape=(num_features,)))
    model.add(
        Dense(
            num_features,
            activation="relu",
            kernel_initializer="normal",
        )
    ),
    model.add(Dense(16, activation="relu", kernel_initializer="normal"))
    model.add(Dense(16, activation="relu", kernel_initializer="normal"))
    model.add(Dense(1, kernel_initializer="normal", activation="linear"))

    optimizer = SGD(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        momentum=trial.suggest_float("momentum", 0.0, 1.0),
    )
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    X, y = load_wine(return_X_y=True)
    X = standardize(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    model = create_model(X.shape[1], trial)
    model.fit(X_train, y_train, shuffle=True, batch_size=BATCHSIZE, epochs=EPOCHS, verbose=False)

    return model.evaluate(X_valid, y_valid, verbose=0)


if __name__ == "__main__":
    mlflc = MLflowCallback(metric_name="mean_squared_error")
    study = optuna.create_study()
    study.optimize(objective, n_trials=100, timeout=600, callbacks=[mlflc])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
