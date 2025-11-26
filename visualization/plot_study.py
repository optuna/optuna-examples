"""
Optuna example that visualizes the optimization result of multi-layer perceptrons.

In this example, we optimize the validation accuracy of object recognition using
scikit-learn and Fashion-MNIST. We optimize a neural network. As it is too time
consuming to use the whole Fashion-MNIST dataset, we here use a small subset of it.

We can execute this example as follows.

    $ python plot_study.py

**Note:** If a parameter contains missing values, a trial with missing values is not plotted.
"""

from mnists import FashionMNIST
import numpy as np
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def objective(trial):
    # Load Fashion-MNIST dataset using mnists
    fmnist = FashionMNIST()
    X = fmnist.train_images()  # shape: (n_samples, 28, 28)
    y = fmnist.train_labels()  # shape: (n_samples,)

    # For demonstrational purpose, only use a subset of the dataset.
    n_samples = 4000
    X = X[:n_samples]
    y = y[:n_samples]

    # Flatten images (28x28 -> 784)
    X = X.reshape(n_samples, -1).astype(np.float32)

    classes = list(set(y))

    x_train, x_valid, y_train, y_valid = train_test_split(X, y)
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(
            [trial.suggest_int("n_units_l{}".format(i), 32, 64) for i in range(3)]
        ),
        learning_rate_init=trial.suggest_float("lr_init", 1e-5, 1e-1, log=True),
    )

    for step in range(100):
        clf.partial_fit(x_train, y_train, classes=classes)
        value = clf.score(x_valid, y_valid)

        # Report intermediate objective value.
        trial.report(value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return value


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100, timeout=600)

    # Visualize the optimization history.
    plot_optimization_history(study).show()

    # Visualize the learning curves of the trials.
    plot_intermediate_values(study).show()

    # Visualize high-dimensional parameter relationships.
    plot_parallel_coordinate(study).show()

    # Select parameters to visualize.
    plot_parallel_coordinate(study, params=["lr_init", "n_units_l0"]).show()

    # Visualize hyperparameter relationships.
    plot_contour(study).show()

    # Select parameters to visualize.
    plot_contour(study, params=["n_units_l0", "n_units_l1"]).show()

    # Visualize individual hyperparameters.
    plot_slice(study).show()

    # Select parameters to visualize.
    plot_slice(study, params=["n_units_l0", "n_units_l1"]).show()

    # Visualize parameter importances.
    plot_param_importances(study).show()
