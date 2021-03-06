"""
Optuna example that demonstrates a pruner for Chainer.

In this example, we optimize the hyperparameters of a neural network for fashion
product recognition in terms of validation loss. The network is implemented by Chainer and
evaluated by FashionMNIST dataset. Throughout the training of neural networks, a pruner observes
intermediate results and stops unpromising trials.

You can run this example as follows:
    $ python chainer_integration.py

"""

import numpy as np
import optuna
from optuna.trial import TrialState
from packaging import version

import chainer
import chainer.functions as F
import chainer.links as L


if version.parse(chainer.__version__) < version.parse("4.0.0"):
    raise RuntimeError("Chainer>=4.0.0 is required for this example.")

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10
PRUNER_INTERVAL = 3


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 32, 256, log=True)
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    model = L.Classifier(create_model(trial))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    rng = np.random.RandomState(0)
    train, valid = chainer.datasets.get_fashion_mnist()
    train = chainer.datasets.SubDataset(
        train, 0, N_TRAIN_EXAMPLES, order=rng.permutation(len(train))
    )
    valid = chainer.datasets.SubDataset(
        valid, 0, N_VALID_EXAMPLES, order=rng.permutation(len(valid))
    )
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)
    valid_iter = chainer.iterators.SerialIterator(valid, BATCHSIZE, repeat=False, shuffle=False)

    # Setup trainer.
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, "epoch"))

    # Add Chainer extension for pruners.
    trainer.extend(
        optuna.integration.ChainerPruningExtension(
            trial, "validation/main/accuracy", (PRUNER_INTERVAL, "epoch")
        )
    )

    trainer.extend(chainer.training.extensions.Evaluator(valid_iter, model))
    trainer.extend(
        chainer.training.extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "validation/main/loss",
                "main/accuracy",
                "validation/main/accuracy",
            ]
        )
    )
    log_report_extension = chainer.training.extensions.LogReport(log_name=None)
    trainer.extend(log_report_extension)

    # Run training.
    # Please set show_loop_exception_msg False to inhibit messages about TrialPruned exception.
    # ChainerPruningExtension raises TrialPruned exception to stop training, and
    # trainer shows some messages every time it receive TrialPruned.
    trainer.run(show_loop_exception_msg=False)

    # Save loss and accuracy to user attributes.
    log_last = log_report_extension.log[-1]
    for key, value in log_last.items():
        trial.set_user_attr(key, value)

    return log_report_extension.log[-1]["validation/main/accuracy"]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
