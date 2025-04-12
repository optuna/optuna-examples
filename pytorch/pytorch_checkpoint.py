"""
Optuna example that optimizes multi-layer perceptrons using PyTorch with checkpoint.

In this example, we optimize the validation accuracy of fastion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

Even if the process where the trial is running is killed for some reason, you can restart from
previous saved checkpoint using heartbeat.

    $ timeout 20 python pytorch/pytorch_checkpoint.py
    $ python pytorch/pytorch_checkpoint.py
"""

import os
import tempfile

import optuna
from optuna.artifacts import download_artifact
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna.storages import RetryFailedTrialCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms


DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
CHECKPOINT_DIR = "pytorch_checkpoint"

base_path = "./artifacts"
os.makedirs(base_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=base_path)


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    return train_loader, valid_loader


def objective(trial):
    model = define_model(trial).to(DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    artifact_id = None
    retry_history = RetryFailedTrialCallback.retry_history(trial)
    for trial_number in reversed(retry_history):
        artifact_id = trial.study.trials[trial_number].user_attrs.get("artifact_id")
        if artifact_id is not None:
            retry_trial_number = trial_number
            break

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

        if artifact_id is not None:
            download_artifact(
                artifact_store=artifact_store,
                file_path=checkpoint_path,
                artifact_id=artifact_id,
            )
            checkpoint = torch.load(checkpoint_path)
            epoch_begin = checkpoint["epoch"] + 1

            print(
                f"Loading a checkpoint from trial {retry_trial_number} in epoch {checkpoint['epoch']}."
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            accuracy = checkpoint["accuracy"]
        else:
            epoch_begin = 0

        train_loader, valid_loader = get_mnist()

        for epoch in range(epoch_begin, EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_loader):
                    if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                        break
                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)
            trial.report(accuracy, epoch)

            print(f"Saving a checkpoint in epoch {epoch}.")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                },
                checkpoint_path,
            )

            artifact_id = upload_artifact(
                artifact_store=artifact_store,
                file_path=checkpoint_path,
                study_or_trial=trial,
            )
            trial.set_user_attr("artifact_id", artifact_id)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage,
        study_name="pytorch_checkpoint",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    optuna.visualization.plot_intermediate_values(study).show()
