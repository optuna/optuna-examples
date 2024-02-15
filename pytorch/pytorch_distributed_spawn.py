"""
Optuna example that optimizes multi-layer perceptrons using PyTorch distributed.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch distributed data parallel and FashionMNIST.
This example is dedicated to a use case of using one machine with multiple GPUs
where no MPI is needed.
We optimize the neural network architecture as well as the optimizer configuration.
As it is too time consuming to use the whole FashionMNIST dataset, we
here use a small subset of it.

You can execute this example with a command as follows.
Device ids such GPU ids can be specified with --device_ids argument:
    $ python pytorch_distributed_spawn.py --device-ids 1 2
Otherwise, CPU will be used as default.

To run more than 1 instances on the same machine, different port number must be passed:
    $ python pytorch_distributed_spawn.py --device-ids 3 --master-port 12356

Please note that this example only works with optuna >= 3.1.0.
If you wish to use optuna < 3.1.0, you would need to pass
`device=device_id` in TorchDistributedTrial.
"""

import argparse
from functools import partial
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms


BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
N_TRIALS = 20


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = nn.ModuleList()

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    # Load FashionMNIST dataset.
    train_dataset = datasets.FashionMNIST(DIR, train=True, transform=transforms.ToTensor())
    train_dataset = torch.utils.data.Subset(train_dataset, indices=range(N_TRAIN_EXAMPLES))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset)

    valid_dataset = datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor())
    valid_dataset = torch.utils.data.Subset(valid_dataset, indices=range(N_VALID_EXAMPLES))
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=valid_dataset, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCHSIZE,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=BATCHSIZE,
        shuffle=False,
    )

    return train_loader, valid_loader


def objective(single_trial, device_id):
    trial = optuna.integration.TorchDistributedTrial(single_trial)

    # Generate the model.
    model = DDP(
        define_model(trial).to(device_id), device_ids=None if device_id == "cpu" else [device_id]
    )

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_mnist()

    accuracy = 0
    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        # Shuffle train dataset.
        train_loader.sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.view(data.size(0), -1).to(device_id), target.to(device_id)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.view(data.size(0), -1).to(device_id), target.to(device_id)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        correct_tensor = torch.tensor([correct], dtype=torch.int).to(device_id)
        dist.all_reduce(correct_tensor)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(valid_loader.dataset)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Trial was pruned at epoch {epoch}.")

    return accuracy


def setup(backend, rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Using {backend} backend.")


def cleanup():
    dist.destroy_process_group()


def run_optimize(rank, world_size, device_ids, return_dict, master_port):
    device = "cpu" if len(device_ids) == 0 else device_ids[rank]
    print(f"Running basic DDP example on rank {rank} device {device}.")

    # Set environmental variables required by torch.distributed.
    backend = "gloo"
    if torch.distributed.is_nccl_available():
        if device != "cpu":
            backend = "nccl"
    setup(backend, rank, world_size, master_port)

    if rank == 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            partial(objective, device_id=device),
            n_trials=N_TRIALS,
            timeout=300,
        )
        return_dict["study"] = study
    else:
        for _ in range(N_TRIALS):
            try:
                objective(None, device)
            except optuna.TrialPruned:
                pass

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch distributed data-parallel training with spawn example."
    )
    parser.add_argument(
        "--device-ids",
        "-d",
        nargs="+",
        type=int,
        default=[0],
        help="Specify device_ids if using GPUs.",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA training."
    )
    parser.add_argument("--master-port", type=str, default="12355", help="Specify port number.")
    args = parser.parse_args()
    if args.no_cuda:
        device_ids = []
    else:
        device_ids = args.device_ids

    # Download dataset before starting the optimization.
    datasets.FashionMNIST(DIR, train=True, download=True)

    world_size = max(len(device_ids), 1)
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        run_optimize,
        args=(world_size, device_ids, return_dict, args.master_port),
        nprocs=world_size,
        join=True,
    )
    study = return_dict["study"]

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
