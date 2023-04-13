"""

In this example, we optimize a custom LSTM model by minimizing the validation loss.

"""

import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from optuna.trial import TrialState
from pytorchtools import EarlyStopping
from tqdm.notebook import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


BATCH_SIZE = 32
EPOCHS = 20
criterion = torch.nn.MSELoss()


class CustomModel(nn.Sequential):
    def __init__(self, n_layers, n_hidden, input_dim, dropout_ps, out_features):
        super(CustomModel, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.input_dim = input_dim
        self.dropout_ps = dropout_ps
        self.out_features = out_features

        for i in range(n_layers):
            self.add_module(
                "lstm{}".format(i),
                nn.LSTM(self.input_dim, self.n_hidden, self.n_layers, batch_first=True),
            )
            self.add_module("dropout{}".format(i), nn.Dropout(self.dropout_ps[i]))

        self.add_module("linear", nn.Linear(self.n_hidden, out_features=self.out_features))

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(2), self.n_hidden).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.n_layers, x.size(2), self.n_hidden).requires_grad_().to(DEVICE)
        x = torch.permute(
            x, (2, 0, 1)
        )  # permute the input to match The desired ordering of dimensions for lstm layer
        for module in self._modules.values():
            if isinstance(module, nn.LSTM):
                out = module(x, (h0, c0))[0]
        out = out[:, -1, :]
        out = self._modules["linear"](out)
        return out


# your sequential dataset depends on your specific task
def get_data(train_dataset, test_dataset):
    train = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    return train, test


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 15)
    n_hidden = trial.suggest_int("n_hidden", 1, 50)
    dropout_ps = [trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5) for i in range(n_layers)]

    return CustomModel(n_layers, n_hidden, dropout_ps, out_features=1)


def objective(self, trial):
    # Generate the model
    model = self.define_model(trial).to(DEVICE)
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-8, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    train_loader, test_loader = self.get_data()
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_loss = 0.0
        for train_batch in train_loader:
            data = train_batch["sequence"].to(
                DEVICE
            )  # data.shape = (batch_size, sequence_length, features)
            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, train_batch["label"].to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        test_loss = 0.0
        with torch.no_grad():
            for test_batch in train_loader:
                data = test_batch["sequence"].to(DEVICE)

                output = model.forward(data)
                loss = criterion(output, test_batch["label"].to(DEVICE))
                test_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        trial.report(test_loss, epoch)
        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="lstm-optimization")
    study.optimize(objective, n_trials=100, timeout=600)

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
