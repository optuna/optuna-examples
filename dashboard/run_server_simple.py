import optuna
from optuna_dashboard import run_server


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


if __name__ == "__main__":
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage, study_name="dashboard-example")
    study.optimize(objective, n_trials=100)

    # Start the Optuna Dashboard server on localhost:8080
    run_server(storage, host="localhost", port=8080)
