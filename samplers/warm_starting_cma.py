"""
Optuna example using warm starting CMA-ES.

In this example, we first optimize a *biased* sphere function as a source task.
With the result of optimization, we optimize a sphere function as a target task
by using warm starting CMA-ES.

"""

import optuna


# 2-dimensional box-constrained sphere function
# The optimum is (x, y) = (0, 0)
def sphere(trial):
    x = trial.suggest_float("x", -15, 15)
    y = trial.suggest_float("y", -15, 15)
    return x**2 + y**2


# 2-dimensional box-constrained sphere function
# This function is *biased*; the optimum is (x, y) = (1, 1)
def biased_sphere(trial):
    x = trial.suggest_float("x", -15, 15)
    y = trial.suggest_float("y", -15, 15)
    return (x - 1) ** 2 + (y - 1) ** 2


if __name__ == "__main__":
    # Perform optimization on a source task
    cma = optuna.samplers.CmaEsSampler()
    source_study = optuna.create_study(sampler=cma)
    source_study.optimize(biased_sphere, n_trials=50)
    print(
        f"Best value on the source task: {source_study.best_value},"
        f" (params: {source_study.best_params}\n"
    )

    # Perform optimization on a target task by warm starting CMA-ES
    ws_cma = optuna.samplers.CmaEsSampler(source_trials=source_study.trials)
    target_study = optuna.create_study(sampler=ws_cma)
    target_study.optimize(sphere, n_trials=50)
    print(
        f"Best value on the target task: {target_study.best_value},"
        f" (params: {target_study.best_params}\n"
    )
