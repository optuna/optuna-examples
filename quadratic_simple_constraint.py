"""
Optuna example that optimizes a simple quadratic function with constraints.

This setup is called `constrained optimization`.
Constrained optimization is useful when we would like to optimize some metrics under some constraints.
For example, we would like to maximize the accuracy of a deep neural networks while guaranteeing that
deep neural networks fit on your hardware, e.g. 8GB of memory consumption.
In this case, constrained optimization aims to yield an optimal solution that satisfied such constraints.

Please also check https://optuna.readthedocs.io/en/stable/faq.html#id16 as well.

"""

import optuna


# Define a simple 2-dimensional objective function to minimize.
def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


# Define a function that returns constraints.
# The constraints are to satisfy `c1 <= 0` and `c2 <= 0` simultaneously.
# If we would like to make the constraint like `c1 <= a`, we simply need to return `c1 - a` instead of `c1`.
def constraints(trial):
    params = trial.params
    x, y = params["x"], params["y"]
    c1 = 3 * x * y + 10
    c2 = x * y + 30
    return c1, c2


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(constraints_func=constraints)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=500, timeout=1)

    print("Number of finished trials: ", len(study.trials))

    feasible_trial_ids = [
        trial.number
        for trial in study.trials
        if all(c <= 0.0 for c in trial.system_attrs["constraints"])
    ]
    if len(feasible_trial_ids) == 0:
        print("No trials satisfied all the constraints.")
    else:
        best_trial_id = sorted(feasible_trial_ids, key=lambda i: study.trials[i].value)[0]
        best_trial = study.trials[best_trial_id]
        print(f"Best trial was found at Trial#{best_trial_id}")
        print(f"  Params: {best_trial.params}")
        print(f"  Value: {best_trial.value}")
        c1, c2 = best_trial.system_attrs['constraints']
        print(f"  Constraints: {c1=}, {c2=}")
