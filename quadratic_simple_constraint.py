"""
Optuna example that optimizes a simple quadratic function with constraints.

In this example, we optimize a simple quadratic function with constraints.

Please check https://optuna.readthedocs.io/en/stable/faq.html#id16 as well.

"""

import optuna


# Define a simple 2-dimensional objective function with constraints.
def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


# Define a function that returns constraints.
def constraints(trial):
    params = trial.params
    x, y = params["x"], params["y"]
    # c1 <= 0 and c2 <= 0 must be satisfied.
    c1 = 3 * x * y + 10
    c2 = x * y + 30
    return [c1, c2]


if __name__ == "__main__":
    # We minimize obj1 and maximize obj2.
    sampler = optuna.samplers.TPESampler(constraints_func=constraints)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=500, timeout=1)

    best_trial_id, best_value, best_params = None, float("inf"), None
    for t in study.trials:
        infeasible = any(c > 0.0 for c in t.system_attrs["constraints"])
        if infeasible:
            continue
        if best_value > t.value:
            best_value = t.value
            best_params = t.params.copy()
            best_trial_id = t._trial_id

    if best_trial_id is None:
        print("All trials violated the constraints.")
    else:
        print(f"Best value is {best_value} with params {best_params}")
