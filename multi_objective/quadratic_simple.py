"""
Optuna example that optimizes simple quadratic functions.

In this example, we optimize simple quadratic functions.

"""

import optuna


# Define simple 2-dimensional objective functions.
def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    obj1 = x**2 + y
    obj2 = -((x - 2) ** 2 + y)
    return [obj1, obj2]


if __name__ == "__main__":
    # We minimize obj1 and maximize obj2.
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(objective, n_trials=500, timeout=1)

    pareto_front = [t.values for t in study.best_trials]
    pareto_sols = [t.params for t in study.best_trials]

    for i, (params, values) in enumerate(zip(pareto_sols, pareto_front)):
        print(f"The {i}-th Pareto solution and its objective values")
        print("\t", params, values)
