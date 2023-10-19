"""
Optuna example that optimizes simple quadratic functions.

In this example, we optimize two objective values.
Unlike a single-objective optimization, an optimization gives the trade-off between two objectives.
As a result, we get best trade-offs between two objectives, a.k.a Pareto solutions.

"""

import optuna


# Define two objective functions.
# We would like to minimize obj1 and maximize obj2.
def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    obj1 = x**2 + y
    obj2 = -((x - 2) ** 2 + y)
    return obj1, obj2


if __name__ == "__main__":
    # We minimize the first objective value and maximize the second objective value.
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(objective, n_trials=500, timeout=1)

    print("Number of finished trials: ", len(study.trials))

    for i, best_trial in enumerate(study.best_trials):
        print(f"The {i}-th Pareto solution was found at Trial#{best_trial.number}.")
        print(f"  Params: {best_trial.params}")
        print(f"  Values: {best_trial.values}")
