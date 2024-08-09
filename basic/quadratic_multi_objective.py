"""
Optuna example that optimizes simple quadratic functions.

In this example, we optimize two objective values.
Unlike single-objective optimization, an optimization gives a trade-off between two objectives.
As a result, we get the best trade-offs between two objectives, a.k.a Pareto solutions.

"""

import optuna


# Define two objective functions.
# We would like to minimize f1 and maximize f2.
def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    f1 = x**2 + y
    f2 = -((x - 2) ** 2 + y)
    return f1, f2


if __name__ == "__main__":
    # We minimize the first objective value and maximize the second objective value.
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(objective, n_trials=500, timeout=1)

    print("Number of finished trials: ", len(study.trials))

    for i, best_trial in enumerate(study.best_trials):
        print("The {}-th Pareto solution was found at Trial#{}.".format(i, best_trial.number))
        print("  Params: {}".format(best_trial.params))
        f1, f2 = best_trial.values
        print("  Values: f1={}, f2={}".format(f1, f2))
