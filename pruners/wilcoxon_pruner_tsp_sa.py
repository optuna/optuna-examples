from dataclasses import dataclass
import math
import sys

import numpy as np
from numpy.linalg import norm
import optuna


@dataclass
class SAOptions:
    max_iter: int = 10000
    T0: float = 1.0
    alpha: float = 2.0
    patience: int = 50


def tsp_cost(vertices: np.ndarray, idxs: np.ndarray) -> float:
    return norm(vertices[idxs] - vertices[np.roll(idxs, 1)], axis=-1).sum()


# Greedy solution for initial guess.
def tsp_greedy(vertices: np.ndarray) -> np.ndarray:
    idxs = [0]
    for _ in range(len(vertices) - 1):
        dists_from_last = norm(vertices[idxs[-1], None] - vertices, axis=-1)
        dists_from_last[idxs] = np.inf
        idxs.append(np.argmin(dists_from_last))
    return np.array(idxs)


# A minimal implementation of TSP solver using simulated annealing on 2-opt neighbors.
def tsp_simulated_annealing(vertices: np.ndarray, options: SAOptions) -> np.ndarray:

    def temperature(t: float):
        # t: 0 ... 1
        return options.T0 * (1 - t) ** options.alpha

    N = len(vertices)

    idxs = tsp_greedy(vertices)
    cost = tsp_cost(vertices, idxs)
    best_idxs = idxs.copy()
    best_cost = cost
    remaining_patience = options.patience

    for iter in range(options.max_iter):

        i = np.random.randint(0, N)
        j = (i + 2 + np.random.randint(0, N - 3)) % N
        i, j = min(i, j), max(i, j)
        # Reverse the order of vertices between range [i+1, j].

        # cost difference by 2-opt reversal
        delta_cost = (
            -norm(vertices[idxs[(i + 1) % N]] - vertices[idxs[i]])
            - norm(vertices[idxs[j]] - vertices[idxs[(j + 1) % N]])
            + norm(vertices[idxs[i]] - vertices[idxs[j]])
            + norm(vertices[idxs[(i + 1) % N]] - vertices[idxs[(j + 1) % N]])
        )
        temp = temperature(iter / options.max_iter)
        if delta_cost <= 0.0 or np.random.random() < math.exp(-delta_cost / temp):
            # accept the 2-opt reversal
            cost += delta_cost
            idxs[i + 1 : j + 1] = idxs[i + 1 : j + 1][::-1]
            if cost < best_cost:
                best_idxs[:] = idxs
                best_cost = cost
                remaining_patience = options.patience

        if cost > best_cost:
            # If the best solution is not updated for "patience" iteratoins,
            # restart from the best solution.
            remaining_patience -= 1
            if remaining_patience == 0:
                idxs[:] = best_idxs
                cost = best_cost
                remaining_patience = options.patience

    return best_idxs


def make_dataset(num_vertex: int, num_problem: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.random((num_problem, num_vertex, 2))


dataset = make_dataset(
    num_vertex=100,
    num_problem=50,
)

N_TRIALS = 50

# We set a very small number of SA iterations for demonstration purpose.
# In practice, you should set a larger number of iterations.
N_SA_ITER = 10000
count = 0


def objective(trial: optuna.Trial) -> float:
    global count
    options = SAOptions(
        max_iter=N_SA_ITER,
        T0=trial.suggest_float("T0", 0.01, 10.0, log=True),
        alpha=trial.suggest_float("alpha", 1.0, 10.0, log=True),
        patience=trial.suggest_int("patience", 10, 1000, log=True),
    )
    results = []

    # For best results, shuffle the evaluation order in each trial.
    ordering = np.random.permutation(len(dataset))
    for i in ordering:
        count += 1
        result_idxs = tsp_simulated_annealing(vertices=dataset[i], options=options)
        result_cost = tsp_cost(dataset[i], result_idxs)
        results.append(result_cost)

        trial.report(result_cost, i)
        if trial.should_prune():
            print(
                f"[{trial.number}] Pruned at {len(results)}/{len(dataset)}",
                file=sys.stderr,
            )
            # raise optuna.TrialPruned()

            # Return the current predicted value when pruned.
            # This is a workaround for the problem that
            # current TPE sampler cannot utilize pruned trials effectively.
            return sum(results) / len(results)

    print(f"[{trial.number}] Not pruned ({len(results)}/{len(dataset)})", file=sys.stderr)
    return sum(results) / len(results)


if __name__ == "__main__":
    np.random.seed(0)
    sampler = optuna.samplers.TPESampler(seed=1)
    pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.enqueue_trial({"T0": 1.0, "alpha": 2.0, "patience": 50})  # default params
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"The number of trials: {len(study.trials)}")
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    print(f"Number of evaluations: {count} / {N_TRIALS * len(dataset)}")
