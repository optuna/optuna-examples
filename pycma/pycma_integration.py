"""
Optuna example that demonstrates a CMA-ES sampler using PyCmaSampler.

In this example, we optimize the hyperparameters of a RandomForestClassifier on the Iris dataset.
We use the PyCmaSampler, which wraps CMA-ES (Covariance Matrix Adaptation Evolution Strategy),
to sample hyperparameters such as the number of estimators, maximum depth, and feature selection strategy.

This example also demonstrates how to use Optuna for meta-optimization of the underlying CMA-ES parameters
(e.g., sigma and maximum number of function evaluations). The script trains the model using the best-found
configuration and visualizes the parameter importance.

You can run this example as follows:
    $ python pycma_integration.py
"""
import matplotlib.pyplot as plt
import optuna
from optuna.integration import PyCmaSampler

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Define objective for Optuna + PyCmaSampler
def cma_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    max_features_idx = trial.suggest_int("max_features_idx", 0, 2)

    feature_options = ["sqrt", "log2", None]
    max_features = feature_options[max_features_idx]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )

    score = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring="accuracy").mean()
    return score


# Use PyCmaSampler for Optuna
print("\nStarting CMA-ES optimization using Optuna's PyCmaSampler...")
sampler = PyCmaSampler(seed=42)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(cma_objective, n_trials=50)

# Map best parameters
best_params = study.best_params
feature_options = ["sqrt", "log2", None]
best_params["max_features"] = feature_options[best_params.pop("max_features_idx")]

# Train final model
final_model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    max_features=best_params["max_features"],
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X_train_scaled, y_train)
test_accuracy = accuracy_score(y_test, final_model.predict(X_test_scaled))

# Print final results
print("\nFinal model (Optuna PyCmaSampler):")
print(f"  Best parameters: {best_params}")

# Visualization: Parameter importance
ax = optuna.visualization.matplotlib.plot_param_importances(study)
fig = ax.figure  # Get the figure from the axes
fig.set_size_inches(10, 6)
fig.tight_layout()
fig.savefig("parameter_importance.png", dpi=300)
plt.close(fig)

# Visualization: Contour plot
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_contour(study, params=["n_estimators", "max_depth"])
plt.title("Contour Plot: n_estimators vs max_depth")
plt.tight_layout()
plt.savefig("contour_plot.png", dpi=300)
plt.close()
