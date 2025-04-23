"""
Optuna example that optimizes a Random Forest classifier for the Iris dataset with SHAP.

In this example, we optimize the classification accuracy on the Iris flower dataset using
scikit-learn's RandomForestClassifier. We systematically tune the hyperparameters of the model
including the number of estimators, maximum depth, and feature selection strategy.

After finding the optimal hyperparameters, the code demonstrates model explainability by using
SHAP (SHapley Additive exPlanations) to visualize which features most influence the model's
predictions. This provides insights into the model's decision-making process for flower
classification.

The example generates the following visualization output:
1. A feature importance summary across all classes
"""

import matplotlib.pyplot as plt
import optuna
from optuna.integration import ShapleyImportanceEvaluator

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# -----------------------------
# Step 1: Load Iris Dataset
# -----------------------------
print("Loading Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Split into train/test
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# Step 2: Optuna Objective Function
# -----------------------------
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )

    return cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()


# -----------------------------
# Step 3: Run Optuna Study
# -----------------------------
print("Running hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best trial:")
print(f"  Value: {study.best_trial.value:.4f}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# -----------------------------
# Step 4: Train Final Model
# -----------------------------
best_params = study.best_params
final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# Evaluate
y_pred = final_model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)


evaluator = ShapleyImportanceEvaluator()
importances = evaluator.evaluate(study)

# Sort and plot the importances
importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

plt.figure(figsize=(10, 6))
plt.barh(list(importances.keys()), list(importances.values()))
plt.gca().invert_yaxis()
plt.xlabel("Importance (estimated with SHAP)")
plt.title("Feature Importances via Optuna ShapleyImportanceEvaluator")
plt.tight_layout()
plt.savefig("shap_optuna_importance.png", dpi=300)
print("Optuna SHAP-based importance plot saved as 'shap_optuna_importance.png'")
