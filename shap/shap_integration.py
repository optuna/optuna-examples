"""
Optuna example that optimizes a Random Forest classifier for the Iris dataset with SHAP.

In this example, we optimize the classification accuracy on the Iris flower dataset using
scikit-learn's RandomForestClassifier. We systematically tune the hyperparameters of the model
including the number of estimators, maximum depth, and feature selection strategy.

After finding the optimal hyperparameters, the code demonstrates model explainability by using
SHAP (SHapley Additive exPlanations) to visualize which features most influence the model's
predictions. This provides insights into the model's decision-making process for flower
classification.

The example generates the following visualization outputs:
1. A SHAP bar plot showing feature contributions for a single prediction
2. A feature importance summary across all classes
3. Plot SHAP explanation for the predicted class
"""

import matplotlib.pyplot as plt
import optuna

import shap
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
print("Training final model with best parameters...")
best_params = study.best_params
final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# Evaluate
y_pred = final_model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# -----------------------------
# Step 5: Explain with SHAP
# -----------------------------
print("Explaining predictions with SHAP...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_valid)  # list of arrays, one per class

# Get the predicted class for sample 0
sample_idx = 0
predicted_class = final_model.predict([X_valid[sample_idx]])[0]
actual_class = y_valid[sample_idx]
print(f"Explaining sample {sample_idx}:")
print(f"  Actual class: {actual_class} ({iris.target_names[actual_class]})")
print(f"  Predicted class: {predicted_class} ({iris.target_names[predicted_class]})")

# Create and plot SHAP explanation for the predicted class
plt.figure(figsize=(10, 6))
explanation = shap.Explanation(
    values=shap_values[predicted_class][sample_idx],
    base_values=explainer.expected_value[predicted_class],
    data=X_valid[sample_idx],
    feature_names=iris.feature_names,
)


predicted_class = final_model.predict([X_valid[0]])[0]

# Use the corresponding SHAP values
explanation = shap.Explanation(
    values=shap_values[predicted_class][0],
    base_values=explainer.expected_value[predicted_class],
    data=X_valid[0],
    feature_names=[f"pixel_{i}" for i in range(X_valid.shape[1])],
)

shap.plots.bar(explanation, max_display=10)
plt.tight_layout()
plt.savefig("shap_plot_predicted_class.png", dpi=300)
plt.close()

# Save bar plot
shap.plots.bar(explanation)
plt.title(f"SHAP values for class: {iris.target_names[predicted_class]}")
plt.tight_layout()
plt.savefig("shap_plot_iris.png", dpi=300)
print("SHAP plot saved as 'shap_plot_iris.png'")

# Feature importance plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_valid, feature_names=iris.feature_names, plot_type="bar")
plt.tight_layout()
plt.savefig("shap_feature_importance_iris.png", dpi=300)
print("Feature importance plot saved as 'shap_feature_importance_iris.png'")
