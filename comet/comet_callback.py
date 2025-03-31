import comet_ml
import optuna
from optuna_integration.comet import CometCallback

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


# Create the experiment first
experiment = comet_ml.start(online=False)

# Then set the name
experiment.set_name("comet-optuna-example")

# Load dataset
random_state = 42
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=random_state
)


# Evaluation function
def evaluate(y_true, y_pred):
    return {
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 20)

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    score = f1_score(y_test, y_pred)

    # Log the metric manually
    experiment.log_metric("f1_score", score, step=trial.number)

    return score


# Optuna Study with Comet ML callback
study = optuna.create_study(direction="maximize")
comet_callback = CometCallback(
    study, project_name="comet-optuna-sklearn-example", metric_names=["f1_score"]
)
study.optimize(objective, n_trials=20, callbacks=[comet_callback])

# Train final model with best parameters
best_params = study.best_params
clf = RandomForestClassifier(**best_params, random_state=random_state)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Log training metrics
with experiment.train():
    experiment.log_metrics(evaluate(y_train, y_train_pred))

# Log testing metrics
with experiment.test():
    experiment.log_metrics(evaluate(y_test, y_test_pred))

experiment.end()
