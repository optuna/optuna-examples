"""
A simple example of Optuna Artifact Store.

In this example, we optimize coefficients (a, b) of a quadratic function:
    f(x) = a * x**2 + b

The demo works as follows:
1. Create a dataset by ``create_dataset``,
2. For each trial, Optuna suggests a candidate of (a, b),
3. For each trial, plot the prediction,
4. Upload the prediction figure to the artifact store, and
5. After the optimization, check the prediction for best_trial using the download API.

"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.artifacts import download_artifact
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import get_all_artifact_meta
from optuna.artifacts import upload_artifact
import pandas as pd


dataset_path = "demo-dataset.csv"
fig_name = "result-trial.png"
# NOTE: The uploaded artifacts can be viewed in Optuna Dashboard with the following command:
# $ optuna-dashboard sqlite:///simple-artifact-store-demo.db --artifact-dir ./save-artifact-here
base_path = "./save-artifact-here"
# Make the directory used for artifact store.
os.makedirs(base_path, exist_ok=True)
# Instantiate an artifact store.
artifact_store = FileSystemArtifactStore(base_path=base_path)
# Instantiate an RDB.
storage = optuna.storages.RDBStorage("sqlite:///simple-artifact-store-demo.db")


def create_dataset(dataset_path):
    # The coefficients we would like to find by Optuna.
    a_true, b_true = 2, -3
    X = np.random.random(20) * 10 - 5
    Y = a_true * X**2 + b_true
    dataset = pd.DataFrame({"X": X, "Y": Y})
    dataset.to_csv(dataset_path)
    return dataset


dataset = create_dataset(dataset_path)


def plot_predictions(a, b, trial, tmp_dir):
    # Create an artifact, which is figure in this example, to upload.
    _, ax = plt.subplots()
    x = np.linspace(-5, 5, 100)
    ax.scatter(dataset["X"], dataset["Y"], label="Dataset", color="blue")
    ax.plot(x, a * x**2 + b, label="Prediction", color="darkred")
    ax.set_title(f"a={a:.2f}, b={b:.2f}")
    ax.grid()
    ax.legend()
    plt.savefig(os.path.join(tmp_dir, fig_name))
    plt.close()


def objective(trial):
    a = trial.suggest_float("a", -5, 5)
    b = trial.suggest_float("b", -5, 5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_predictions(a, b, trial, tmp_dir)
        fig_path = os.path.join(tmp_dir, fig_name)
        # Link the plotted figure with trial using artifact store API.
        upload_artifact(artifact_store=artifact_store, file_path=fig_path, study_or_trial=trial)

    return np.mean((a * dataset["X"] ** 2 + b - dataset["Y"]) ** 2)


def show_best_result(study, artifact_store):
    best_trial = study.best_trial
    # Get all the artifact information linked to best_trial. (Here we have only one.)
    artifact_meta = get_all_artifact_meta(best_trial, storage=storage)
    fig_path = "./result-best-trial.png"
    # Download the figure from the artifact store to fig_path.
    download_artifact(
        artifact_store=artifact_store,
        artifact_id=artifact_meta[0].artifact_id,
        file_path=fig_path,
    )
    # Display the figure for the best result.
    best_result_img = plt.imread(fig_path)
    plt.figure()
    plt.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
    plt.imshow(best_result_img)
    plt.show()


if __name__ == "__main__":
    # Create a study with a SQLite storage.
    study = optuna.create_study(study_name="demo", storage=storage, load_if_exists=True)
    # Upload the dataset to use by artifact store API.
    upload_artifact(artifact_store=artifact_store, file_path=dataset_path, study_or_trial=study)
    study.optimize(objective, n_trials=30)
    show_best_result(study, artifact_store)
