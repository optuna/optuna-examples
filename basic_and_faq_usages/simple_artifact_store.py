import os

import matplotlib.pyplot as plt

import numpy as np

import optuna
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import download_artifact
from optuna.artifacts import get_all_artifact_meta
from optuna.artifacts import upload_artifact

import pandas as pd


dataset_path = "demo-dataset.csv"


def create_dataset(dataset_path):
    # The coefficients we would like to find by Optuna.
    a_true, b_true = 2, -3
    X = np.random.random(20)
    Y = a_true * X + b_true
    dataset = pd.DataFrame({"X": X, "Y": Y})
    dataset.to_csv(dataset_path)
    return dataset


dataset = create_dataset(dataset_path)


def plot_predictions(a, b, trial):
    # Create an artifact, which is figure in this example, to upload.
    os.makedirs("figs/", exist_ok=True)
    _, ax = plt.subplots()
    fig_path = f"figs/result-trial{trial.number}.png"
    x = np.linspace(0, 1, 100)
    ax.scatter(dataset["X"], dataset["Y"], label="Dataset", color="blue")
    ax.plot(x, a * x + b, label="Prediction", color="darkred")
    ax.grid()
    ax.legend()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def objective(trial, artifact_store):
    a = trial.suggest_float("a", -5, 5)
    b = trial.suggest_float("b", -5, 5)
    fig_path = plot_predictions(a, b, trial)

    # Link the plotted figure with trial using artifact store API.
    upload_artifact(artifact_store=artifact_store, file_path=fig_path, study_or_trial=trial)

    return np.mean((a * dataset["X"] + b - dataset["Y"]) ** 2)


def show_best_result(study, artifact_store):
    best_trial = study.best_trial
    # Get all the artifact information linked to best_trial. (Here we have only one.)
    artifact_meta = get_all_artifact_meta(best_trial, storage=study._storage)
    fig_path = "figs/result-best-trial.png"
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
    # NOTE: The uploaded artifacts can be viewed in Optuna Dashboard with the following command:
    # $ optuna-dashboard sqlite:///artifact-demo.db --artifact-dir ./save-artifact-here
    # Create a study with a SQLite storage.
    study = optuna.create_study(
        study_name="demo", storage="sqlite:///simple-artifact-store-demo.db", load_if_exists=True
    )
    base_path = os.path.join("./save-artifact-here")
    # Make the directory used for artifact store.
    os.makedirs(base_path, exist_ok=True)
    # Instantiate an artifact store.
    artifact_store = FileSystemArtifactStore(base_path=base_path)
    # Upload the dataset to use by artifact store API.
    upload_artifact(artifact_store=artifact_store, file_path=dataset_path, study_or_trial=study)
    study.optimize(lambda trial: objective(trial, artifact_store), n_trials=30)
    show_best_result(study, artifact_store)
