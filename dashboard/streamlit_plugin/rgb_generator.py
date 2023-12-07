from __future__ import annotations

import os
import tempfile
import time
from typing import NoReturn

import optuna
from optuna.trial import TrialState
from optuna_dashboard import ChoiceWidget
from optuna_dashboard import register_objective_form_widgets
from optuna_dashboard import save_note
from optuna_dashboard.artifact import upload_artifact
from optuna_dashboard.artifact.file_system import FileSystemBackend
from PIL import Image


def suggest_and_generate_image(
    study: optuna.Study, artifact_backend: FileSystemBackend, tmpdir: str
) -> None:
    # 1. Ask new parameters
    trial = study.ask()
    r = trial.suggest_int("r", 0, 255)
    g = trial.suggest_int("g", 0, 255)
    b = trial.suggest_int("b", 0, 255)

    # 2. Generate image
    image_path = os.path.join(tmpdir, f"sample-{trial.number}.png")
    image = Image.new("RGB", (320, 240), color=(r, g, b))
    image.save(image_path)

    # 3. Upload Artifact
    artifact_id = upload_artifact(artifact_backend, trial, image_path)
    trial.set_user_attr("artifact_id", artifact_id)

    # 4. Save Note
    save_note(trial, f"## Trial {trial.number}")


def main() -> NoReturn:
    # 1. Create Artifact Store
    artifact_path = os.path.join(os.path.dirname(__file__), "artifact")
    artifact_backend = FileSystemBackend(base_path=artifact_path)

    if not os.path.exists(artifact_path):
        os.mkdir(artifact_path)

    # 2. Create Study
    study = optuna.create_study(
        study_name="Human-in-the-loop Optimization",
        storage="sqlite:///streamlit-db.sqlite3",
        sampler=optuna.samplers.TPESampler(constant_liar=True, n_startup_trials=5),
        load_if_exists=True,
    )
    study.set_metric_names(["Looks like sunset color?"])

    # 4. Register ChoiceWidget
    register_objective_form_widgets(
        study,
        widgets=[
            ChoiceWidget(
                choices=["Good 👍", "So-so👌", "Bad 👎"],
                values=[-1, 0, 1],
                description="Please input your score!",
            ),
        ],
    )

    # 5. Start Human-in-the-loop Optimization
    n_batch = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        while True:
            running_trials = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
            if len(running_trials) >= n_batch:
                time.sleep(1)  # Avoid busy-loop
                continue
            suggest_and_generate_image(study, artifact_backend, tmpdir)


if __name__ == "__main__":
    main()
