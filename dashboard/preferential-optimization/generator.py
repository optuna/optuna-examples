from __future__ import annotations

import os
import tempfile
import time
from typing import NoReturn

from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna_dashboard import register_preference_feedback_component
from optuna_dashboard.preferential import create_study
from optuna_dashboard.preferential.samplers.gp import PreferentialGPSampler
from PIL import Image


STORAGE_URL = "sqlite:///example.db"
artifact_path = os.path.join(os.path.dirname(__file__), "artifact")
artifact_store = FileSystemArtifactStore(base_path=artifact_path)
os.makedirs(artifact_path, exist_ok=True)


def main() -> NoReturn:
    study = create_study(
        n_generate=4,
        study_name="Preferential Optimization",
        storage=STORAGE_URL,
        sampler=PreferentialGPSampler(),
        load_if_exists=True,
    )
    # Change the component, displayed on the human feedback pages.
    # By default (component_type="note"), the Trial's Markdown note is displayed.
    user_attr_key = "rgb_image"
    register_preference_feedback_component(study, "artifact", user_attr_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        while True:
            # If study.should_generate() returns False,
            # the generator waits for human evaluation.
            if not study.should_generate():
                time.sleep(0.1)  # Avoid busy-loop
                continue

            trial = study.ask()
            # 1. Ask new parameters
            r = trial.suggest_int("r", 0, 255)
            g = trial.suggest_int("g", 0, 255)
            b = trial.suggest_int("b", 0, 255)

            # 2. Generate image
            image_path = os.path.join(tmpdir, f"sample-{trial.number}.png")
            image = Image.new("RGB", (320, 240), color=(r, g, b))
            image.save(image_path)

            # 3. Upload Artifact and set artifact_id to trial.user_attrs["rgb_image"].
            artifact_id = upload_artifact(trial, image_path, artifact_store)
            trial.set_user_attr(user_attr_key, artifact_id)


if __name__ == "__main__":
    main()
