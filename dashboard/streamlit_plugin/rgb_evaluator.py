from __future__ import annotations

import os
import shutil
import tempfile
import uuid

import optuna
from optuna.trial import TrialState
from optuna_dashboard.artifact.file_system import FileSystemBackend
from optuna_dashboard.streamlit import render_objective_form_widgets
from optuna_dashboard.streamlit import render_trial_note
import streamlit as st


artifact_path = os.path.join(os.path.dirname(__file__), "artifact")
artifact_backend = FileSystemBackend(base_path=artifact_path)


def get_tmp_dir() -> str:
    if "tmp_dir" not in st.session_state:
        tmp_dir_name = str(uuid.uuid4())
        tmp_dir_path = os.path.join(tempfile.gettempdir(), tmp_dir_name)
        os.makedirs(tmp_dir_path, exist_ok=True)
        st.session_state.tmp_dir = tmp_dir_path

    return st.session_state.tmp_dir


def start_streamlit() -> None:
    tmpdir = get_tmp_dir()
    study = optuna.load_study(
        storage="sqlite:///streamlit-db.sqlite3", study_name="Human-in-the-loop Optimization"
    )
    selected_trial = st.sidebar.selectbox("Trial", study.trials, format_func=lambda t: t.number)

    if selected_trial is None:
        return
    render_trial_note(study, selected_trial)
    artifact_id = selected_trial.user_attrs.get("artifact_id")
    if artifact_id:
        with artifact_backend.open(artifact_id) as fsrc:
            tmp_img_path = os.path.join(tmpdir, artifact_id + ".png")
            with open(tmp_img_path, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
        st.image(tmp_img_path, caption="Image")

    if selected_trial.state == TrialState.RUNNING:
        render_objective_form_widgets(study, selected_trial)


if __name__ == "__main__":
    start_streamlit()
