from pathlib import Path

from aai_fine_tuning.settings import (
    COURSE_ROOT_VARIABLE,
    DEFAULT_COURSE_ROOT,
    PROJECT,
    course_root,
    ensure_local_paths,
    local_paths,
)


def test_default_state_root_is_ignored_and_project_local(monkeypatch):
    monkeypatch.delenv(COURSE_ROOT_VARIABLE, raising=False)
    assert course_root() == DEFAULT_COURSE_ROOT
    assert DEFAULT_COURSE_ROOT == PROJECT / ".aai" / "course-v1"


def test_environment_variable_overrides_the_state_root(monkeypatch, tmp_path):
    monkeypatch.setenv(COURSE_ROOT_VARIABLE, str(tmp_path / "elsewhere"))
    assert course_root() == (tmp_path / "elsewhere").resolve()


def test_local_paths_stay_under_one_disposable_root(tmp_path):
    paths = local_paths(tmp_path)
    for directory in (paths.mlflow_dir, paths.mlflow_artifacts, paths.hf_home):
        assert directory.is_relative_to(tmp_path)
    assert paths.mlflow_uri == f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}"


def test_ensure_local_paths_creates_the_directories(tmp_path):
    root = tmp_path / "state"
    paths = ensure_local_paths(root)
    assert paths.root.is_dir()
    assert paths.mlflow_artifacts.is_dir()
    assert isinstance(paths.root, Path)
