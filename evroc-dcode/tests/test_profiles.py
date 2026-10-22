"""Tests for dynamic model profiles in evroc_dcode.data._profiles."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_PROFILES_FILE = (
    Path(__file__).resolve().parents[1] / "evroc_dcode" / "data" / "_profiles.py"
)


def _load_profiles_module() -> Any:
    """Execute _profiles.py the way dcode's discovery loader does."""
    spec = importlib.util.spec_from_file_location(
        "evroc_dcode.data._profiles", _PROFILES_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _purge_evroc_modules() -> None:
    """Drop cached evroc_dcode modules so path constants re-read the env."""
    for name in list(sys.modules):
        if name == "evroc_dcode" or name.startswith("evroc_dcode."):
            del sys.modules[name]


def test_profiles_load_models_json_from_data_dir(tmp_path, monkeypatch):
    """The profiles load from EVROC_DATA_DIR regardless of the process cwd."""
    (tmp_path / "models.json").write_text(
        json.dumps(["org/model-a", "org/model-b"]), encoding="utf-8"
    )
    monkeypatch.setenv("EVROC_DATA_DIR", str(tmp_path))
    # Discovery runs in the dcode server process, whose cwd is a scaffold
    # directory with no .evroc beside it.
    monkeypatch.chdir(tmp_path)

    _purge_evroc_modules()
    module = _load_profiles_module()

    assert module._PROFILES == {
        "org/model-a": {
            "name": "org/model-a",
            "tool_calling": True,
            "text_inputs": True,
            "text_outputs": True,
        },
        "org/model-b": {
            "name": "org/model-b",
            "tool_calling": True,
            "text_inputs": True,
            "text_outputs": True,
        },
    }


def test_profiles_empty_without_model_file(tmp_path, monkeypatch):
    """No model file means no profiles, not an error."""
    monkeypatch.setenv("EVROC_DATA_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    _purge_evroc_modules()
    module = _load_profiles_module()

    assert module._PROFILES == {}
