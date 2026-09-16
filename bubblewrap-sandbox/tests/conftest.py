"""Shared fixtures for sandbox tests."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def outside_dir() -> Iterator[Path]:
    """Throwaway directory under ``$HOME``, outside the repo/launch dir.

    Under ``$HOME`` so the sandbox fence applies (seatbelt denies ``/Users``;
    bubblewrap does not mount ``$HOME``). Cleaned up after the test.
    """
    d = Path.home().resolve() / f".dcode-sandbox-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        pytest.skip(f"cannot create ad hoc dir under $HOME: {exc}")
    yield d
    shutil.rmtree(d, ignore_errors=True)
