from pathlib import Path

import pytest


@pytest.fixture
def suite_dir() -> Path:
    return Path(__file__).resolve().parents[1]
