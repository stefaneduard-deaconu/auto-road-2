import os

# must be set before matplotlib is imported anywhere
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Some code writes result files into the working directory; keep each test apart."""
    monkeypatch.chdir(tmp_path)
