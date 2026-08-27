"""Execute the tutorial notebooks end-to-end (marker ``notebook``, slow).

Each notebook is run with ``nbconvert`` in a fresh kernel from the repository
root; any raised exception fails the test. Run with::

    uv run python -m pytest -m notebook --no-cov
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted((REPO_ROOT / "tutorials").glob("*.ipynb"))


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[nb.stem for nb in NOTEBOOKS])
def test_tutorial_executes(notebook: Path, tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output",
            str(tmp_path / f"{notebook.stem}.executed.ipynb"),
            str(notebook),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "WANDB_MODE": "offline",
            "PYTHONPATH": str(REPO_ROOT),
            **_passthrough_env(),
        },
    )
    assert result.returncode == 0, result.stderr[-4000:]


def _passthrough_env() -> dict:
    import os

    return {
        k: v for k, v in os.environ.items() if k not in {"WANDB_MODE", "PYTHONPATH"}
    }
