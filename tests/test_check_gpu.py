"""scripts/check_gpu.py must fail politely where torch is absent.

These tests run only where torch is NOT installed (the main .venv and CI), so they can never
touch a GPU. The real check runs in .venv-gpu, only after the owner's yes (docs/11 §2, D-9).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_gpu.py"


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="torch is installed here; the real GPU check needs the owner's yes (D-9)",
)
@pytest.mark.parametrize("args", [[], ["--device", "cuda"]])
def test_without_torch_it_points_to_the_gpu_environment(args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 1
    assert ".venv-gpu" in result.stdout
