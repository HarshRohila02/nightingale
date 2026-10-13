"""The cloud notebooks: each code cell parses, none fetches the test split, and every flag a notebook
passes to a script is one that script accepts. A notebook only runs on Colab or Kaggle, so a
stale flag would otherwise surface hours into the owner's run."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))
SHELL = re.compile(r"^\s*[!%]")


def code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def python_of(cell: str) -> str:
    """The cell without its Jupyter shell lines (``!git …``, ``%cd …``), which are not Python."""
    return "\n".join(
        line[: len(line) - len(line.lstrip())] + "pass" if SHELL.match(line) else line
        for line in cell.splitlines()
    )


def script_flags(script: str) -> set[str]:
    source = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
    return set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_every_code_cell_parses(path):
    for cell in code_cells(path):
        ast.parse(python_of(cell))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_no_notebook_downloads_the_test_split(path):
    """docs/05 §2: the test split stays closed until Phase 4."""
    for cell in code_cells(path):
        code = "\n".join(line.split("#", 1)[0] for line in cell.splitlines())
        assert "test.csv" not in code, f"{path.name} mentions test.csv outside a comment"


@pytest.mark.parametrize(
    "name", ["colab_b1_dl.ipynb", "kaggle_b1_dl.ipynb"], ids=["colab", "kaggle"]
)
def test_the_deep_notebooks_pass_only_flags_their_scripts_accept(name):
    source = "\n".join(code_cells(REPO_ROOT / "notebooks" / name))
    for script in ("train_deep.py", "evaluate_reduced_evidence.py"):
        accepted = script_flags(script)
        # The flags written after each mention of the script, up to the end of its call.
        for call in re.findall(rf"{re.escape(script)}(.*?)(?:\]|\n\n|$)", source, flags=re.S):
            used = set(re.findall(r'"(--[a-z0-9-]+)"', call)) | set(
                re.findall(r"\s(--[a-z0-9-]+)", call)
            )
            assert used <= accepted, f"{name} passes {sorted(used - accepted)} to {script}"


def test_the_kaggle_notebook_keeps_data_out_of_its_saved_output():
    """Kaggle saves /kaggle/working as the version's output: the repository and DDXPlus go to
    /tmp, and only models, metrics and logs go to /kaggle/working."""
    source = "\n".join(code_cells(REPO_ROOT / "notebooks" / "kaggle_b1_dl.ipynb"))
    assert 'REPO = Path("/tmp/nightingale")' in source
    assert 'OUT = Path("/kaggle/working/nightingale_b1_dl")' in source
    assert "SEEDS = (42, 43, 44, 45, 46)" in source
