"""The cloud notebooks: each code cell parses, none fetches the test split, and every flag a notebook
passes to a script is one that script accepts. A notebook only runs on Colab or Kaggle, so a
stale flag would otherwise surface hours into the owner's run."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

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
    "name",
    ["colab_b1_dl.ipynb", "kaggle_b1_dl.ipynb", "kaggle_exp021_reports.ipynb"],
    ids=["colab", "kaggle", "kaggle-reports"],
)
def test_the_deep_notebooks_pass_only_flags_their_scripts_accept(name):
    source = "\n".join(code_cells(REPO_ROOT / "notebooks" / name))
    for script in ("train_deep.py", "evaluate_reduced_evidence.py", "report_deep.py"):
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


def test_the_retrieval_notebook_passes_only_flags_its_script_accepts():
    """Task 3a (docs/11 §4.6): the corpus notebook runs scripts/build_retrieval_corpus.py."""
    source = "\n".join(code_cells(REPO_ROOT / "notebooks" / "kaggle_retrieval_corpus.ipynb"))
    accepted = script_flags("build_retrieval_corpus.py")
    calls = re.findall(r"build_retrieval_corpus\.py(.*?)(?:\]|\n\n|$)", source, flags=re.S)
    assert len(calls) == 2  # the smoke test and the real build
    for call in calls:
        used = set(re.findall(r'"(--[a-z0-9-]+)"', call))
        assert used and used <= accepted, f"passes {sorted(used - accepted)}"
    assert {"--smoke", "--per-condition", "--out", "--cache", "--email"} <= accepted


def test_the_retrieval_notebook_never_touches_ddxplus_and_saves_only_the_bundle():
    """It needs no patient data at all; the code goes to /tmp, the bundle to /kaggle/working."""
    source = "\n".join(code_cells(REPO_ROOT / "notebooks" / "kaggle_retrieval_corpus.ipynb"))
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    for forbidden in (
        "ddxplus",
        "hf_hub_download",
        "aai530",
        "train.csv",
        "validate.csv",
        "api_key",
    ):
        assert forbidden not in code.lower(), forbidden
    assert 'REPO = Path("/tmp/nightingale")' in source
    assert 'OUT = Path("/kaggle/working/retrieval_corpus")' in source
    assert "make_archive" in source and "STATUS.txt" in source


def test_the_synthea_notebook_passes_only_flags_its_script_accepts_and_pins_its_release():
    """Task 1d (docs/17 T7): Synthea at the release its SNOMED table was checked against, seed 42,
    the repository in /tmp and only the output in /kaggle/working; and no patient data of ours."""
    from src.patient_kg.mapping import SYNTHEA_VERSION

    source = "\n".join(code_cells(REPO_ROOT / "notebooks" / "kaggle_synthea.ipynb"))
    accepted = script_flags("build_patient_kg.py")
    calls = re.findall(r"build_patient_kg\.py(.*?)(?:\]|cwd=)", source, flags=re.S)
    assert len(calls) == 2  # the smoke test and the real build
    for call in calls:
        used = set(re.findall(r'"(--[a-z0-9-]+)"', call))
        assert used and used <= accepted, f"passes {sorted(used - accepted)}"
    assert f'SYNTHEA_VERSION = "{SYNTHEA_VERSION}"' in source
    assert "SEED = 42" in source and '"-s", SEED' in source
    assert 'REPO = Path("/tmp/nightingale")' in source
    assert 'OUT = Path("/kaggle/working/synthea_patient_kg")' in source
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    for forbidden in ("ddxplus", "train.csv", "validate.csv", "hf_hub_download"):
        assert forbidden not in code.lower(), forbidden


# --------------------------------------------------------------------------------------------- #
# The language-model trial (EXP-009, EXP-010; docs/11 §4.8)
# --------------------------------------------------------------------------------------------- #

LLM_NOTEBOOK = REPO_ROOT / "notebooks" / "kaggle_llm_explain.ipynb"
LLM_SCRIPTS = (
    "run_llm_baselines.py",
    "score_llm_baselines.py",
    "build_ddxplus_chestpain.py",
    "decode_ddxplus.py",
)


def llm_code() -> str:
    return "\n".join(code_cells(LLM_NOTEBOOK))


def test_the_llm_notebook_passes_only_flags_its_scripts_accept():
    code = llm_code()
    accepted = set().union(*(script_flags(s) for s in LLM_SCRIPTS)) | {"--oneline"}  # git log's
    used = set(re.findall(r'"(--[a-z0-9-]+)"', code))
    assert used and used <= accepted, f"unknown flags: {sorted(used - accepted)}"
    # and the calls of the trial's own scripts pass the ones they are meant to
    run_call = re.search(r"run_llm_baselines\.py(.*?)\]", code, flags=re.S).group(1)
    assert set(re.findall(r'"(--[a-z0-9-]+)"', run_call)) == {"--model", "--arms"}
    assert {"--smoke", "--out", "--bundle"} <= script_flags("run_llm_baselines.py")
    assert {"--sheet"} <= script_flags("score_llm_baselines.py")


def test_the_llm_notebook_runs_the_model_on_kaggle_with_the_registered_settings():
    from src.llm.baselines import ARMS

    code = llm_code()
    config = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    assert f'MODEL = "{config["llm"]["model"]}"' in code  # D-5
    assert "ARMS = " + repr(tuple(ARMS)).replace("'", '"') in code
    notebook = json.loads(LLM_NOTEBOOK.read_text(encoding="utf-8"))
    kaggle = notebook["metadata"]["kaggle"]
    assert kaggle["isGpuEnabled"] and kaggle["isInternetEnabled"]
    # Ollama and its model are installed and pulled there, never on the laptop (D-7, D-9).
    assert "ollama.com/install.sh" in code and '"ollama", "pull", MODEL' in code
    assert "nvidia-smi" in code  # no GPU, no run: an 8B model on the CPU would take days


def test_the_llm_notebook_keeps_data_and_the_model_out_of_its_saved_output():
    code = llm_code()
    assert 'REPO = Path("/tmp/nightingale")' in code
    assert 'OUT = Path("/kaggle/working/llm_explain")' in code
    assert '"OLLAMA_MODELS": "/tmp/ollama_models"' in code  # 5 GB must not be saved
    assert "make_archive" in code and "STATUS.txt" in code and "NOT DONE" in code
    # only DDXPlus's validation split, at the snapshot the other notebooks use
    assert '"validate.csv"' in code and "2ad986acc1ec62fb4a94171acc43f4fdd5bfde53" in code
    assert '"train.csv"' not in code


def test_the_llm_notebook_holds_no_secret():
    code = llm_code().lower()
    for forbidden in ("api_key", "password", "token", "neo4j_", "secret"):
        assert forbidden not in code, forbidden
    assert not re.search(r"hf_[a-z0-9]{20,}", code)  # a Hugging Face token (hf_hub_download is not)
    assert not re.search(r"\.env\b", code)  # never reads a .env file (os.environ is fine)


def test_the_helpers_the_llm_notebook_imports_exist():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import report_deep
    import run_llm_baselines

    for name in ("find_bundle", "place_bodhi_s", "previous_results"):
        assert callable(getattr(run_llm_baselines, name))
    assert callable(report_deep.place_models)
    assert (
        "from run_llm_baselines import find_bundle, place_bodhi_s, previous_results" in llm_code()
    )
