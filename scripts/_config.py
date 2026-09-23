"""Reading configs/config.yaml, for the scripts.

Nothing in ``src/`` reads YAML: every component takes its settings as arguments, so it can be
built three ways — by a script, by a test, by the app — without a file existing. The scripts are
where a configuration file turns into arguments, and this is the shared part of that.

Paths in the file are relative to the repository, so they are resolved here rather than against
whatever directory the script was run from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

__all__ = ["DEFAULT_CONFIG", "REPO_ROOT", "load_config", "open_ranker_from_config"]


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """The configuration, or an empty one if the file is missing.

    A missing file is not an error: every caller has defaults, and a prototype has to run on a
    machine that has not been set up yet.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def open_ranker_from_config(config: dict[str, Any]) -> Any:
    """The ranker the ``ml:`` block asks for (src/ml/ranker.py).

    Returns a degraded ranker, never an exception, when the model or the release files are
    absent — which is the ordinary case on a fresh clone, since both are gitignored.
    """
    from src.ml.ranker import DEFAULT_BACKEND, DEFAULT_MODEL_DIR, open_ranker

    ml = config.get("ml") or {}
    paths = config.get("paths") or {}
    raw = REPO_ROOT / paths.get("raw", "data/raw")
    interim = REPO_ROOT / paths.get("interim", "data/interim")
    return open_ranker(
        ml.get("backend", DEFAULT_BACKEND),
        model_dir=REPO_ROOT / ml.get("model_dir", DEFAULT_MODEL_DIR),
        evidences_path=raw / "ddxplus" / "release_evidences.json",
        conditions_path=interim / "ddxplus_chestpain_conditions.json",
        stem=ml.get("stem"),
        include_narrower=bool(ml.get("include_narrower", True)),
    )
