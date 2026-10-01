"""scripts/_config.py: where configs/config.yaml turns into arguments."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from src.pipeline import DiagnosisPipeline
from src.stubs import ConstantRanker, InMemoryGraphStore

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _config import fusion_weights_from_config, load_config  # noqa: E402


def write(tmp_path: Path, data: dict) -> dict:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def test_the_fusion_weights_come_from_the_ranking_block(tmp_path):
    """EXP-006's α must reach the pipeline: before 2c no code read these keys."""
    weights = fusion_weights_from_config(
        write(tmp_path, {"ranking": {"ml_weight": 0.1, "kg_weight": 0.9}})
    )
    assert weights == {"ml_weight": 0.1, "kg_weight": 0.9}
    pipeline = DiagnosisPipeline(ranker=ConstantRanker(), graph=InMemoryGraphStore(), **weights)
    assert (pipeline.ml_weight, pipeline.kg_weight) == (0.1, 0.9)


def test_without_a_ranking_block_the_defaults_apply(tmp_path):
    assert fusion_weights_from_config(write(tmp_path, {})) == {"ml_weight": 0.5, "kg_weight": 0.5}


@pytest.mark.parametrize("pair", [(0.0, 0.0), (-0.1, 1.0)])
def test_a_pair_the_fusion_cannot_use_fails_loudly(tmp_path, pair):
    config = write(tmp_path, {"ranking": {"ml_weight": pair[0], "kg_weight": pair[1]}})
    with pytest.raises(ValueError):
        fusion_weights_from_config(config)


def test_the_committed_config_is_readable():
    weights = fusion_weights_from_config(load_config())
    assert weights["ml_weight"] + weights["kg_weight"] > 0
