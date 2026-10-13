"""EXP-021's full report (scripts/report_deep.py): which systems it compares, and how the Kaggle
notebook finds the models however Kaggle stores an uploaded dataset."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import report_deep  # noqa: E402


def fake_bundles(root: Path) -> None:
    """The three bundles' layout, with one tiny file each where the script looks."""
    for path in (
        "nightingale_b1_asked/b1_asked_aug/b1_logreg.json",
        "nightingale_b1_seeds/b1_aug/seed43/b1_logreg.json",
        "nightingale_b1_dl/b1_dl_asked_aug/seed42/b1_dl.json",
    ):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("{}", encoding="utf-8")


def test_the_bundles_are_found_inside_an_uploaded_zip(tmp_path):
    staged = tmp_path / "staged"
    fake_bundles(staged)
    source = tmp_path / "input" / "nightingale-models"
    source.mkdir(parents=True)
    with zipfile.ZipFile(source / "nightingale_models.zip", "w") as z:
        for path in staged.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(staged).as_posix())
    models = tmp_path / "models"
    assert sorted(report_deep.place_models(tmp_path / "input", models)) == sorted(
        report_deep.BUNDLES
    )
    assert (models / "nightingale_b1_dl" / "b1_dl_asked_aug" / "seed42" / "b1_dl.json").exists()


def test_the_bundles_are_found_where_kaggle_unpacked_them(tmp_path):
    fake_bundles(tmp_path / "input" / "nightingale-models" / "nightingale_models")
    models = tmp_path / "models"
    assert sorted(report_deep.place_models(tmp_path / "input", models)) == sorted(
        report_deep.BUNDLES
    )
    assert (models / "nightingale_b1_seeds" / "b1_aug" / "seed43" / "b1_logreg.json").exists()


def test_each_transformer_is_compared_with_the_models_of_its_own_prime():
    for a, b in report_deep.PAIRS:
        assert a.startswith("B1-DL") and ("′" in a) == ("′" in b)


def test_seed_42_comparators_are_the_configured_bundles_and_the_others_come_from_the_seeds():
    models = Path("models")
    at_42 = report_deep.systems(42, models)
    at_43 = report_deep.systems(43, models)
    assert at_42["B1-LR′+aug"] == (models / "nightingale_b1_asked" / "b1_asked_aug", "logreg")
    assert at_43["B1-XGB+aug"] == (models / "nightingale_b1_seeds" / "b1_aug" / "seed43", "xgboost")
    assert at_43["B1-DL′+aug"][0] == models / "nightingale_b1_dl" / "b1_dl_asked_aug" / "seed43"
