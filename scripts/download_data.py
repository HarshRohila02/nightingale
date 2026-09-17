"""Download every open dataset Nightingale depends on.

Idempotent: re-running skips datasets already present unless --force is passed.
Writes data/raw/MANIFEST.json recording what was fetched and when, so a result
can be traced back to the exact data that produced it.

    python scripts/download_data.py                # all datasets
    python scripts/download_data.py --only ddxplus
    python scripts/download_data.py --force        # re-download

Nothing downloaded here may be committed (docs/03-data-management.md §5).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_data")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST = RAW_DIR / "MANIFEST.json"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str
    license: str
    role: str
    note: str = ""


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="ddxplus",
        hf_id="aai530-group6/ddxplus",
        license="CC-BY-4.0",
        role="Primary ML training and differential-diagnosis ground truth",
        note="EVIDENCES are coded (e.g. E_54_@_V_161); run scripts/decode_ddxplus.py next.",
    ),
    DatasetSpec(
        key="bodhi_s",
        hf_id="ekacare/BODHI-S",
        license="CC-BY-NC-4.0",
        role="Cardiac medical knowledge graph seed",
        note="NON-COMMERCIAL. Attribution to Eka Care is mandatory.",
    ),
    DatasetSpec(
        key="uci_heart",
        hf_id="MLLab-TS/heart_disease_uci",
        license="open (verify mirror before citing)",
        role="Cardiac-risk sub-model and real-data sanity check",
        note="Binary coronary-disease presence — NOT a differential dataset.",
    ),
)


def _load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("MANIFEST.json is corrupt; starting a fresh one")
    return {"datasets": {}}


def _save_manifest(manifest: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download(spec: DatasetSpec, *, force: bool) -> dict | None:
    """Fetch one dataset to data/raw/<key>. Returns its manifest entry."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("The 'datasets' package is missing. Run: pip install -r requirements.txt")
        raise SystemExit(1) from None

    target = RAW_DIR / spec.key
    if target.exists() and not force:
        logger.info("%-10s already present at %s (use --force to re-download)", spec.key, target)
        return None

    logger.info("%-10s downloading %s ...", spec.key, spec.hf_id)
    try:
        dataset = load_dataset(spec.hf_id)
    except Exception as exc:  # noqa: BLE001 - surface a clear message, not a traceback
        logger.error("%-10s FAILED: %s", spec.key, exc)
        logger.error(
            "            Verify the dataset id at https://huggingface.co/datasets/%s "
            "— mirrors do move.",
            spec.hf_id,
        )
        return None

    target.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target))

    splits = {name: len(split) for name, split in dataset.items()}
    logger.info("%-10s saved %s", spec.key, splits)
    if spec.note:
        logger.info("%-10s NOTE: %s", spec.key, spec.note)

    return {
        "hf_id": spec.hf_id,
        "license": spec.license,
        "role": spec.role,
        "splits": splits,
        "path": str(target.relative_to(REPO_ROOT)),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=[d.key for d in DATASETS], help="Download one dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    selected = [d for d in DATASETS if args.only is None or d.key == args.only]
    manifest = _load_manifest()

    for spec in selected:
        entry = download(spec, force=args.force)
        if entry:
            manifest["datasets"][spec.key] = entry

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(manifest)

    print("\n" + "=" * 72)
    print("LICENCE OBLIGATIONS — these are binding, see docs/03-data-management.md")
    print("=" * 72)
    for spec in selected:
        print(f"  {spec.hf_id:<34} {spec.license}")
    print(
        "\n  BODHI-S is CC-BY-NC-4.0: this project must remain NON-COMMERCIAL\n"
        "  and must attribute Eka Care wherever the knowledge graph is used.\n"
    )
    print("  Nothing under data/ may be committed to git.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
