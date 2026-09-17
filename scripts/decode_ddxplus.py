"""Decode the DDXPlus evidence vocabulary and extract the chest-pain subset.

DDXPlus stores patient findings as codes (``E_54_@_V_161``). This script turns
them into readable concepts and writes two interim artifacts:

    data/interim/ddxplus_evidences.json          decoded evidence vocabulary
    data/interim/ddxplus_chestpain_conditions.json   our 13 conditions + evidences

Only the two small release JSON files are needed (~140 kB); the 847 MB of patient
CSVs are NOT required for this step.

    python scripts/decode_ddxplus.py

Caveat on translation quality: DDXPlus was authored in French and the English
fields are machine-translated, sometimes badly. ``déchirante`` (tearing — the
classic aortic-dissection descriptor) is rendered "heartbreaking", and
``lancinante`` (shooting/throbbing) as "haunting". Both the French and English
strings are preserved here so a human can catch these.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import DDXPLUS_LABELS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw" / "ddxplus"
INTERIM = REPO_ROOT / "data" / "interim"

EVIDENCE_CODE_SEP = "_@_"


def load_release_files() -> tuple[dict, dict]:
    """Load release_evidences.json and release_conditions.json."""
    evidences_path = RAW / "release_evidences.json"
    conditions_path = RAW / "release_conditions.json"
    missing = [p for p in (evidences_path, conditions_path) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing DDXPlus release files:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nFetch them (~140 kB, no need for the CSVs):\n"
            '  python -c "from huggingface_hub import hf_hub_download as d; '
            "[d('aai530-group6/ddxplus', f, repo_type='dataset') for f in "
            "['release_evidences.json','release_conditions.json']]\""
        )
    return (
        json.loads(evidences_path.read_text(encoding="utf-8")),
        json.loads(conditions_path.read_text(encoding="utf-8")),
    )


def decode_evidence_vocabulary(evidences: dict) -> dict:
    """Flatten the evidence spec into {code: {label, type, values...}}.

    Multi-choice and categorical evidences expand into one entry per value, keyed
    ``E_54_@_V_161``, which is exactly how they appear in the patient rows.
    """
    vocabulary: dict[str, dict] = {}
    for code, spec in evidences.items():
        question_en = (spec.get("question_en") or "").strip()
        question_fr = (spec.get("question_fr") or "").strip()
        data_type = spec.get("data_type", "B")
        is_antecedent = bool(spec.get("is_antecedent", False))

        vocabulary[code] = {
            "code": code,
            "label_en": question_en,
            "label_fr": question_fr,
            "data_type": data_type,
            "is_antecedent": is_antecedent,
            "kind": "question",
        }

        value_meaning = spec.get("value_meaning") or {}
        for value_code, meaning in value_meaning.items():
            en = (meaning.get("en") or "").strip()
            fr = (meaning.get("fr") or "").strip()
            if en.upper() == "NA":
                continue
            vocabulary[f"{code}{EVIDENCE_CODE_SEP}{value_code}"] = {
                "code": f"{code}{EVIDENCE_CODE_SEP}{value_code}",
                "parent": code,
                "label_en": f"{question_en} {en}".strip(),
                "label_fr": f"{question_fr} {fr}".strip(),
                "value_en": en,
                "value_fr": fr,
                "data_type": data_type,
                "is_antecedent": is_antecedent,
                "kind": "value",
            }
    return vocabulary


def extract_chestpain_conditions(conditions: dict, vocabulary: dict) -> dict:
    """Filter to the 13 in-scope conditions and decode their evidence sets."""
    out: dict[str, dict] = {}
    for name, spec in conditions.items():
        condition = DDXPLUS_LABELS.get(name.strip())
        if condition is None:
            continue  # out of scope for Nightingale

        symptoms = sorted(spec.get("symptoms", {}))
        antecedents = sorted(spec.get("antecedents", {}))
        out[condition.id] = {
            "condition_id": condition.id,
            "label": condition.label,
            "ddxplus_label": name,
            "icd10": spec.get("icd10-id"),
            "severity": spec.get("severity"),
            "is_must_not_miss": condition.is_must_not_miss,
            "symptoms": [
                {"code": c, "label_en": vocabulary.get(c, {}).get("label_en", "")} for c in symptoms
            ],
            "antecedents": [
                {"code": c, "label_en": vocabulary.get(c, {}).get("label_en", "")}
                for c in antecedents
            ],
        }
    return out


def main() -> int:
    evidences, conditions = load_release_files()
    vocabulary = decode_evidence_vocabulary(evidences)
    chestpain = extract_chestpain_conditions(conditions, vocabulary)

    INTERIM.mkdir(parents=True, exist_ok=True)
    (INTERIM / "ddxplus_evidences.json").write_text(
        json.dumps(vocabulary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (INTERIM / "ddxplus_chestpain_conditions.json").write_text(
        json.dumps(chestpain, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    questions = sum(1 for v in vocabulary.values() if v["kind"] == "question")
    values = sum(1 for v in vocabulary.values() if v["kind"] == "value")

    print(
        f"Evidence vocabulary : {len(vocabulary):,} entries "
        f"({questions} questions, {values} expanded values)"
    )
    print(f"Conditions in scope : {len(chestpain)}/13 of the Nightingale set")
    print(f"Total DDXPlus conds : {len(conditions)}")
    print()
    print(f"{'condition':<32} {'icd10':<8} {'sev':>3}  {'sx':>3} {'ante':>4}  must-not-miss")
    print("-" * 78)
    for spec in sorted(chestpain.values(), key=lambda s: s["label"]):
        print(
            f"{spec['label']:<32} {str(spec['icd10'] or '-'):<8} "
            f"{str(spec['severity'] or '-'):>3}  {len(spec['symptoms']):>3} "
            f"{len(spec['antecedents']):>4}  {'YES' if spec['is_must_not_miss'] else ''}"
        )

    missing = set(DDXPLUS_LABELS.values()) - {
        c for c in DDXPLUS_LABELS.values() if c.id in chestpain
    }
    if missing:
        print("\nWARNING — expected but not found in release_conditions.json:")
        for condition in missing:
            print(f"  {condition.label} (looked for {condition.ddxplus_label!r})")

    print(f"\nWrote {INTERIM / 'ddxplus_evidences.json'}")
    print(f"Wrote {INTERIM / 'ddxplus_chestpain_conditions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
