"""The patient knowledge graph from Synthea (task 1d, docs/17 T7): src/patient_kg/ and
scripts/build_patient_kg.py, on a hand-made fixture of five invented patients
(tests/fixtures/synthea_mini/, its README says what each tests).

Nothing here runs Synthea: the notebook does that on Kaggle (notebooks/kaggle_synthea.ipynb).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from src.contracts import Assertion, PatientCase
from src.medical_kg.crosswalk import BY_CONCEPT
from src.patient_kg.graph import (
    PATIENT,
    build_patient_graph,
    patient_findings,
    sex_of,
    unmapped_codes,
)
from src.patient_kg.mapping import (
    CONCEPTS_WITHOUT_CODES,
    CONDITION_CODES,
    NOT_MAPPED,
    SMOKING_STATUS,
)
from src.patient_kg.synthea import parse_date, read_synthea

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthea_mini"
SCRIPT = REPO_ROOT / "scripts" / "build_patient_kg.py"
AS_OF = date(2026, 1, 1)


@pytest.fixture(scope="module")
def records():
    return read_synthea(FIXTURE)


def findings_of(records, patient_id: str) -> dict[str, object]:
    graph = build_patient_graph(records[patient_id], AS_OF)
    return {f.concept_id: f for f in patient_findings(graph)}


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

MAPPINGS = [*CONDITION_CODES.values(), *SMOKING_STATUS.values()]


def test_every_concept_the_table_names_is_a_crosswalk_risk_factor():
    """No new RF:* concept: each one the table maps, or explains, is one the crosswalk holds."""
    named = {m.concept_id for m in MAPPINGS} | set(CONCEPTS_WITHOUT_CODES)
    assert all(c.startswith("RF:") for c in named)
    assert named <= set(BY_CONCEPT), sorted(named - set(BY_CONCEPT))


def test_every_crosswalk_risk_factor_is_mapped_or_explained():
    risk_factors = {c for c in BY_CONCEPT if c.startswith("RF:")}
    mapped = {m.concept_id for m in MAPPINGS}
    assert risk_factors == mapped | set(CONCEPTS_WITHOUT_CODES)
    assert not mapped & set(CONCEPTS_WITHOUT_CODES)


def test_every_mapped_code_names_the_synthea_module_it_was_checked_in():
    for m in MAPPINGS:
        assert m.code.isdigit() and m.module.endswith(".json"), m
        assert m.verify is False, f"{m.code} is marked unchecked; say so in the PR"
    assert not set(CONDITION_CODES) & set(NOT_MAPPED)


def test_only_smoking_status_can_deny():
    denials = [m for m in MAPPINGS if m.assertion is Assertion.ABSENT]
    assert {m.concept_id for m in denials} == {"RF:smoking"}
    assert {m.display for m in denials} == {"Ex-smoker (finding)", "Never smoked tobacco (finding)"}


# --------------------------------------------------------------------------- #
# Reading Synthea's CSVs
# --------------------------------------------------------------------------- #


def test_the_reader_keeps_every_patient_and_their_dated_entries(records):
    assert sorted(records) == ["p-0001", "p-0002", "p-0003", "p-0004", "p-0005"]
    one = records["p-0001"]
    assert (one.birth_date, one.gender) == (date(1960, 3, 15), "M")
    kinds = sorted({e.kind for e in one.entries})
    assert kinds == ["condition", "medication", "observation", "procedure"]
    prediabetes = next(e for e in one.entries if e.code == "714628002")
    assert (prediabetes.start, prediabetes.stop) == (date(2012, 2, 2), date(2015, 3, 3))
    assert records["p-0003"].death_date == date(2024, 5, 1)


def test_dates_ages_and_activity():
    assert parse_date("2024-01-01T09:00:00Z") == date(2024, 1, 1)
    assert parse_date("") is None
    records = read_synthea(FIXTURE)
    one = records["p-0001"]
    assert one.age_on(date(2026, 3, 14)) == 65 and one.age_on(date(2026, 3, 15)) == 66
    pregnancy = [e for e in records["p-0002"].entries if e.code == "72892002"]
    assert [e.active_on(AS_OF) for e in pregnancy] == [False, True]
    assert not records["p-0003"].alive_on(AS_OF)


def test_a_folder_without_patients_is_refused_and_missing_files_are_allowed(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_synthea(tmp_path)
    (tmp_path / "patients.csv").write_text(
        (FIXTURE / "patients.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert all(not r.entries for r in read_synthea(tmp_path).values())


# --------------------------------------------------------------------------- #
# One patient's graph, and its findings
# --------------------------------------------------------------------------- #


def test_active_conditions_map_and_ended_or_future_ones_do_not(records):
    found = findings_of(records, "p-0001")
    assert found["RF:hypertension"].assertion is Assertion.PRESENT
    assert found["RF:diabetes"].assertion is Assertion.PRESENT
    assert found["RF:diabetes"].qualifiers == {"snomed": "44054006", "since": "2015-03-03"}
    assert "RF:heart_failure" not in found, "it begins after the day"


def test_the_latest_smoking_status_on_or_before_the_day_decides(records):
    """p-0001 smoked daily in 2019 and was an ex-smoker in 2023; a 2027 observation is after the
    day. An ex-smoker does not smoke daily now, so RF:smoking is denied."""
    found = findings_of(records, "p-0001")
    assert found["RF:smoking"].assertion is Assertion.ABSENT
    assert found["RF:smoking"].qualifiers["snomed"] == "8517006"


def test_present_wins_when_the_history_says_both(records):
    found = findings_of(records, "p-0004")
    assert found["RF:smoking"].assertion is Assertion.PRESENT
    assert found["RF:smoking"].qualifiers["snomed"] == "449868002,8517006"
    assert found["RF:heart_failure"].assertion is Assertion.PRESENT


def test_pregnancy_now_obesity_and_a_never_smoker(records):
    found = findings_of(records, "p-0002")
    assert found["RF:pregnancy"].qualifiers["since"] == "2025-06-01", "the current pregnancy"
    assert found["RF:obesity"].assertion is Assertion.PRESENT
    assert found["RF:smoking"].assertion is Assertion.ABSENT


def test_findings_are_contract_findings_from_synthea(records):
    for patient_id in ("p-0001", "p-0002", "p-0004"):
        findings = list(findings_of(records, patient_id).values())
        assert findings and all(f.source == "synthea" for f in findings)
        assert all(f.label == BY_CONCEPT[f.concept_id].label for f in findings)
        record = records[patient_id]
        PatientCase(
            case_id=patient_id,
            age=record.age_on(AS_OF),
            sex=sex_of(record),
            risk_factors=findings,
        )


def test_unmapped_active_conditions_are_kept_with_their_reason(records):
    def unmapped(patient_id):
        graph = build_patient_graph(records[patient_id], AS_OF)
        return {row["code"]: row["reason"] for row in unmapped_codes(graph)}

    assert unmapped("p-0001") == {"302870006": NOT_MAPPED["302870006"]}, "prediabetes has ended"
    assert unmapped("p-0004") == {"49436004": NOT_MAPPED["49436004"]}
    assert unmapped("p-0005") == {"10509002": "not in the SNOMED -> RF table"}
    assert findings_of(records, "p-0005") == {}


def test_the_graph_holds_the_history_and_medications_only_as_history(records):
    graph = build_patient_graph(records["p-0001"], AS_OF)
    assert graph.nodes[PATIENT] == {"kind": "patient", "age": 65, "sex": "M", "alive": True}
    relations = {d["relation"] for _, _, d in graph.out_edges(PATIENT, data=True)}
    assert relations == {"HAS_CONDITION", "HAS_OBSERVATION", "TOOK_MEDICATION", "HAD_PROCEDURE"}
    medications = [n for n, kind in graph.nodes(data="kind") if kind == "medication"]
    assert len(medications) == 1
    assert not list(graph.out_edges(medications[0])), "a medication indicates nothing"
    heights = [n for n, d in graph.nodes(data=True) if d.get("code") == "8302-2"]
    assert [graph.nodes[n]["value"] for n in heights] == ["177.5"], "the latest only"
    assert graph.graph == {"patient_id": "p-0001", "as_of": "2026-01-01"}


def test_the_patient_kg_is_not_wired_into_the_ranking():
    """docs/17 T7: whether a history feeds a case is the owner's decision."""
    for path in ("src/pipeline.py", "scripts/_config.py", "src/api/app.py", "app/view.py"):
        assert "patient_kg" not in (REPO_ROOT / path).read_text(encoding="utf-8"), path


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #


def run_builder(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )


def test_the_builder_writes_findings_and_a_summary(tmp_path):
    result = run_builder("--synthea", str(FIXTURE), "--as-of", "2026-01-01", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "3 active condition code(s) the table does not map" in result.stdout
    lines = [json.loads(line) for line in (tmp_path / "findings.jsonl").open(encoding="utf-8")]
    assert [row["patient_id"] for row in lines] == ["p-0001", "p-0002", "p-0004", "p-0005"]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert (summary["patients"], summary["alive_on_as_of"]) == (5, 4)
    assert summary["concepts"]["RF:smoking"] == {"present": 1, "absent": 2}
    assert summary["concepts"]["RF:hypertension"] == {"present": 1, "absent": 0}, "p-0003 died"
    assert set(summary["unmapped_active_conditions"]) == {"10509002", "302870006", "49436004"}
    assert summary["synthea_version"] == "v4.0.0" and summary["as_of"] == "2026-01-01"


def test_the_builder_says_why_when_there_is_no_synthea_output(tmp_path):
    result = run_builder("--synthea", str(tmp_path), "--out", str(tmp_path / "out"))
    assert result.returncode == 1 and "no patients.csv" in result.stderr
