"""Tests for the DDXPlus row decoder (src/ddxplus.py) and the 1a parquet builder.

All fixtures are tiny, hand-written and synthetic — nothing is copied from DDXPlus.
The one test that reads the real release_evidences.json skips when data/ is absent
(as it is in CI, because data/ is never committed).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.ddxplus import (
    INPUT_COLUMNS,
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    EvidenceSpec,
    TokenKind,
    check_split_allowed,
    code_sort_key,
    decode_differential,
    decode_row,
    is_positive,
    load_evidence_specs,
    parse_list_cell,
    parse_token,
    positive_codes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_ddxplus_chestpain.py"
REAL_EVIDENCES = REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json"

# A synthetic evidence spec in the shape of release_evidences.json. The codes mirror
# real DDXPlus ones so the examples read naturally; the content is our own.
SPEC_JSON = {
    "E_53": {"data_type": "B", "default_value": 0, "is_antecedent": False},
    "E_66": {"data_type": "B", "default_value": 0, "is_antecedent": False},
    "E_79": {"data_type": "B", "default_value": 0, "is_antecedent": True},
    "E_54": {
        "data_type": "M",
        "default_value": "V_11",
        "is_antecedent": False,
        "value_meaning": {"V_11": {"en": "NA"}, "V_161": {"en": "sensitive"}},
    },
    "E_56": {"data_type": "C", "default_value": 0, "is_antecedent": False},
    "E_57": {"data_type": "M", "default_value": "V_123", "is_antecedent": False},
    "E_204": {"data_type": "C", "default_value": "V_10", "is_antecedent": True},
}


@pytest.fixture
def specs(tmp_path: Path) -> dict[str, EvidenceSpec]:
    path = tmp_path / "release_evidences.json"
    path.write_text(json.dumps(SPEC_JSON), encoding="utf-8")
    return load_evidence_specs(path)


# --------------------------------------------------------------------------- #
# Token grammar
# --------------------------------------------------------------------------- #


class TestParseToken:
    def test_binary(self):
        token = parse_token("E_91")
        assert (token.code, token.kind, token.value, token.ordinal) == (
            "E_91",
            TokenKind.BINARY,
            None,
            None,
        )

    def test_categorical_value(self):
        token = parse_token("E_54_@_V_161")
        assert (token.code, token.kind, token.value) == ("E_54", TokenKind.VALUE, "V_161")

    def test_numeric_ordinal(self):
        token = parse_token("E_56_@_4")
        assert (token.code, token.kind, token.ordinal) == ("E_56", TokenKind.ORDINAL, 4)

    def test_na_sentinel(self):
        assert parse_token("E_54_@_V_11").kind is TokenKind.NA

    @pytest.mark.parametrize("raw", ["", "E_", "X_12", "E_54_@_", "E_54_@_high", "e_54"])
    def test_malformed_tokens_fail_loudly(self, raw):
        with pytest.raises(ValueError):
            parse_token(raw)

    def test_codes_sort_numerically(self):
        assert sorted(["E_100", "E_2", "E_10"], key=code_sort_key) == ["E_2", "E_10", "E_100"]


class TestPositiveCodes:
    """A listed token is not necessarily a present finding (the 'tokens that mean no')."""

    def test_binary_is_positive_by_being_listed(self, specs):
        assert is_positive(parse_token("E_53"), specs)

    def test_default_value_means_no(self, specs):
        # "travelled abroad: N" — listed for ~90% of real patients, yet a "no".
        assert not is_positive(parse_token("E_204_@_V_10"), specs)
        assert is_positive(parse_token("E_204_@_V_7"), specs)

    def test_nowhere_means_no(self, specs):
        assert not is_positive(parse_token("E_57_@_V_123"), specs)

    def test_na_sentinel_is_never_positive(self, specs):
        assert not is_positive(parse_token("E_54_@_V_11"), specs)

    def test_ordinal_zero_is_the_default(self, specs):
        assert not is_positive(parse_token("E_56_@_0"), specs)
        assert is_positive(parse_token("E_56_@_7"), specs)

    def test_unknown_code_fails_loudly(self, specs):
        with pytest.raises(ValueError, match="E_999"):
            is_positive(parse_token("E_999"), specs)

    def test_positive_codes_deduplicate_sort_and_drop_the_nos(self, specs):
        tokens = [
            parse_token(t)
            for t in ["E_204_@_V_10", "E_66", "E_54_@_V_161", "E_54_@_V_161", "E_57_@_V_123"]
        ]
        assert positive_codes(tokens, specs) == ["E_54", "E_66"]

    def test_na_meaning_is_verified_on_load(self, tmp_path):
        bad = {"E_1": {"data_type": "M", "value_meaning": {"V_11": {"en": "left arm"}}}}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="does not mean NA"):
            load_evidence_specs(path)


class TestCells:
    def test_list_cell_parses_literals_only(self):
        assert parse_list_cell("['E_53', 'E_66']") == ["E_53", "E_66"]
        with pytest.raises(ValueError):
            parse_list_cell("__import__('os').getcwd()")

    def test_list_cell_rejects_non_lists(self):
        with pytest.raises(ValueError, match="list"):
            parse_list_cell("{'E_53': 1}")

    def test_differential_keeps_out_of_scope_entries(self):
        entries = decode_differential("[['Pulmonary embolism', 0.6], ['Anemia', 0.4]]")
        assert entries == [
            {
                "pathology": "Pulmonary embolism",
                "condition_id": "COND:pulmonary_embolism",
                "probability": 0.6,
            },
            {"pathology": "Anemia", "condition_id": None, "probability": 0.4},
        ]


# --------------------------------------------------------------------------- #
# Rows and column roles
# --------------------------------------------------------------------------- #

ROW = {
    "split": "validate",
    "source_row": 7,
    "age": 58,
    "sex": "M",
    "pathology": "Possible NSTEMI / STEMI",
    "evidences_cell": "['E_53', 'E_54_@_V_161', 'E_56_@_7', 'E_79', 'E_204_@_V_10']",
    "initial_evidence": "E_53",
    "differential_cell": "[['Possible NSTEMI / STEMI', 0.7], ['Anemia', 0.3]]",
}


class TestDecodeRow:
    def test_in_scope_row(self, specs):
        record = decode_row(**ROW, specs=specs)
        assert record is not None
        assert record["case_id"] == "ddxplus-validate-0000007"
        assert record["label_condition_id"] == "COND:nstemi_stemi"
        assert record["evidences"][-1] == "E_204_@_V_10", "raw tokens are kept as they were"
        assert record["positive_codes"] == ["E_53", "E_54", "E_56", "E_79"]

    def test_out_of_scope_row_is_dropped(self, specs):
        assert decode_row(**{**ROW, "pathology": "Anemia"}, specs=specs) is None

    def test_unexpected_sex_fails_loudly(self, specs):
        with pytest.raises(ValueError, match="SEX"):
            decode_row(**{**ROW, "sex": "X"}, specs=specs)

    def test_record_has_exactly_the_declared_columns(self, specs):
        record = decode_row(**ROW, specs=specs)
        assert list(record) == [*METADATA_COLUMNS, *INPUT_COLUMNS, *LABEL_COLUMNS]

    def test_the_differential_is_a_label_never_an_input(self):
        """docs/03 §4: training on DIFFERENTIAL_DIAGNOSIS is leakage."""
        assert "label_differential" in LABEL_COLUMNS
        assert not set(INPUT_COLUMNS) & (set(LABEL_COLUMNS) | set(METADATA_COLUMNS))
        assert not any("differential" in c or c.startswith("label_") for c in INPUT_COLUMNS)


class TestSplitGate:
    def test_test_split_is_refused_by_default(self):
        with pytest.raises(PermissionError, match="Phase 4"):
            check_split_allowed("test", allow_test_split=False)

    def test_test_split_opens_only_with_the_switch(self):
        check_split_allowed("test", allow_test_split=True)

    def test_development_splits_are_allowed(self):
        check_split_allowed("train", allow_test_split=False)
        check_split_allowed("validate", allow_test_split=False)

    def test_unknown_split(self):
        with pytest.raises(ValueError):
            check_split_allowed("valid", allow_test_split=False)


# --------------------------------------------------------------------------- #
# The builder script, end to end, on a synthetic raw directory
# --------------------------------------------------------------------------- #

CSV = """AGE,DIFFERENTIAL_DIAGNOSIS,SEX,PATHOLOGY,EVIDENCES,INITIAL_EVIDENCE
58,"[['Possible NSTEMI / STEMI', 0.7], ['Anemia', 0.3]]",M,Possible NSTEMI / STEMI,"['E_53', 'E_54_@_V_161', 'E_56_@_7', 'E_79', 'E_204_@_V_10']",E_53
30,"[['Anemia', 1.0]]",F,Anemia,"['E_66']",E_66
42,"[['Pulmonary embolism', 0.6], ['Spontaneous pneumothorax', 0.4]]",F,Pulmonary embolism,"['E_53', 'E_54_@_V_11', 'E_57_@_V_123', 'E_66']",E_66
"""


def _run_builder(tmp_path: Path, split: str, allow_test: bool) -> subprocess.CompletedProcess:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    (raw_dir / "release_evidences.json").write_text(json.dumps(SPEC_JSON), encoding="utf-8")
    (raw_dir / f"{split}.csv").write_text(CSV, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"evaluation:\n  allow_test_split: {str(allow_test).lower()}\n", encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(BUILDER), "--split", split, "--raw-dir", str(raw_dir)]
        + ["--out-dir", str(tmp_path / "out"), "--config", str(config)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


class TestBuilderScript:
    def test_writes_the_in_scope_rows_with_the_declared_schema(self, tmp_path):
        result = _run_builder(tmp_path, "validate", allow_test=False)
        assert result.returncode == 0, result.stderr

        table = pq.read_table(tmp_path / "out" / "ddxplus_chestpain_validate.parquet")
        assert table.column_names == [*METADATA_COLUMNS, *INPUT_COLUMNS, *LABEL_COLUMNS]
        rows = table.to_pylist()
        assert [r["label_condition_id"] for r in rows] == [
            "COND:nstemi_stemi",
            "COND:pulmonary_embolism",
        ]
        assert [r["source_row"] for r in rows] == [0, 2]
        assert rows[1]["positive_codes"] == ["E_53", "E_66"], "NA and 'nowhere' are not findings"
        assert rows[0]["label_differential"][1] == {
            "pathology": "Anemia",
            "condition_id": None,
            "probability": 0.3,
        }

    def test_summary_audits_the_labels(self, tmp_path):
        assert _run_builder(tmp_path, "validate", allow_test=False).returncode == 0
        summary = json.loads(
            (tmp_path / "out" / "ddxplus_chestpain_validate.summary.json").read_text("utf-8")
        )
        assert (summary["source_rows"], summary["in_scope_rows"]) == (3, 2)
        diff = summary["differential"]
        assert diff["rows_with_out_of_scope_entries"] == 1
        assert diff["out_of_scope_mass_mean"] == pytest.approx(0.15)
        # Row 1: D = {NSTEMI, Anemia}; a closed-world system reaches 1 of 2.
        assert diff["recall_at_5_ceiling_full_differential"] == pytest.approx(0.75)
        assert diff["recall_at_5_ceiling_in_scope_differential"] == pytest.approx(1.0)
        # 1 and 2 in-scope entries: even a perfect top-3 scores Precision@3 of 1/3 and 2/3.
        assert diff["precision_at_3_ceiling"] == pytest.approx(0.5)

    def test_refuses_the_test_split_before_reading_it(self, tmp_path):
        result = _run_builder(tmp_path, "test", allow_test=False)
        assert result.returncode == 2
        assert "Refusing to read the DDXPlus test split" in result.stderr
        assert not (tmp_path / "out").exists(), "nothing may be written from the test split"


@pytest.mark.skipif(not REAL_EVIDENCES.exists(), reason="data/ is not committed (CI)")
def test_real_evidence_spec_matches_the_documented_defaults():
    specs = load_evidence_specs(REAL_EVIDENCES)
    assert specs["E_204"].default_value == "V_10"  # travelled abroad: N
    assert specs["E_57"].default_value == "V_123"  # radiates: nowhere
    assert specs["E_54"].default_value == "V_11"  # pain character: NA
    assert specs["E_56"].default_value == "0"  # pain intensity 0-10
