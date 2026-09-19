"""Tests for feature encoding (task 1c, src/ml/features.py).

The specs are synthetic, in the shape load_evidence_specs returns, with codes that mirror real
DDXPlus ones. The test on the real validate parquet skips when data/ is absent (as in CI).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ddxplus import EvidenceSpec, parse_token, positive_codes
from src.ml.features import EvidenceEncoder, evidence_codes_in_scope

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_EVIDENCES = REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json"
REAL_CONDITIONS = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json"
REAL_PARQUET = REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_validate.parquet"
needs_real_data = pytest.mark.skipif(
    not all(p.exists() for p in (REAL_EVIDENCES, REAL_CONDITIONS, REAL_PARQUET)),
    reason="data/ is not committed (CI)",
)

SCALE = tuple(str(i) for i in range(11))
SPECS = {
    "E_91": EvidenceSpec("E_91", "B", None, False),
    "E_79": EvidenceSpec("E_79", "B", None, True),
    # multi-choice, NA is the default (pain character)
    "E_54": EvidenceSpec("E_54", "M", "V_11", False, ("V_11", "V_112", "V_71")),
    # multi-choice, "nowhere" is the default (pain location)
    "E_55": EvidenceSpec("E_55", "M", "V_123", False, ("V_123", "V_16", "V_14", "V_15")),
    "E_56": EvidenceSpec("E_56", "C", "0", False, SCALE),  # intensity, 0-10
    "E_59": EvidenceSpec("E_59", "C", "0", False, SCALE),  # onset speed: 0 is a real answer
    "E_204": EvidenceSpec("E_204", "C", "V_10", True, ("V_10", "V_0", "V_1")),  # travel: N
    # not in DDXPlus's scope: NA as an allowed answer that is not the default
    "E_300": EvidenceSpec("E_300", "M", "V_123", False, ("V_123", "V_20", "V_11")),
}


@pytest.fixture(scope="module")
def encoder() -> EvidenceEncoder:
    return EvidenceEncoder(SPECS, SPECS)


def features(encoder: EvidenceEncoder, *tokens: str, age: float = 50, sex: str = "M") -> dict:
    """The row for these tokens, as {column name: value}, nonzero columns only."""
    row = encoder.encode(age, sex, list(tokens))
    return {n: float(v) for n, v in zip(encoder.feature_names, row, strict=True) if v}


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


def test_the_columns_come_from_the_specs_in_a_fixed_order(encoder):
    assert encoder.feature_names == (
        "age",
        "sex=F",
        "E_54",
        "E_54=V_71",
        "E_54=V_112",
        "E_55",
        "E_55=V_14",
        "E_55=V_15",
        "E_55=V_16",
        "E_56=value",
        "E_56=answered",
        "E_59=value",
        "E_59=answered",
        "E_79",
        "E_91",
        "E_204",
        "E_204=V_0",
        "E_204=V_1",
        "E_300",
        "E_300=NA",
        "E_300=V_20",
    )
    assert encoder.codes == ("E_54", "E_55", "E_56", "E_59", "E_79", "E_91", "E_204", "E_300")


def test_the_fingerprint_names_the_column_set(encoder):
    assert EvidenceEncoder(SPECS, SPECS).fingerprint == encoder.fingerprint
    fewer = {code: spec for code, spec in SPECS.items() if code != "E_300"}
    assert EvidenceEncoder(fewer, fewer).fingerprint != encoder.fingerprint


def test_an_evidence_missing_from_the_specs_is_an_error():
    with pytest.raises(ValueError, match="E_999"):
        EvidenceEncoder(SPECS, [*SPECS, "E_999"])


# --------------------------------------------------------------------------- #
# Each token form
# --------------------------------------------------------------------------- #


def test_age_and_sex(encoder):
    assert features(encoder, age=63, sex="F") == {"age": 63.0, "sex=F": 1.0}
    assert features(encoder, age=40, sex="M") == {"age": 40.0}
    with pytest.raises(ValueError, match="sex"):
        encoder.encode(40, "X", [])


def test_a_binary_evidence_is_one_when_listed(encoder):
    assert features(encoder, "E_91", "E_79") == {"age": 50.0, "E_79": 1.0, "E_91": 1.0}


def test_multi_choice_answers_are_multi_hot_under_their_question(encoder):
    row = features(encoder, "E_55_@_V_14", "E_55_@_V_16")
    assert row == {"age": 50.0, "E_55": 1.0, "E_55=V_14": 1.0, "E_55=V_16": 1.0}


def test_a_default_answer_means_no(encoder):
    """Travelled abroad: N, and radiates: nowhere, add nothing (EXP-013)."""
    assert features(encoder, "E_204_@_V_10", "E_55_@_V_123") == {"age": 50.0}


def test_ordinals_stay_ordered_numbers_with_an_answered_flag(encoder):
    row = features(encoder, "E_56_@_7", "E_59_@_0")
    assert row == {"age": 50.0, "E_56=value": 7.0, "E_56=answered": 1.0, "E_59=answered": 1.0}
    assert "E_59=answered" not in features(encoder), "not listed is not the same as 0"


def test_na_is_never_a_positive_answer(encoder):
    assert features(encoder, "E_54_@_V_11") == {"age": 50.0}, "NA is E_54's default"
    assert features(encoder, "E_300_@_V_11") == {"age": 50.0, "E_300=NA": 1.0}


def test_token_order_does_not_matter(encoder):
    tokens = ["E_91", "E_55_@_V_16", "E_56_@_3", "E_204_@_V_1", "E_300_@_V_20"]
    assert np.array_equal(
        encoder.encode(30, "F", tokens), encoder.encode(30, "F", list(reversed(tokens)))
    )


@pytest.mark.parametrize(
    "token",
    [
        "E_999",  # not an encoded evidence
        "E_55_@_V_999",  # not an answer to E_55
        "E_91_@_V_1",  # a yes/no evidence given an answer
        "E_56_@_11",  # off the 0-10 scale
        "E_56_@_V_14",  # an ordinal given a categorical answer
        "E_55",  # a multi-choice evidence with no answer
        "E_204_@_3",  # a categorical evidence given a number
        "E_56_@_V_11",  # NA, which the scale does not allow
        "fever",  # not a token at all
    ],
)
def test_malformed_input_fails_loudly(encoder, token):
    with pytest.raises(ValueError):
        encoder.encode(50, "M", [token])


# --------------------------------------------------------------------------- #
# The question columns are the patient's positive codes
# --------------------------------------------------------------------------- #

PATIENTS = [
    [],
    ["E_91", "E_55_@_V_14", "E_56_@_5"],
    ["E_54_@_V_11", "E_55_@_V_123", "E_59_@_0", "E_204_@_V_10"],
    ["E_54_@_V_71", "E_54_@_V_112", "E_204_@_V_0", "E_300_@_V_11", "E_79"],
    ["E_300_@_V_20", "E_59_@_9", "E_55_@_V_15", "E_55_@_V_16"],
]


@pytest.mark.parametrize("tokens", PATIENTS)
def test_question_columns_hold_exactly_the_positive_codes(encoder, tokens):
    row = encoder.encode(50, "M", tokens)
    assert decoded_positive_codes(encoder, row) == set(
        positive_codes([parse_token(t) for t in tokens], SPECS)
    )


def decoded_positive_codes(encoder: EvidenceEncoder, row: np.ndarray) -> set[str]:
    """Rebuild positive_codes from a row: questions and binaries set, ordinals above 0."""
    names = encoder.feature_names
    positive = {name for name, value in zip(names, row, strict=True) if value and "=" not in name}
    positive.discard("age")
    positive |= {
        name.split("=")[0]
        for name, value in zip(names, row, strict=True)
        if name.endswith("=value") and value > 0
    }
    return positive


# --------------------------------------------------------------------------- #
# Many patients
# --------------------------------------------------------------------------- #


def test_transform_matches_encode_row_by_row(encoder):
    frame = pd.DataFrame(
        {"age": [20, 70, 45, 33, 58], "sex": ["F", "M", "F", "M", "F"], "evidences": PATIENTS}
    )
    matrix = encoder.transform(frame)
    assert matrix.dtype == np.float32 and matrix.shape == (5, len(encoder.feature_names))
    for index, (age, sex, tokens) in enumerate(frame.itertuples(index=False)):
        assert np.array_equal(matrix[index], encoder.encode(age, sex, tokens))
    assert np.array_equal(encoder.transform(frame.to_dict("records")), matrix)


def test_transform_never_reads_a_label(encoder):
    """docs/03 §4: only age, sex and evidences are inputs."""
    inputs = pd.DataFrame({"age": [20, 70], "sex": ["F", "M"], "evidences": PATIENTS[1:3]})
    labelled = inputs.assign(
        label_condition_id=["COND:gerd", "COND:psvt"],
        label_differential=[[{"pathology": "GERD"}], []],
        positive_codes=[["not", "used"], []],
    )
    relabelled = labelled.assign(label_condition_id=["COND:psvt", "COND:gerd"])
    expected = encoder.transform(inputs)
    assert np.array_equal(encoder.transform(labelled), expected)
    assert np.array_equal(encoder.transform(relabelled), expected)


def test_an_error_names_the_row(encoder):
    frame = pd.DataFrame(
        {"age": [20, 30, 40], "sex": ["F", "M", "F"], "evidences": [["E_91"], ["E_91"], ["E_999"]]}
    )
    with pytest.raises(ValueError, match="row 2"):
        encoder.transform(frame)


def test_codes_in_scope_come_from_the_conditions_file(tmp_path):
    path = tmp_path / "ddxplus_chestpain_conditions.json"
    path.write_text(
        '{"COND:gerd": {"symptoms": [{"code": "E_91"}], "antecedents": [{"code": "E_79"}]},'
        ' "COND:psvt": {"symptoms": [{"code": "E_91"}, {"code": "E_56"}]}}',
        encoding="utf-8",
    )
    assert evidence_codes_in_scope(path) == {"E_91", "E_79", "E_56"}


# --------------------------------------------------------------------------- #
# The real validate split (local only)
# --------------------------------------------------------------------------- #


@needs_real_data
def test_every_validate_patient_encodes_and_matches_their_positive_codes():
    encoder = EvidenceEncoder.from_release(REAL_EVIDENCES, REAL_CONDITIONS)
    frame = pd.read_parquet(REAL_PARQUET, columns=["age", "sex", "evidences", "positive_codes"])
    start = time.perf_counter()
    matrix = encoder.transform(frame)
    elapsed = time.perf_counter() - start

    assert matrix.shape == (33_963, len(encoder.feature_names))
    assert np.isfinite(matrix).all()
    assert len(encoder.codes) == 84
    mismatches = [
        index
        for index, codes in enumerate(frame["positive_codes"])
        if decoded_positive_codes(encoder, matrix[index]) != set(codes)
    ]
    assert not mismatches, f"{len(mismatches)} patients differ, first {mismatches[:5]}"
    answered = [i for i, name in enumerate(encoder.feature_names) if name.endswith("=answered")]
    assert matrix[:, answered].sum() == 94_062, "every ordinal token (docs/03 §2.1) is encoded"
    assert elapsed < 60, f"encoding took {elapsed:.1f} s"
