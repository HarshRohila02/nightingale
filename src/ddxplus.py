"""DDXPlus patient rows: the evidence-token grammar, labels, and the test-split gate.

Each DDXPlus patient is one CSV row. Its ``EVIDENCES`` cell is the Python repr of a
list of tokens, and every token takes one of four forms (EXP-002):

    E_91          binary evidence, present              fever -> yes
    E_54_@_V_161  categorical or multi-choice value     pain character -> "sensitive"
    E_56_@_4      numeric ordinal on a 0-10 scale       pain intensity -> 4
    E_54_@_V_11   the "NA" sentinel                     pain character -> not applicable

**A listed token does not mean the finding is present.** Every categorical,
multi-choice and ordinal evidence has a ``default_value`` that means "no" or "none".
``E_204_@_V_10`` is "travelled abroad: N" and is listed for about 90% of patients;
``E_57_@_V_123`` is "the pain radiates: nowhere". :func:`positive_codes` is the one
place that answers "which evidences does this patient actually have", by comparing
each value with that evidence's default in ``release_evidences.json``.

Work with codes, never the English labels: DDXPlus English is machine-translated from
French and unreliable (``déchirante``, *tearing*, became "heartbreaking").

Standard library only, so every workstream can import it.

Reference: docs/03-data-management.md §1.1 and §4, docs/05-evaluation-protocol.md §2.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.conditions import from_ddxplus_label

__all__ = [
    "CONCEPT_PREFIX",
    "INPUT_COLUMNS",
    "LABEL_COLUMNS",
    "METADATA_COLUMNS",
    "NA_VALUE",
    "SPLITS",
    "TOKEN_SEP",
    "EvidenceSpec",
    "EvidenceToken",
    "TokenKind",
    "check_split_allowed",
    "code_sort_key",
    "concept_id",
    "decode_differential",
    "decode_row",
    "is_positive",
    "load_evidence_specs",
    "parse_list_cell",
    "parse_token",
    "positive_codes",
]

TOKEN_SEP = "_@_"
NA_VALUE = "V_11"
SPLITS: tuple[str, ...] = ("train", "validate", "test")
CONCEPT_PREFIX = "DDX:"

INPUT_COLUMNS: tuple[str, ...] = ("age", "sex", "evidences", "positive_codes")
"""Model inputs. docs/05 §2 allows AGE, SEX and EVIDENCES; ``positive_codes`` is
derived from EVIDENCES alone."""

LABEL_COLUMNS: tuple[str, ...] = ("label_condition_id", "label_pathology", "label_differential")
"""Ground truth — never a model input. ``label_differential`` *is* DDXPlus
DIFFERENTIAL_DIAGNOSIS; training on it is the leakage docs/03 §4 warns about."""

METADATA_COLUMNS: tuple[str, ...] = ("case_id", "split", "source_row", "initial_evidence")
"""Neither input nor label. INITIAL_EVIDENCE is always one of the EVIDENCES and docs/05 §2
does not list it as an input, so using it as a feature would need a protocol amendment."""

_CODE = re.compile(r"^E_\d+$")
_VALUE = re.compile(r"^V_\d+$")
_ORDINAL = re.compile(r"^\d+$")


class TokenKind(str, Enum):
    """The four token forms. Feature encoding (1c) must treat each differently."""

    BINARY = "binary"
    VALUE = "value"
    ORDINAL = "ordinal"
    NA = "na"


@dataclass(frozen=True)
class EvidenceToken:
    """One parsed token from a patient's EVIDENCES list."""

    raw: str
    code: str
    kind: TokenKind
    value: str | None = None
    ordinal: int | None = None


@dataclass(frozen=True)
class EvidenceSpec:
    """What ``release_evidences.json`` says about one evidence question.

    Attributes:
        code: e.g. ``E_204``.
        data_type: ``B`` binary, ``C`` categorical or ordinal, ``M`` multi-choice.
        default_value: The answer that means "no" / "none", as a string (``"V_10"``,
            ``"0"``). None for binary evidences, which are simply absent when "no".
        is_antecedent: DDXPlus's own risk-factor flag. Not always consistent with how
            ``release_conditions.json`` uses the code (E_16, "anxious", is flagged an
            antecedent but listed as a symptom).
        possible_values: Every answer the question allows, as strings (``"V_16"``, ``"4"``),
            default included. Empty for binary evidences.
    """

    code: str
    data_type: str
    default_value: str | None
    is_antecedent: bool
    possible_values: tuple[str, ...] = ()

    @property
    def is_ordinal(self) -> bool:
        """A 0-10 scale (``E_56_@_4``): categorical, with numbers for answers."""
        return (
            self.data_type == "C"
            and bool(self.possible_values)
            and all(v.isdigit() for v in self.possible_values)
        )


def code_sort_key(code: str) -> int:
    """Numeric order for evidence codes, so that E_2 < E_10 < E_100."""
    return int(code.split("_", 2)[1])


def concept_id(code: str) -> str:
    """The knowledge-graph concept id of a DDXPlus evidence question: E_53 -> DDX:E_53.

    The ``DDX:`` namespace keeps DDXPlus-derived concepts apart from the hand-authored
    ``SYM:*`` / ``RF:*`` ids that the red-flag rules and golden cases use. Mapping one
    onto the other is the crosswalk, a separate 1b task.
    """
    if not _CODE.match(code):
        raise ValueError(f"not a DDXPlus evidence code: {code!r}")
    return f"{CONCEPT_PREFIX}{code}"


def parse_token(raw: str) -> EvidenceToken:
    """Parse one EVIDENCES token.

    Raises:
        ValueError: if ``raw`` is not a DDXPlus evidence token. Corrupt input must fail
            loudly rather than become a silently missing finding.
    """
    code, sep, value = raw.partition(TOKEN_SEP)
    if not _CODE.match(code):
        raise ValueError(f"not a DDXPlus evidence token: {raw!r}")
    if not sep:
        return EvidenceToken(raw=raw, code=code, kind=TokenKind.BINARY)
    if _ORDINAL.match(value):
        return EvidenceToken(raw=raw, code=code, kind=TokenKind.ORDINAL, ordinal=int(value))
    if _VALUE.match(value):
        kind = TokenKind.NA if value == NA_VALUE else TokenKind.VALUE
        return EvidenceToken(raw=raw, code=code, kind=kind, value=value)
    raise ValueError(f"not a DDXPlus evidence token: {raw!r}")


def parse_list_cell(cell: str) -> list[Any]:
    """Parse a CSV cell holding a Python list literal, such as EVIDENCES.

    ``ast.literal_eval`` evaluates literals only, never code, so a malformed or
    hostile cell raises instead of executing.
    """
    value = ast.literal_eval(cell)
    if not isinstance(value, list):
        raise ValueError(f"expected a list literal, got {type(value).__name__}")
    return value


def load_evidence_specs(path: Path) -> dict[str, EvidenceSpec]:
    """Load ``release_evidences.json`` into {code: EvidenceSpec}.

    Also checks the assumption :func:`parse_token` relies on: wherever ``V_11`` is a
    defined value, it means "NA".
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    specs: dict[str, EvidenceSpec] = {}
    for code, spec in raw.items():
        na_meaning = (spec.get("value_meaning") or {}).get(NA_VALUE)
        if na_meaning is not None and (na_meaning.get("en") or "").upper() != "NA":
            raise ValueError(f"{code}: {NA_VALUE} does not mean NA ({na_meaning!r})")
        data_type = spec.get("data_type", "B")
        default = spec.get("default_value")
        specs[code] = EvidenceSpec(
            code=code,
            data_type=data_type,
            default_value=None if data_type == "B" or default is None else str(default),
            is_antecedent=bool(spec.get("is_antecedent", False)),
            possible_values=(
                () if data_type == "B" else tuple(str(v) for v in spec.get("possible-values", []))
            ),
        )
    return specs


def is_positive(token: EvidenceToken, specs: Mapping[str, EvidenceSpec]) -> bool:
    """True if the token asserts that the patient *has* this evidence.

    Binary tokens are positive by being listed. An NA token never is. A value or an
    ordinal is positive unless it equals the evidence's default: "N", "nowhere", or 0.
    An ordinal 0 can be a real answer (E_59, "how fast did the pain appear?"), but
    the raw value stays in ``evidences`` for the ML features either way.

    Raises:
        ValueError: if the code is not in ``release_evidences.json``.
    """
    spec = specs.get(token.code)
    if spec is None:
        raise ValueError(f"evidence {token.code} is not in release_evidences.json")
    if token.kind is TokenKind.BINARY:
        return True
    if token.kind is TokenKind.NA:
        return False
    observed = token.value if token.kind is TokenKind.VALUE else str(token.ordinal)
    return observed != spec.default_value


def positive_codes(tokens: Iterable[EvidenceToken], specs: Mapping[str, EvidenceSpec]) -> list[str]:
    """The evidence *questions* a patient answered with something other than "no".

    This is the level the knowledge graph works at: ``release_conditions.json`` links
    conditions to questions (``E_54``, "characterize your pain"), not to answers.
    """
    codes = {t.code for t in tokens if is_positive(t, specs)}
    return sorted(codes, key=code_sort_key)


def decode_differential(cell: str) -> list[dict[str, Any]]:
    """Decode a DIFFERENTIAL_DIAGNOSIS cell. This is a LABEL, never an input.

    Returns:
        One dict per entry, in DDXPlus order (probability, descending):
        ``{"pathology", "condition_id", "probability"}``. ``condition_id`` is None for
        the pathologies outside Nightingale's scope, which are kept, not dropped: they
        are part of the ground truth (risk R-13).
    """
    entries = []
    for pathology, probability in parse_list_cell(cell):
        condition = from_ddxplus_label(pathology)
        entries.append(
            {
                "pathology": pathology,
                "condition_id": condition.id if condition else None,
                "probability": float(probability),
            }
        )
    return entries


def decode_row(
    *,
    split: str,
    source_row: int,
    age: int,
    sex: str,
    pathology: str,
    evidences_cell: str,
    initial_evidence: str,
    differential_cell: str,
    specs: Mapping[str, EvidenceSpec],
) -> dict[str, Any] | None:
    """Decode one raw DDXPlus row into a parquet record.

    Returns:
        A dict keyed exactly by METADATA_COLUMNS + INPUT_COLUMNS + LABEL_COLUMNS, or
        None when the row's pathology is outside the 13 in-scope conditions.

    Raises:
        ValueError: on a malformed token, an unknown evidence code, or an unknown sex.
    """
    condition = from_ddxplus_label(pathology)
    if condition is None:
        return None
    if sex not in ("M", "F"):
        raise ValueError(f"row {source_row}: unexpected SEX {sex!r}")

    raw_tokens = parse_list_cell(evidences_cell)
    tokens = [parse_token(str(t)) for t in raw_tokens]
    return {
        "case_id": f"ddxplus-{split}-{source_row:07d}",
        "split": split,
        "source_row": int(source_row),
        "initial_evidence": initial_evidence,
        "age": int(age),
        "sex": sex,
        "evidences": [t.raw for t in tokens],
        "positive_codes": positive_codes(tokens, specs),
        "label_condition_id": condition.id,
        "label_pathology": pathology,
        "label_differential": decode_differential(differential_cell),
    }


def check_split_allowed(split: str, *, allow_test_split: bool) -> None:
    """Refuse the test split unless Phase 4 has opened it.

    Call this BEFORE opening the file. ``allow_test_split`` comes from
    ``evaluation.allow_test_split`` in configs/config.yaml, the single switch, which is
    flipped once, in Phase 4 (docs/05-evaluation-protocol.md §2, risk R-08).

    Raises:
        ValueError: for an unknown split name.
        PermissionError: for the test split while it is closed.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown DDXPlus split {split!r}; expected one of {SPLITS}")
    if split == "test" and not allow_test_split:
        raise PermissionError(
            "Refusing to read the DDXPlus test split: it is opened once, in Phase 4 "
            "(docs/05-evaluation-protocol.md §2). Set evaluation.allow_test_split: true "
            "in configs/config.yaml only when Phase 4 begins."
        )
