"""Model features from DDXPlus evidence tokens (task 1c).

The ML ranker sees a patient as a fixed-length row of numbers. This module builds that row from
the model inputs only: ``age``, ``sex`` and ``evidences`` (``INPUT_COLUMNS`` in src/ddxplus.py).
It never reads a label, and it works with evidence codes rather than English labels, which are
machine-translated and unreliable.

**The columns depend only on the release files, never on patients.** Every machine builds the
same columns in the same order, and a hand-written case can use an answer no training patient
gave. :attr:`EvidenceEncoder.fingerprint` names the column set: a trained model records it and
must refuse to score rows encoded with any other.

How each token form (EXP-002) becomes columns:

* **binary**, ``E_91``: column ``E_91``, 1 if listed.
* **categorical or multi-choice**, ``E_55_@_V_16``: column ``E_55`` for the question, 1 if any
  listed answer is not the default; and one column per non-default answer, ``E_55=V_16``, 1 if
  that answer is listed. The default ("no", "nowhere", "N") gets no column, so it leaves the
  question at 0, exactly as if the question were not listed.
* **ordinal**, a 0-10 scale such as ``E_56_@_4``: ``E_56=value`` holds the number itself,
  ordered rather than one-hot, and 0 when not listed. ``E_56=answered`` is 1 if listed, because
  0 can be a real answer (``E_59``, how fast the pain appeared).
* **NA**, ``E_54_@_V_11``: never a positive answer. Where NA is the question's default (it is,
  for ``E_54``, the only question with NA tokens in scope), it adds nothing. Where it is another
  allowed answer, it gets its own column, ``E_nn=NA``, and still leaves the question at 0.

Plus ``age`` in years and ``sex=F``: 1 for female, 0 for male.

So the question and binary columns hold exactly the patient's ``positive_codes``, the level the
knowledge graph works at. The tests check that on every validate patient.

**The "asked" channel** (``asked_channel=True``, risk R-18). Without it, a default answer and a
question nobody asked both leave their columns at 0, so a short history looks exactly like a long
list of denials, and XGBoost answers atrial fibrillation to an empty row (EXP-017). The channel adds
one column per question, ``E_nn=unasked``, 1 when the question was not asked. It is stored as the
complement so that a full-evidence row stays sparse:

* **A DDXPlus patient at full evidence was asked everything**: an unlisted question means "no"
  (the closed world DDXPlus generates), so ``asked=None`` sets no ``unasked`` column. "Asked" is
  never read from the default answers DDXPlus happens to list (``E_204_@_V_10``, "did not
  travel"): those are listed only for conditions whose definition includes the question, so
  reading them would leak the label.
* **A masked patient** (:mod:`src.ml.evidence_masks`) or **a hand-authored case**
  (:mod:`src.ml.case_tokens`) passes the questions it asked; every other question is ``unasked``.
  A listed answer to a question not asked is an error.

The channel only means something to a model trained on masked copies of patients: at full
evidence it is constant. With the channel off, ``asked`` is ignored and the columns and the
fingerprint are exactly those the B1 models were trained on (``3a0d5a5e01d7f427``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ddxplus import (
    NA_VALUE,
    EvidenceSpec,
    TokenKind,
    code_sort_key,
    load_evidence_specs,
    parse_token,
)

__all__ = ["EvidenceEncoder", "evidence_codes_in_scope"]

SEXES = ("M", "F")


def evidence_codes_in_scope(conditions_path: Path) -> set[str]:
    """Every evidence code the 13 conditions' definitions use.

    Args:
        conditions_path: ``data/interim/ddxplus_chestpain_conditions.json``.
    """
    conditions = json.loads(Path(conditions_path).read_text(encoding="utf-8"))
    return {
        entry["code"]
        for spec in conditions.values()
        for key in ("symptoms", "antecedents")
        for entry in spec.get(key, [])
    }


def _value_order(value: str) -> int:
    """V_2 < V_10 < V_123."""
    return int(value.split("_", 1)[1])


def _inputs(
    patients: pd.DataFrame | Iterable[Mapping[str, Any]],
) -> tuple[Iterator[tuple[Any, Any, Any, Any]], int]:
    """(age, sex, evidences, asked) for each patient, and how many there are.

    Nothing else is read. ``asked`` is optional: a mask writes it (src/ml/evidence_masks.py), and
    without it a patient was asked everything.
    """
    if isinstance(patients, pd.DataFrame):
        asked = patients["asked"] if "asked" in patients else [None] * len(patients)
        columns = (patients["age"], patients["sex"], patients["evidences"], asked)
        return zip(*columns, strict=True), len(patients)
    records = list(patients)
    rows = [(r["age"], r["sex"], r["evidences"], r.get("asked")) for r in records]
    return iter(rows), len(records)


def _asked_set(asked: Any) -> frozenset[str] | None:
    """None for "asked everything"; otherwise the questions asked. A missing cell is None."""
    if asked is None or (isinstance(asked, float) and np.isnan(asked)):
        return None
    return frozenset(str(code) for code in asked)


class EvidenceEncoder:
    """Turns patients into a float32 matrix with one fixed column per feature.

    Args:
        specs: ``load_evidence_specs(release_evidences.json)``.
        codes: The evidences to encode: those the in-scope conditions use
            (:func:`evidence_codes_in_scope`). A patient token for any other evidence is an
            error, not a silently dropped finding.
        asked_channel: Add one ``E_nn=unasked`` column per evidence (R-18; module docstring).
            Off by default, which keeps the columns the B1 models were trained on.

    Raises:
        ValueError: if a code is not in ``specs``, or a categorical evidence lists no answers.
    """

    def __init__(
        self,
        specs: Mapping[str, EvidenceSpec],
        codes: Iterable[str],
        *,
        asked_channel: bool = False,
    ) -> None:
        ordered = sorted(set(codes), key=code_sort_key)
        missing = [code for code in ordered if code not in specs]
        if missing:
            raise ValueError(f"not in release_evidences.json: {missing}")
        self._specs = {code: specs[code] for code in ordered}
        names = ["age", "sex=F"]
        self._binary: dict[str, int] = {}
        self._question: dict[str, int] = {}
        self._answer: dict[tuple[str, str], int] = {}
        self._ordinal: dict[str, tuple[int, int]] = {}
        for code in ordered:
            spec = self._specs[code]
            if spec.data_type == "B":
                self._binary[code] = len(names)
                names.append(code)
            elif spec.is_ordinal:
                self._ordinal[code] = (len(names), len(names) + 1)
                names += [f"{code}=value", f"{code}=answered"]
            else:
                if not spec.possible_values:
                    raise ValueError(f"{code} is categorical but lists no possible answers")
                self._question[code] = len(names)
                names.append(code)
                for value in sorted(spec.possible_values, key=_value_order):
                    if value == spec.default_value:
                        continue
                    self._answer[(code, value)] = len(names)
                    names.append(f"{code}={'NA' if value == NA_VALUE else value}")
        self.asked_channel = asked_channel
        self._unasked: dict[str, int] = {}
        if asked_channel:  # appended, so the columns before them keep their positions
            for code in ordered:
                self._unasked[code] = len(names)
                names.append(f"{code}=unasked")
        self.feature_names: tuple[str, ...] = tuple(names)
        self.fingerprint: str = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_release(
        cls, evidences_path: Path, conditions_path: Path, *, asked_channel: bool = False
    ) -> EvidenceEncoder:
        """The encoder for the 13 in-scope conditions.

        Args:
            evidences_path: ``data/raw/ddxplus/release_evidences.json``.
            conditions_path: ``data/interim/ddxplus_chestpain_conditions.json``.
            asked_channel: As the constructor.
        """
        return cls(
            load_evidence_specs(evidences_path),
            evidence_codes_in_scope(conditions_path),
            asked_channel=asked_channel,
        )

    @property
    def codes(self) -> tuple[str, ...]:
        """The evidences this encoder knows, in column order."""
        return tuple(self._specs)

    def encode(
        self,
        age: float,
        sex: str,
        evidences: Sequence[str],
        asked: Iterable[str] | None = None,
    ) -> np.ndarray:
        """One patient as a 1-D float32 row.

        Args:
            age, sex, evidences: The model inputs.
            asked: The questions asked, for the "asked" channel. None means all of them, as for
                a DDXPlus patient at full evidence. Ignored when the channel is off.

        Raises:
            ValueError: on an unknown sex, an evidence the encoder does not know, an answer that
                evidence does not allow, or an answer to a question that was not asked.
        """
        row = np.zeros(len(self.feature_names), dtype=np.float32)
        self._fill(row, age, sex, evidences, asked)
        return row

    def transform(self, patients: pd.DataFrame | Iterable[Mapping[str, Any]]) -> np.ndarray:
        """Many patients as a 2-D float32 matrix, one row each, in order.

        Reads only ``age``, ``sex`` and ``evidences``, and ``asked`` when a mask wrote it. A data
        frame may hold other columns, labels included; they are never looked at.

        Raises:
            ValueError: naming the first row that cannot be encoded, and why.
        """
        rows, count = _inputs(patients)
        return self._dense(rows, count, offset=0)

    def transform_sparse(
        self, patients: pd.DataFrame | Iterable[Mapping[str, Any]], *, chunk: int = 20_000
    ) -> Any:
        """As :meth:`transform`, but a SciPy CSR matrix, built ``chunk`` rows at a time.

        About 28 of the 607 columns are nonzero per patient, so the train split takes about
        60 MB this way instead of 640 MB dense. The ``unasked`` columns are zero at full
        evidence, so the asked channel adds nothing there.
        """
        from scipy import sparse  # only the training job needs SciPy

        rows, count = _inputs(patients)
        blocks = [
            sparse.csr_matrix(self._dense(rows, min(chunk, count - start), offset=start))
            for start in range(0, count, chunk)
        ]
        if not blocks:
            return sparse.csr_matrix((0, len(self.feature_names)), dtype=np.float32)
        return sparse.vstack(blocks, format="csr")

    # -- internals --------------------------------------------------------- #

    def _dense(
        self, rows: Iterator[tuple[Any, Any, Any, Any]], count: int, *, offset: int
    ) -> np.ndarray:
        """The next ``count`` rows as a dense block. Errors name the row across all blocks."""
        matrix = np.zeros((count, len(self.feature_names)), dtype=np.float32)
        for index, (age, sex, evidences, asked) in zip(range(count), rows, strict=False):
            try:
                self._fill(matrix[index], age, sex, evidences, asked)
            except ValueError as exc:
                raise ValueError(f"row {offset + index}: {exc}") from exc
        return matrix

    def _fill(
        self, row: np.ndarray, age: float, sex: str, evidences: Sequence[str], asked: Any = None
    ) -> None:
        if sex not in SEXES:
            raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
        row[0] = float(age)
        row[1] = 1.0 if sex == "F" else 0.0
        for raw in evidences:
            token = parse_token(str(raw))
            spec = self._specs.get(token.code)
            if spec is None:
                raise ValueError(f"{raw}: {token.code} is not one of the encoded evidences")
            if token.code in self._binary:
                if token.kind is not TokenKind.BINARY:
                    raise ValueError(
                        f"{raw}: {token.code} is a yes/no evidence and takes no answer"
                    )
                row[self._binary[token.code]] = 1.0
            elif token.code in self._ordinal:
                if (
                    token.kind is not TokenKind.ORDINAL
                    or str(token.ordinal) not in spec.possible_values
                ):
                    raise ValueError(f"{raw}: {token.code} takes a number from its 0-10 scale")
                value_column, answered_column = self._ordinal[token.code]
                row[value_column] = float(token.ordinal)
                row[answered_column] = 1.0
            else:
                if token.kind not in (TokenKind.VALUE, TokenKind.NA):
                    raise ValueError(f"{raw}: {token.code} needs one of its answers (V_nn)")
                if token.value not in spec.possible_values:
                    raise ValueError(f"{raw}: {token.value} is not an answer to {token.code}")
                if token.value == spec.default_value:
                    continue
                row[self._answer[(token.code, token.value)]] = 1.0
                if token.kind is not TokenKind.NA:
                    row[self._question[token.code]] = 1.0
        if self.asked_channel:
            self._fill_unasked(row, evidences, _asked_set(asked))

    def _fill_unasked(
        self, row: np.ndarray, evidences: Sequence[str], asked: frozenset[str] | None
    ) -> None:
        if asked is None:
            return  # asked everything: the closed world of a full DDXPlus record
        unknown = sorted(asked - self._specs.keys(), key=code_sort_key)
        if unknown:
            raise ValueError(f"asked {unknown}, which are not encoded evidences")
        answered = {parse_token(str(raw)).code for raw in evidences}
        unasked_but_answered = sorted(answered - asked, key=code_sort_key)
        if unasked_but_answered:
            raise ValueError(f"{unasked_but_answered} answered but not asked")
        for code, column in self._unasked.items():
            if code not in asked:
                row[column] = 1.0
