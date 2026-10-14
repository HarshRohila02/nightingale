"""Reading Synthea's CSV output into per-patient records (task 1d).

Synthea writes one file per resource under ``<exporter.baseDirectory>/csv/``. This reads the five
the patient knowledge graph needs, by column name, so the columns a release adds or reorders do not
matter. The columns are those of Synthea's ``CSVConstants`` at the pinned release
(:data:`~src.patient_kg.mapping.SYNTHEA_VERSION`):

* ``patients.csv``: ``Id, BIRTHDATE, DEATHDATE, ..., GENDER, ...``;
* ``conditions.csv``: ``START, STOP, PATIENT, ENCOUNTER, SYSTEM, CODE, DESCRIPTION``;
* ``observations.csv``: ``DATE, PATIENT, ENCOUNTER, CATEGORY, CODE, DESCRIPTION, VALUE, UNITS,
  TYPE``, where a coded value is written as its display text;
* ``medications.csv``: ``START, STOP, PATIENT, ..., CODE, DESCRIPTION, ...``;
* ``procedures.csv``: ``START, STOP, PATIENT, ENCOUNTER, SYSTEM, CODE, DESCRIPTION, ...``.

Dates are ``YYYY-MM-DD`` or ISO 8601 timestamps; only the date is kept. Synthea's patients are
synthetic, so nothing here is personal data; the reader still keeps only what the graph uses
(no names, addresses or identifiers beyond Synthea's own patient id). Standard library only.

Reference: docs/16 §1, docs/17 T7.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

__all__ = ["CSV_FILES", "Entry", "PatientRecord", "parse_date", "read_synthea"]

CSV_FILES = ("patients", "conditions", "observations", "medications", "procedures")
"""The CSV files read, by Synthea's resource key."""


def parse_date(value: str | None) -> date | None:
    """``2020-05-01`` or ``2020-05-01T12:00:00Z`` -> the date; empty -> None."""
    value = (value or "").strip()
    return date.fromisoformat(value[:10]) if value else None


@dataclass(frozen=True)
class Entry:
    """One dated, coded fact: a condition, an observation, a medication or a procedure."""

    kind: str
    code: str
    description: str
    start: date | None
    stop: date | None = None
    value: str | None = None
    units: str | None = None

    def active_on(self, day: date) -> bool:
        """Begun by ``day`` and not yet ended (a stop on ``day`` itself has ended it)."""
        return (self.start is None or self.start <= day) and (self.stop is None or self.stop > day)


@dataclass
class PatientRecord:
    """One Synthea patient: who, and their coded history."""

    patient_id: str
    birth_date: date | None
    death_date: date | None
    gender: str
    entries: list[Entry] = field(default_factory=list)

    def age_on(self, day: date) -> int | None:
        if self.birth_date is None:
            return None
        before_birthday = (day.month, day.day) < (self.birth_date.month, self.birth_date.day)
        return day.year - self.birth_date.year - before_birthday

    def alive_on(self, day: date) -> bool:
        return self.death_date is None or self.death_date > day


def _rows(path: Path) -> Iterable[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_synthea(csv_dir: Path | str) -> dict[str, PatientRecord]:
    """Every patient in ``csv_dir`` with their conditions, observations, medications and
    procedures, in file order.

    Raises:
        FileNotFoundError: when ``patients.csv`` is missing; any other file may be absent (a
            population without, say, procedures).
    """
    csv_dir = Path(csv_dir)
    if not (csv_dir / "patients.csv").exists():
        raise FileNotFoundError(f"no patients.csv in {csv_dir}: is this Synthea's csv/ folder?")
    patients = {
        row["Id"]: PatientRecord(
            patient_id=row["Id"],
            birth_date=parse_date(row.get("BIRTHDATE")),
            death_date=parse_date(row.get("DEATHDATE")),
            gender=(row.get("GENDER") or "").strip().upper(),
        )
        for row in _rows(csv_dir / "patients.csv")
    }
    for kind, start_column in (
        ("conditions", "START"),
        ("observations", "DATE"),
        ("medications", "START"),
        ("procedures", "START"),
    ):
        for row in _rows(csv_dir / f"{kind}.csv"):
            record = patients.get(row.get("PATIENT", ""))
            if record is None:
                continue
            record.entries.append(
                Entry(
                    kind=kind.removesuffix("s"),
                    code=(row.get("CODE") or "").strip(),
                    description=" ".join((row.get("DESCRIPTION") or "").split()),
                    start=parse_date(row.get(start_column)),
                    stop=parse_date(row.get("STOP")),
                    value=" ".join((row.get("VALUE") or "").split()) or None,
                    units=(row.get("UNITS") or "").strip() or None,
                )
            )
    return patients
