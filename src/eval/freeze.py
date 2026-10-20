"""The test split is opened once, after the freeze (docs/05 §2, risk R-08).

Phase 4 reads the test split a single time, with everything fixed: the models, the fusion weight,
the calibrator, the red-flag rules, the graph. This module is the gate in front of that read.
``scripts/evaluate_ablations.py --split test`` calls :func:`require_freeze` before it loads a
single test label, and records the opening with :func:`record_opening`, so a second opening is
visible and has to give its reason.

What the gate checks, with ``git``:

* the freeze tag (``v0.9-freeze``) exists and is an ancestor of ``HEAD``;
* nothing the freeze fixed has changed since: ``configs/``, ``src/`` outside ``src/eval/``, and
  the golden cases. The evaluation code itself may be fixed after the freeze (a bug in a metric
  is not a change to the system), and every file changed since the tag is returned so the report
  can list them;
* the working tree is clean, so the code that scores is the code at ``HEAD``, which the result
  records.

It cannot see the models (``models/`` is gitignored); the runner records the hash of every file
it loads for that.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "FREEZE_TAG",
    "NotFrozen",
    "previous_openings",
    "record_opening",
    "require_freeze",
]

FREEZE_TAG = "v0.9-freeze"
PROTECTED = ("configs/", "src/", "tests/fixtures/")
"""Paths the freeze fixes."""
ALLOWED = ("src/eval/",)
"""Under a protected path, but evaluation code: it may be fixed after the freeze."""


class NotFrozen(RuntimeError):
    """The test split may not be opened yet; the message says why."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False, encoding="utf-8"
    )


def _protected(path: str) -> bool:
    return path.startswith(PROTECTED) and not path.startswith(ALLOWED)


def require_freeze(repo: Path, tag: str = FREEZE_TAG) -> dict[str, Any]:
    """Return what is being scored if the work is frozen; raise :class:`NotFrozen` if not.

    Returns:
        ``{"commit": HEAD, "tag": tag, "tag_commit": ..., "changed_since_freeze": [...]}``.
    """
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise NotFrozen(f"{repo} is not a git repository with a commit: {head.stderr.strip()}")
    found = _git(repo, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if found.returncode != 0:
        raise NotFrozen(f"the freeze tag {tag} does not exist: tag the frozen commit first")
    tagged = found.stdout.strip()
    if _git(repo, "merge-base", "--is-ancestor", tagged, "HEAD").returncode != 0:
        raise NotFrozen(f"{tag} ({tagged[:8]}) is not an ancestor of HEAD ({head.stdout[:8]})")
    status = _git(repo, "status", "--porcelain")
    if status.stdout.strip():
        dirty = ", ".join(line[3:] for line in status.stdout.splitlines()[:8])
        raise NotFrozen(f"the working tree is not clean ({dirty}): commit or stash first")
    changed = _git(repo, "diff", "--name-only", tagged, "HEAD").stdout.split()
    moved = [path for path in changed if _protected(path)]
    if moved:
        raise NotFrozen(
            f"{len(moved)} file(s) the freeze fixed changed since {tag}: {', '.join(moved[:8])}"
        )
    return {
        "commit": head.stdout.strip(),
        "tag": tag,
        "tag_commit": tagged,
        "changed_since_freeze": sorted(changed),
    }


def previous_openings(marker: Path) -> list[dict[str, Any]]:
    """The openings recorded so far, oldest first; empty if the split has never been opened."""
    if not marker.exists():
        return []
    return list(json.loads(marker.read_text(encoding="utf-8")))


def record_opening(
    marker: Path, commit: str, *, reopen_because: str | None = None
) -> list[dict[str, Any]]:
    """Record that the test split is being opened, before its labels are read.

    The first opening needs nothing. A later one is refused unless it gives its reason, which is
    kept with it (a re-run after a crash, with no change to anything the freeze fixed, say): the
    protocol asks for one opening, so a second one must be visible rather than prevented.

    Raises:
        NotFrozen: for a second opening without ``reopen_because``.
    """
    openings = previous_openings(marker)
    if openings and not reopen_because:
        first = openings[0]
        raise NotFrozen(
            f"the test split was already opened at {first['at']} (commit {first['commit'][:8]}); "
            "a second opening has to give its reason (--reopen-because) and is disclosed"
        )
    openings.append(
        {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": commit,
            "reopen_because": reopen_because,
        }
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(openings, indent=1), encoding="utf-8")
    return openings
