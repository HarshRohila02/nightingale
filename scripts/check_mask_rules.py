"""Compare the two candidate reduced-evidence mask rules for `docs/05` amendment 3 (decision D-10).

Mask properties only: this reads the validate parquet and the release file, and it scores no
model and trains nothing. The team is asked to choose a rule *before* any model is scored under
it (docs/proposals/05-amendment-3.md §3), so this script must never grow a model.

The two rules, keyed exactly as the proposal states them:

* **token** — EXP-017's diagnostic rule. Keep the initial evidence's tokens; of the other tokens
  ``R``, keep ``round(len(R) * share)`` chosen by ``rng.choice(len(R), n, replace=False)``, a
  fresh generator per level, in their original order.
* **question** — the proposal's §3.7. Keep the initial evidence's question with all its tokens;
  of the other distinct questions ``Q`` (first-appearance order), keep the first
  ``round(len(Q) * share)`` of one ``rng.permutation(len(Q))``, so the levels nest; a kept
  follow-up brings its parent question (``E_54``-``E_59`` follow ``E_53``, ``E_152`` follows
  ``E_151``) when the patient has it; every token of a kept question stays, in original order.

Both use ``numpy.random.default_rng(int.from_bytes(sha256(f"42:{case_id}").digest()[:8], "big"))``.

It reports, per rule and level: mean tokens and questions kept; patients left with a **positive**
follow-up answer (``is_positive``: not the default "no" / "nowhere", not NA) whose positive parent
question was dropped, which no full-evidence patient has; patients keeping only part of a
multi-choice answer; whether each patient's 25% history lies inside their 50% history; and the
digest the proposal specifies (SHA-256 over ``case_id<TAB>level<TAB>kept tokens``, sorted).

    python scripts/check_mask_rules.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ddxplus import is_positive, load_evidence_specs, parse_token  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LEVELS = (0.5, 0.25)
PARENT = {f"E_{n}": "E_53" for n in range(54, 60)} | {"E_152": "E_151"}
"""Follow-up question -> the question it follows (``code_question`` in the release file)."""


def _rng(case_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{case_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _question(token: str) -> str:
    return token.partition("_@_")[0]


def token_rule(case_id: str, initial: str, tokens: list[str], share: float) -> list[str]:
    fixed = [i for i, t in enumerate(tokens) if _question(t) == initial]
    rest = [i for i in range(len(tokens)) if i not in fixed]
    n = round(len(rest) * share)
    picked = _rng(case_id).choice(len(rest), n, replace=False) if n else []
    kept = set(fixed) | {rest[j] for j in picked}
    return [tokens[i] for i in sorted(kept)]


def question_rule(case_id: str, initial: str, tokens: list[str], share: float) -> list[str]:
    questions = list(dict.fromkeys(_question(t) for t in tokens))
    others = [q for q in questions if q != initial]
    order = _rng(case_id).permutation(len(others))
    kept = {initial} | {others[i] for i in order[: round(len(others) * share)]}
    has = set(questions)
    kept |= {PARENT[q] for q in list(kept) if q in PARENT and PARENT[q] in has}
    return [t for t in tokens if _question(t) in kept]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interim", type=Path, default=REPO_ROOT / "data" / "interim")
    parser.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw" / "ddxplus")
    args = parser.parse_args(argv)

    specs = load_evidence_specs(args.raw / "release_evidences.json")
    frame = pd.read_parquet(
        args.interim / "ddxplus_chestpain_validate.parquet",
        columns=["case_id", "initial_evidence", "evidences"],
    )
    rows = [
        (c, i, list(t))
        for c, i, t in zip(frame.case_id, frame.initial_evidence, frame.evidences, strict=True)
    ]
    n = len(rows)
    print(f"validate patients: {n}")
    print(
        f"full evidence: {np.mean([len(t) for *_, t in rows]):.2f} tokens, "
        f"{np.mean([len({_question(x) for x in t}) for *_, t in rows]):.2f} questions"
    )

    def positive_questions(tokens: list[str]) -> set[str]:
        return {_question(t) for t in tokens if is_positive(parse_token(t), specs)}

    def orphaned(full: list[str], kept: list[str]) -> bool:
        had, has = positive_questions(full), positive_questions(kept)
        return any(q in PARENT and PARENT[q] in had and PARENT[q] not in has for q in has)

    def partial_multi(full: list[str], kept: list[str]) -> bool:
        a, b = Counter(map(_question, full)), Counter(map(_question, kept))
        return any(specs[q].data_type == "M" and 0 < b[q] < a[q] for q in b)

    print(
        f"orphaned positive follow-ups at full evidence: "
        f"{sum(orphaned(t, t) for *_, t in rows)}"
    )
    for name, rule in (("token (EXP-017)", token_rule), ("question (proposed)", question_rule)):
        masks = {s: [rule(c, i, t, s) for c, i, t in rows] for s in LEVELS}
        print(f"\n{name}")
        for s in LEVELS:
            kept = masks[s]
            o = sum(orphaned(t, k) for (*_, t), k in zip(rows, kept, strict=True))
            p = sum(partial_multi(t, k) for (*_, t), k in zip(rows, kept, strict=True))
            print(
                f"  {s:.0%}: {np.mean([len(k) for k in kept]):.2f} tokens, "
                f"{np.mean([len({_question(x) for x in k}) for k in kept]):.2f} questions; "
                f"orphaned positive follow-up {o} ({o / n:.1%}); "
                f"partly kept multi-choice {p} ({p / n:.1%})"
            )
        nested = sum(set(a) <= set(b) for a, b in zip(masks[0.25], masks[0.5], strict=True))
        print(f"  25% history inside the 50% history: {nested} ({nested / n:.1%})")
        lines = sorted(
            f"{c}\t{s}\t{' '.join(k)}"
            for s in LEVELS
            for (c, _, _), k in zip(rows, masks[s], strict=True)
        )
        digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        print(f"  digest (validate, levels 0.5 and 0.25): {digest[:16]}  numpy {np.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
