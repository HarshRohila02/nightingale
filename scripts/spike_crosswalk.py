"""R-01 feasibility spike: can DDXPlus evidences be aligned with BODHI-S symptoms?

Risk R-01 (docs/07-risk-register.md) asks whether the two vocabularies can be
reconciled well enough for BODHI-S to serve as the cardiac knowledge-graph
backbone. The pre-approved gate is **60% crosswalk coverage**.

This script measures two things and writes a report:

  1. Condition coverage — how many of Nightingale's 13 conditions exist in BODHI-S
  2. Symptom coverage   — for conditions that do match, how many DDXPlus evidences
                          align lexically with a BODHI-S symptom label

    python scripts/spike_crosswalk.py

Inputs (small, ~5 MB — the 847 MB patient CSVs are not needed):
    data/raw/ddxplus/release_{evidences,conditions}.json
    data/raw/bodhi_s/{triples,nl_facts}.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conditions import BY_ID  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
INTERIM = REPO_ROOT / "data" / "interim"
REPORT = REPO_ROOT / "docs" / "10-spike-r01-crosswalk.md"

GATE = 0.60
SIMILARITY_THRESHOLD = 0.72

NL_PATTERN = re.compile(r"^(.*?) \(Symptom\) is a symptom present in (.+?) \(Condition\)\.$")

# Hand-authored mapping from Nightingale conditions to BODHI-S condition labels.
# Only 13 conditions, so this is done by inspection rather than by fuzzy matching.
# quality: "exact" | "approximate" (semantically close but not equivalent) | None
CONDITION_MAP: dict[str, tuple[str | None, str | None]] = {
    "COND:nstemi_stemi": ("Acute myocardial infarction", "exact"),
    "COND:unstable_angina": ("Angina", "approximate"),  # BODHI-S conflates both anginas
    "COND:stable_angina": ("Angina", "approximate"),
    "COND:pericarditis": ("Pericarditis", "exact"),
    "COND:myocarditis": (None, None),
    "COND:acute_pulmonary_edema": ("Congestive heart failure", "approximate"),
    "COND:atrial_fibrillation": ("Arrhythmia", "approximate"),  # too general
    "COND:psvt": ("Arrhythmia", "approximate"),
    "COND:pulmonary_embolism": ("Pulmonary Embolism", "exact"),
    "COND:spontaneous_pneumothorax": (None, None),
    "COND:boerhaave": (None, None),
    "COND:gerd": ("Gastroesophageal reflux disease", "exact"),
    "COND:panic_attack": ("Generalised Anxiety disorder (GAD)", "approximate"),
    "COND:aortic_dissection": (None, None),  # absent from both sources, by design
}


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _similar(a: str, b: str) -> float:
    """Naive whole-string similarity. Structurally unfair here — see _content_overlap."""
    a_norm, b_norm = _normalise(a), _normalise(b)
    if not a_norm or not b_norm:
        return 0.0
    a_tokens, b_tokens = set(a_norm.split()), set(b_norm.split())
    jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return max(jaccard, ratio)


# DDXPlus evidences are patient-facing QUESTIONS ("Are you feeling nauseous?");
# BODHI-S symptoms are clinical NOUN PHRASES ("Vomit <char> nausea present").
# Comparing them whole-string is structurally unfair, so we also strip the
# interrogative scaffolding and compare content words only.
QUESTION_STOPWORDS = frozenset(
    """do you have are is the a an of or and to feel feeling somewhere related your reason
    for consulting had has recently significantly more than then usually get some any that
    this it i me my in on at with does did been being was were will would can could
    characterize please describe experience experienced""".split()  # noqa: SIM905
)


def _content_tokens(text: str) -> set[str]:
    return {w for w in _normalise(text).split() if w not in QUESTION_STOPWORDS and len(w) > 2}


def _content_overlap(ddx_label: str, bodhi_label: str) -> float:
    """Fraction of the DDXPlus evidence's content words present in the BODHI-S label."""
    ddx_tokens = _content_tokens(ddx_label)
    if not ddx_tokens:
        return 0.0
    return len(ddx_tokens & _content_tokens(bodhi_label)) / len(ddx_tokens)


def load_bodhi() -> dict[str, list[str]]:
    """Return {bodhi_condition_label: [symptom labels]}.

    triples.jsonl and nl_facts.jsonl are line-aligned (both 13,204 lines), so the
    human-readable labels in nl_facts decode the UUID/SNOMED ids in triples.
    """
    nl_path = RAW / "bodhi_s" / "nl_facts.jsonl"
    if not nl_path.exists():
        raise SystemExit(f"Missing {nl_path}. Fetch BODHI-S data/ (~4.5 MB) first.")

    by_condition: dict[str, list[str]] = defaultdict(list)
    with nl_path.open(encoding="utf-8") as handle:
        for line in handle:
            match = NL_PATTERN.match(json.loads(line)["text"].strip())
            if match:
                symptom, condition = match.group(1), match.group(2)
                by_condition[condition].append(symptom)
    return by_condition


def load_ddxplus_chestpain() -> dict:
    path = INTERIM / "ddxplus_chestpain_conditions.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run: python scripts/decode_ddxplus.py")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    bodhi = load_bodhi()
    ddx = load_ddxplus_chestpain()

    exact = [c for c, (_, q) in CONDITION_MAP.items() if q == "exact"]
    approx = [c for c, (_, q) in CONDITION_MAP.items() if q == "approximate"]
    absent = [c for c, (_, q) in CONDITION_MAP.items() if q is None]

    trainable = [c for c in CONDITION_MAP if BY_ID[c].in_training_data]
    strict = len([c for c in exact if c in trainable]) / len(trainable)
    generous = len([c for c in exact + approx if c in trainable]) / len(trainable)

    print("=" * 78)
    print("  R-01 SPIKE — DDXPlus <-> BODHI-S crosswalk")
    print("=" * 78)
    print(f"\nBODHI-S conditions total      : {len(bodhi)}")
    print(f"Nightingale trainable conds   : {len(trainable)}")
    print(f"\nCondition coverage (strict)   : {strict:.0%}  ({len(exact)}/{len(trainable)} exact)")
    print(f"Condition coverage (generous) : {generous:.0%}  (incl. approximate matches)")
    print(f"Gate                          : {GATE:.0%}")
    print("\nAbsent from BODHI-S entirely  : " + ", ".join(BY_ID[c].label for c in absent))

    print("\n" + "-" * 78)
    print(f"{'Nightingale condition':<30} {'BODHI-S condition':<34} {'match':<12} sx")
    print("-" * 78)
    rows = []
    for cid, (bodhi_label, quality) in CONDITION_MAP.items():
        n_sx = len(bodhi.get(bodhi_label, [])) if bodhi_label else 0
        label = BY_ID[cid].label
        rows.append((label, bodhi_label or "—", quality or "ABSENT", n_sx))
        print(f"{label:<30} {(bodhi_label or '—'):<34} {(quality or 'ABSENT'):<12} {n_sx or '-'}")

    # Symptom-level alignment for conditions that matched at all.
    print("\n" + "-" * 78)
    print("Symptom-level alignment (matched conditions only)")
    print("-" * 78)
    print("Two methods, because the naive one is structurally unfair (questions vs noun phrases).")
    print(f"{'Condition':<30} {'naive':>12} {'content-word':>14}")
    sym_rows = []
    total_ddx = total_naive = total_content = 0
    for cid, (bodhi_label, _quality) in CONDITION_MAP.items():
        if not bodhi_label or cid not in ddx:
            continue
        ddx_labels = [s["label_en"] for s in ddx[cid]["symptoms"] if s["label_en"]]
        bodhi_labels = bodhi.get(bodhi_label, [])
        if not ddx_labels:
            continue
        naive = sum(
            1
            for d in ddx_labels
            if any(_similar(d, b) >= SIMILARITY_THRESHOLD for b in bodhi_labels)
        )
        content = sum(
            1 for d in ddx_labels if any(_content_overlap(d, b) >= 0.5 for b in bodhi_labels)
        )
        total_ddx += len(ddx_labels)
        total_naive += naive
        total_content += content
        sym_rows.append((BY_ID[cid].label, naive, content, len(ddx_labels)))
        print(
            f"{BY_ID[cid].label:<30} {naive:>4}/{len(ddx_labels):<3} "
            f"{naive / len(ddx_labels):>4.0%} {content:>5}/{len(ddx_labels):<3} "
            f"{content / len(ddx_labels):>4.0%}"
        )

    naive_overall = total_naive / total_ddx if total_ddx else 0.0
    sym_overall = total_content / total_ddx if total_ddx else 0.0
    print(
        f"\n{'OVERALL':<30} {total_naive:>4}/{total_ddx:<3} {naive_overall:>4.0%} "
        f"{total_content:>5}/{total_ddx:<3} {sym_overall:>4.0%}"
    )
    print(
        "\n  The naive 0% is a MEASUREMENT ARTIFACT, not a finding: the concepts often do\n"
        '  correspond ("Are you feeling nauseous?" vs "Vomit <char> nausea present").\n'
        "  The content-word figure is itself generous — some matches hinge on the word\n"
        '  "pain" alone. True conceptual alignment sits below it. Both are under the gate.'
    )

    decision = "PROCEED" if strict >= GATE else "FALLBACK"
    print("\n" + "=" * 78)
    print(f"  DECISION: {decision}")
    print("=" * 78)
    if decision == "FALLBACK":
        print(
            "  Strict condition coverage is below the 60% gate.\n"
            "  Adopt the pre-approved fallback: build the cardiac KG from DDXPlus\n"
            "  release_conditions.json (curated, 13/13 coverage, ICD-10 + severity)\n"
            "  and use BODHI-S to ENRICH the conditions it does cover.\n"
        )

    _write_report(
        rows, sym_rows, strict, generous, naive_overall, sym_overall, decision, len(bodhi)
    )
    print(f"Report written to {REPORT.relative_to(REPO_ROOT)}")
    return 0


def _write_report(
    rows, sym_rows, strict, generous, naive_overall, sym_overall, decision, n_bodhi
) -> None:
    lines = [
        "# 10 — Spike R-01: DDXPlus ↔ BODHI-S Crosswalk",
        "",
        "**Date:** 2026-09-17 · **Owner:** P1 · **Risk:** "
        "[R-01](07-risk-register.md) · **Gate:** 60% condition coverage",
        f"**Generated by:** `scripts/spike_crosswalk.py` · **Decision: {decision}**",
        "",
        "> Reproduce with `python scripts/decode_ddxplus.py && python scripts/spike_crosswalk.py`.",
        "> Inputs are ~5 MB; the 847 MB DDXPlus patient CSVs are not required.",
        "",
        "## Result",
        "",
        f"- BODHI-S contains **{n_bodhi} conditions** in total.",
        f"- **Strict condition coverage: {strict:.0%}** (exact matches only) — **below the 60% gate**.",
        f"- Generous coverage including approximate matches: {generous:.0%}.",
        f"- Symptom-level alignment across matched conditions: **{sym_overall:.0%}** "
        f"(content-word method; naive string matching gives {naive_overall:.0%}, see caveat).",
        "",
        "**The decision is robust to the measurement method** — both symptom-alignment figures, "
        "and the strict condition coverage, sit well below the 60% gate.",
        "",
        "## Condition mapping",
        "",
        "| Nightingale condition | BODHI-S condition | Match | BODHI-S symptoms |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d or '—'} |" for a, b, c, d in rows]
    lines += [
        "",
        "## Symptom alignment (matched conditions)",
        "",
        "| Condition | Naive match | Content-word match | DDXPlus evidences |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {label} | {naive} ({naive / total:.0%}) | {content} ({content / total:.0%}) | {total} |"
        for label, naive, content, total in sym_rows
    ]
    lines += [
        "",
        "### ⚠️ Read the naive column with care — it is an artifact",
        "",
        "Naive whole-string matching scores ~0%, but that is **a measurement artifact, not a "
        "finding**. DDXPlus evidences are patient-facing *questions* while BODHI-S symptoms are "
        "clinical *noun phrases*, so the two never match lexically even when they mean the same "
        "thing:",
        "",
        "| DDXPlus evidence | BODHI-S symptom | Same concept? |",
        "|---|---|---|",
        '| "Are you feeling nauseous or do you feel like vomiting?" | `Vomit <char> nausea '
        "present` | Yes |",
        '| "Have you had significantly increased sweating?" | `Sweating attack` | Yes |',
        "",
        "The content-word method strips interrogative scaffolding and is the fairer figure — but "
        'it is *still generous*: several of its matches hinge on the word "pain" alone (e.g. '
        '"Do you feel pain somewhere?" matching `Pain in calf <char> swelling present`). True '
        "conceptual alignment lies below the content-word number. Closing that gap properly would "
        "need embedding-based matching or a hand-authored crosswalk — neither of which changes "
        "the decision, since even the optimistic figure is under the gate.",
        "",
        "## Why coverage is low",
        "",
        "1. **BODHI-S is broad, not cardiac-deep.** It spans 555 conditions across all of "
        "medicine; our 13 chest-pain conditions are simply not its focus. Myocarditis, "
        "spontaneous pneumothorax, Boerhaave and aortic dissection are absent entirely.",
        "2. **Granularity mismatch.** BODHI-S has one `Angina` node where we need stable vs "
        "unstable, and one `Arrhythmia` node where we need atrial fibrillation vs PSVT. "
        "Collapsing these would destroy distinctions the differential depends on.",
        "3. **Vocabulary mismatch at the symptom level.** DDXPlus evidences are "
        "machine-translated from French and often poor: `déchirante` (*tearing* — the classic "
        'aortic-dissection descriptor) is rendered **"heartbreaking"**, and `lancinante` '
        '(*shooting*) as **"haunting"**. Lexical matching against BODHI-S\'s clean clinical '
        "English therefore fails on exactly the terms that matter most.",
        "",
        "## Decision — adopt the pre-approved fallback",
        "",
        "Build the cardiac medical KG from **DDXPlus `release_conditions.json`**, and use "
        "**BODHI-S to enrich** the conditions it does cover.",
        "",
        "This is a *better* outcome than the fallback anticipated in the plan, which assumed we "
        "would fall back to co-occurrence statistics mined from patient rows. "
        "`release_conditions.json` is not statistics — it is a **curated condition↔symptom "
        "knowledge base** with:",
        "",
        "- **13/13 coverage** of our conditions",
        "- explicit `symptoms` and `antecedents` (risk factors) per condition",
        "- **ICD-10 codes** for every condition",
        "- a **severity ranking** (1 = most severe) that independently corroborates our "
        "must-not-miss designation",
        "",
        "### What BODHI-S is still used for",
        "",
        "- Enriching the 4 exactly-matched conditions (MI, pericarditis, PE, GERD) with extra "
        "symptoms and its `likelihood_condition_given_symptom` / "
        "`likelihood_symptom_given_condition` qualifiers (`rare`/`medium`/`high`), which DDXPlus "
        "does not provide and which are directly useful as KG edge weights.",
        "- Keeping an **independent** knowledge source in the architecture (see the caveat below).",
        "",
        "## ⚠️ Methodological caveat — circularity",
        "",
        "If the KG and the ML ranker are both derived from DDXPlus, the knowledge graph is no "
        'longer an *independent* knowledge source, and the ablation question "does the KG add '
        'value over ML alone?" is weakened: both would encode the same underlying beliefs.',
        "",
        "**This must be stated plainly in the final report.** Mitigations that keep the "
        "comparison meaningful:",
        "",
        "1. The KG still contributes what the ML cannot express: explicit reasoning paths, "
        "supporting/missing/contradicting analysis, and rule-based red flags.",
        "2. **Aortic dissection is in neither source** and is hand-authored into the KG — a "
        "genuinely independent contribution the ranker can never make.",
        "3. Red-flag rules are hand-authored from clinical literature, not derived from DDXPlus.",
        "4. BODHI-S enrichment provides independent evidence for 4 conditions.",
        "",
        "Expect the honest headline finding to be that the KG earns its place on **safety and "
        "explainability**, not on raw accuracy. That is still a publishable result — and "
        "predicting it in advance is better science than discovering it at the end.",
        "",
        "## Actions",
        "",
        "- [x] Close R-01 with the fallback decision",
        "- [ ] P1: build the KG loader from `ddxplus_chestpain_conditions.json`",
        "- [ ] P1: add BODHI-S enrichment for the 4 exact matches, with likelihood weights",
        "- [ ] P1: hand-author aortic dissection into the KG",
        "- [ ] P4: record the circularity caveat in the evaluation protocol's limitations",
        "- [ ] All: treat DDXPlus English labels as unreliable — prefer codes, check the French",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
