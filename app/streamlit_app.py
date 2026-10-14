"""Nightingale's dashboard (task 2e; FR-9): enter a chest-pain case, see the ranked differential.

    streamlit run app/streamlit_app.py
    NIGHTINGALE_GRAPH=networkx streamlit run app/streamlit_app.py   # the local graph, not Aura

It serves on 127.0.0.1 only (``.streamlit/config.toml``): a prototype, never exposed beyond this
machine.

**Decision support only, never a diagnosis.** The disclaimer is shown on every screen (FR-6.4).

The pipeline is the demo's (``scripts/_config.py``'s ``build_pipeline``: the configured ranker and
graph, the template explainer, the configured fusion weights), built once per session. **It
re-ranks on every change to the form**: the whole ranking takes about 1–2 ms on the laptop CPU
(EXP-022), so the differential follows the history as it is taken (D-14, "real time"). What each
row says is decided in ``app/view.py``, which CI tests without Streamlit.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _config import build_pipeline, load_config  # noqa: E402

from app.view import (  # noqa: E402
    build_case,
    candidate_rows,
    concept_options,
    evidence_markdown,
    evidence_note,
    evidence_rows,
    example_cases,
    model_reasons,
)
from src.contracts import DISCLAIMER  # noqa: E402
from src.medical_kg.crosswalk import expand_case  # noqa: E402
from src.medical_kg.neo4j_store import GraphUnavailable  # noqa: E402

MISSING_SHOWN = 8
DETAILED = 5

st.set_page_config(page_title="Nightingale", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading the knowledge graph and the model…")
def load() -> dict:
    return build_pipeline(load_config(), os.environ.get("NIGHTINGALE_GRAPH") or None)


def show_evidence(rows: list | None, note: str | None, retrieved: bool) -> None:
    """The candidate's evidence (task 3a, FR-9.4): its excerpts with their citations, or one
    quiet line saying why there are none. Shown beside the ranking, never used to make it."""
    st.markdown("**Evidence**")
    if note:
        st.caption(note)
    elif not retrieved:
        st.caption("Evidence is retrieved for the first candidates only.")
    elif not rows:
        st.caption("No screened passage in the corpus names this condition.")
    else:
        for row in rows:
            st.markdown(evidence_markdown(row))
        st.caption(
            "Excerpts from PMC Open Access articles, chosen for this condition and these "
            "findings; about diagnosis only, so passages about treatment are never shown."
        )


def apply_example() -> None:
    """Fill the form from the chosen golden case."""
    choice = st.session_state.get("example")
    if not choice or choice == "—":
        return
    example = example_cases()[choice]
    st.session_state.update(
        age=example["age"],
        sex=example["sex"],
        findings_present=[c for c in example["present"] if not c.startswith("RF:")],
        findings_denied=[c for c in example["denied"] if not c.startswith("RF:")],
        risks_present=[c for c in example["present"] if c.startswith("RF:")],
        risks_denied=[c for c in example["denied"] if c.startswith("RF:")],
    )


st.title("Nightingale")
st.caption("Clinical decision support for acute chest pain: 14 possible causes, ranked.")
st.warning(DISCLAIMER, icon="⚕️")

try:
    built = load()
except GraphUnavailable as exc:
    st.error(f"No knowledge graph: {exc}. Build it with scripts/decode_ddxplus.py.")
    st.stop()

options = concept_options()
labels = {cid: label for kind in options.values() for cid, label in kind}
examples = example_cases()
FORM = {
    "age": 55,
    "sex": "M",
    "findings_present": [],
    "findings_denied": [],
    "risks_present": [],
    "risks_denied": [],
}
for key, default in FORM.items():  # seeded once, so a widget never gets two defaults
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("The patient")
    st.selectbox(
        "Start from an example",
        ["—", *examples],
        key="example",
        on_change=apply_example,
        format_func=lambda c: c if c == "—" else f"{c}: {examples[c]['description'][:60]}…",
    )
    st.number_input("Age", min_value=0, max_value=120, step=1, key="age")
    st.radio("Sex", ["M", "F"], horizontal=True, key="sex")
    st.caption(
        "Choose each finding as present or denied. Anything not chosen counts as **not asked**, which is no evidence either way."
    )
    finding_ids = [cid for cid, _ in options["finding"]]
    risk_ids = [cid for cid, _ in options["risk_factor"]]
    st.multiselect("Findings present", finding_ids, format_func=labels.get, key="findings_present")
    st.multiselect("Findings denied", finding_ids, format_func=labels.get, key="findings_denied")
    st.multiselect("Risk factors present", risk_ids, format_func=labels.get, key="risks_present")
    st.multiselect("Risk factors denied", risk_ids, format_func=labels.get, key="risks_denied")

try:
    case = build_case(
        int(st.session_state["age"]),
        st.session_state["sex"],
        [*st.session_state["findings_present"], *st.session_state["risks_present"]],
        [*st.session_state["findings_denied"], *st.session_state["risks_denied"]],
    )
except ValueError as exc:
    st.error(f"Please fix the form: {exc}.")
    st.stop()

started = time.perf_counter()
ranked = expand_case(case, built["labels"]) if built["labels"] is not None else case
result = built["pipeline"].run(ranked)
elapsed = (time.perf_counter() - started) * 1000
rows = candidate_rows(result, ranked, model_reasons(built["ranker"], ranked))
evidence = evidence_rows(result, built.get("retriever"))

st.caption(
    f"{built['about']['ranker']} · {built['about']['graph']} · "
    f"{built['about'].get('retriever', 'retrieval: none')} · ranked in {elapsed:.1f} ms; "
    "the list re-ranks as you change the findings."
)
if result.degraded_components:
    st.info("Running without: " + ", ".join(result.degraded_components) + " (docs/02 §7).")
if not case.findings and not case.risk_factors:
    st.info("No findings yet: the ranking below uses age and sex only.")

for flag in result.red_flags:
    st.error(f"Red flag: {flag}", icon="⚠️")

st.subheader("Differential")
table = pd.DataFrame(
    [
        {
            "#": r["rank"],
            "Condition": r["condition"],
            "Red flag": "⚠️" if r["red_flag"] else "",
            "Must not miss": "●" if r["must_not_miss"] else "",
            "Model's rank": "—" if r["model_rank"] is None else str(r["model_rank"]),
            "Graph's rank": "—" if r["graph_rank"] is None else str(r["graph_rank"]),
            "✓": len(r["supporting"]),
            "✗": len(r["against"]),
            "?": len(r["missing"]),
        }
        for r in rows
    ]
)
st.dataframe(table, hide_index=True)
st.caption(
    "Red-flagged conditions come first, then the fused score of the model and the knowledge "
    "graph. Ranks, not raw scores: the scores are not calibrated probabilities. "
    "✓ supporting · ✗ against · ? expected but not recorded. "
    "Aortic dissection is not in the training data, so the graph and the rules rank it."
)

for r in rows[:DETAILED]:
    title = f"{r['rank']}. {r['condition']}" + ("  ⚠️" if r["red_flag"] else "")
    with st.expander(title, expanded=r["rank"] == 1):
        if r["red_flag_reason"]:
            st.markdown(f"**Red flag:** {r['red_flag_reason']}")
        if r["model_note"]:
            st.caption(f"Model: {r['model_note']}.")
        left, right = st.columns(2)
        with left:
            st.markdown("**✓ Supporting**")
            st.markdown("\n".join(f"- {s}" for s in r["supporting"]) or "_none recorded_")
            st.markdown("**✗ Against**")
            st.markdown("\n".join(f"- {s}" for s in r["against"]) or "_none recorded_")
        with right:
            st.markdown("**? Expected, not recorded**")
            shown = r["missing"][:MISSING_SHOWN]
            more = len(r["missing"]) - len(shown)
            st.markdown(
                "\n".join(f"- {s}" for s in shown) + (f"\n- …and {more} more" if more > 0 else "")
                or "_nothing missing_"
            )
        if r["model_for"] or r["model_against"]:
            st.markdown(
                "**The model's reasons**: "
                + (("for: " + ", ".join(r["model_for"])) if r["model_for"] else "")
                + ("; " if r["model_for"] and r["model_against"] else "")
                + (("against: " + ", ".join(r["model_against"])) if r["model_against"] else "")
            )
            st.caption(
                "Each recorded finding removed alone, strongest first; direction and order only. "
                "Findings that matter only together can each look weak."
            )
        if r["paths"]:
            st.markdown("**The graph's reasoning**")
            st.markdown("\n".join(f"- {p}" for p in r["paths"][:MISSING_SHOWN]))
        show_evidence(
            evidence.get(r["condition_id"]),
            evidence_note(result),
            r["rank"] <= built["pipeline"].top_k_evidence,
        )

if result.explanation:
    st.subheader("Explanation")
    st.write(result.explanation.text)

st.divider()
st.caption(
    "The system knows 14 conditions only: any other cause is forced into one of them (R-13). "
    "Trained on synthetic DDXPlus patients. " + DISCLAIMER
)
