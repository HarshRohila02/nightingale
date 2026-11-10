"""The "Patient history (Synthea, display only)" panel (task 1d): its builder
(scripts/build_patient_display.py), its view functions (app/view.py) and the page itself
(app/streamlit_app.py).

The decision of 2026-10-20 (PROGRESS §6) is that a Synthea patient's history is shown beside a case
and never feeds the ranking, the case's findings or its risk factors. Most of this file is that
decision as tests: the cards hold nothing the page could rank on, the panel's code cannot reach the
case (checked in the source), and the page ranks the same whether a patient is chosen or not (checked
by driving it, where Streamlit and the real data are installed). CI has no ``data/``, so everything
else runs on the invented five-patient fixture of tests/test_patient_kg.py.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from app.view import (
    MEDICATION_HEADING,
    PATIENT_NOTE,
    load_patient_cards,
    patient_card_markdown,
    patient_label,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_patient_display as builder  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthea_mini"
SCRIPT = REPO_ROOT / "scripts" / "build_patient_display.py"
APP = REPO_ROOT / "app" / "streamlit_app.py"
AS_OF = date(2026, 1, 1)
REAL = [
    REPO_ROOT / "data" / "raw" / "ddxplus" / "release_evidences.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_chestpain_conditions.json",
    REPO_ROOT / "data" / "interim" / "ddxplus_evidences.json",
    REPO_ROOT / "models" / "nightingale_b1_dl" / "b1_dl_asked_aug" / "seed42" / "b1_dl.json",
]
CARD_KEYS = {
    "id",
    "age",
    "sex",
    "as_of",
    "conditions",
    "conditions_total",
    "risk_factors",
    "medications",
    "medications_total",
    "procedures",
    "observations",
    "unmapped",
    "unmapped_total",
}

UUIDS = {
    "p-0001": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
    "p-0002": "9c858901-8a57-4791-81fe-4c455b099bc9",
    "p-0003": "ec7f5b4e-6d3c-4f1a-a1b2-7f6a2d5e9c10",
    "p-0004": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
    "p-0005": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
}


def uuid_fixture(dst: Path) -> Path:
    """The fixture with Synthea-style UUIDs for its ids, so a test can see what the builder keeps."""
    dst.mkdir(parents=True, exist_ok=True)
    for source in FIXTURE.glob("*.csv"):
        text = source.read_text(encoding="utf-8")
        for short, full in UUIDS.items():
            text = text.replace(short, full)
        (dst / source.name).write_text(text, encoding="utf-8")
    return dst


def write_csv(folder: Path, name: str, header: str, rows: list[str]) -> None:
    (folder / name).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def crowded_folder(tmp_path: Path) -> Path:
    """One patient with more active conditions and medications than a card holds."""
    folder = tmp_path / "crowded"
    folder.mkdir()
    write_csv(folder, "patients.csv", "Id,BIRTHDATE,DEATHDATE,GENDER", ["p-9,1970-01-01,,F"])
    conditions = [
        f"2000-01-{i:02d},,p-9,{9000000 + i},Condition {i:02d} (disorder)" for i in range(1, 21)
    ]
    conditions.append(f"1999-12-31,,p-9,9000099,{'Very long description ' * 20}")
    write_csv(folder, "conditions.csv", "START,STOP,PATIENT,CODE,DESCRIPTION", conditions)
    medications = [f"2010-01-{i:02d},,p-9,{800000 + i},Drug {i:02d} 10 MG" for i in range(1, 16)]
    medications.append("2012-02-02,,p-9,800001,Drug 01 10 MG")  # the same drug again
    write_csv(folder, "medications.csv", "START,STOP,PATIENT,CODE,DESCRIPTION", medications)
    return folder


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    out = tmp_path_factory.mktemp("cards")
    builder.build(FIXTURE, AS_OF, out)
    return {c["id"]: c for c in load_patient_cards(out / "patients.json")}


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #


def test_the_builder_writes_one_card_per_living_patient(tmp_path):
    counts = builder.build(FIXTURE, AS_OF, tmp_path)
    assert (counts["read"], counts["written"]) == (5, 4), "p-0003 died before the day"
    data = json.loads((tmp_path / "patients.json").read_text(encoding="utf-8"))
    assert data["synthea_version"] == "v4.0.0" and data["as_of"] == "2026-01-01"
    assert [c["id"] for c in data["patients"]] == ["p-0001", "p-0002", "p-0004", "p-0005"]
    assert all(set(c) == CARD_KEYS for c in data["patients"])
    assert data["caps"] == builder.CAPS
    assert counts["bytes"] == (tmp_path / "patients.json").stat().st_size
    # An earlier day: p-0003 was alive then.
    earlier = builder.build(FIXTURE, date(2020, 1, 1), tmp_path / "earlier")
    assert earlier["written"] == 5


def test_a_card_holds_the_history_as_of_the_day(cards):
    one = cards["p-0001"]
    assert (one["age"], one["sex"], one["as_of"]) == (65, "M", "2026-01-01")
    assert [(c["description"], c["since"]) for c in one["conditions"]] == [
        ("Essential hypertension (disorder)", "2010-01-01"),
        ("Diabetes mellitus type 2 (disorder)", "2015-03-03"),
        ("Hypertriglyceridemia (disorder)", "2016-04-04"),
    ], "active ones, oldest first; prediabetes ended and heart failure starts after the day"
    assert one["conditions_total"] == 3
    assert {(r["label"], r["present"]) for r in one["risk_factors"]} == {
        ("Hypertension", True),
        ("Diabetes mellitus", True),
        ("Smoking", False),
    }
    assert next(r for r in one["risk_factors"] if r["label"] == "Smoking")["since"] == "2023-01-01"
    assert (one["procedures"], one["observations"]) == (1, 4), "the 2027 observation is not yet"
    assert one["unmapped"] == ["Hypertriglyceridemia (disorder)"]
    assert cards["p-0005"]["unmapped"] == ["Acute bronchitis (disorder)"]
    assert cards["p-0005"]["risk_factors"] == []
    assert cards["p-0002"]["sex"] == "F" and cards["p-0002"]["age"] == 35


def test_medications_are_descriptions_only(cards, tmp_path):
    one = cards["p-0001"]
    assert one["medications"] == ["lisinopril 10 MG Oral Tablet"] and one["medications_total"] == 1
    assert cards["p-0002"]["medications"] == []
    builder.build(FIXTURE, AS_OF, tmp_path)
    text = (tmp_path / "patients.json").read_text(encoding="utf-8")
    for not_kept in ("314076", "REASON", "BASE_COST", "PAYER"):
        assert not_kept not in text, not_kept
    assert '"code"' not in text, "no code reaches a card"
    for card in json.loads(text)["patients"]:
        assert all(isinstance(m, str) for m in card["medications"])


def test_a_card_keeps_eight_characters_of_the_uuid_and_nothing_else_of_it(tmp_path):
    folder = uuid_fixture(tmp_path / "uuid")
    out = tmp_path / "out"
    builder.build(folder, AS_OF, out)
    text = (out / "patients.json").read_text(encoding="utf-8")
    kept = json.loads(text)["patients"]
    living = [UUIDS[p] for p in ("p-0001", "p-0002", "p-0004", "p-0005")]
    assert [c["id"] for c in kept] == sorted(full[:8] for full in living)
    assert all(len(c["id"]) == 8 for c in kept)
    for full in living:
        assert full[:8] in text
        for beyond in (full[:9], full[8:], full[9:], full):
            assert beyond not in text, full
    assert UUIDS["p-0003"][:8] not in text, "the patient who died has no card"


def test_lists_are_capped_with_their_full_length_beside_them(tmp_path):
    folder = crowded_folder(tmp_path)
    out = tmp_path / "out"
    counts = builder.build(folder, AS_OF, out)
    (card,) = json.loads((out / "patients.json").read_text(encoding="utf-8"))["patients"]
    assert len(card["conditions"]) == builder.CAPS["conditions"] == 12
    assert card["conditions_total"] == 21
    assert [c["since"] for c in card["conditions"]] == [
        "1999-12-31",
        *[f"2000-01-{i:02d}" for i in range(1, 12)],
    ], "oldest first, and the cut keeps the oldest"
    assert len(card["conditions"][0]["description"]) <= builder.TEXT_LIMIT
    assert card["conditions"][0]["description"].endswith("…"), "clipped, and says so"
    assert len(card["medications"]) == builder.CAPS["medications"] == 10
    assert card["medications_total"] == 15, "the repeated drug counts once"
    assert len(card["unmapped"]) == builder.CAPS["unmapped"] == 12
    assert card["unmapped_total"] == 21
    assert counts["with_a_list_cut"] == 1


def test_a_card_feeds_nothing_it_is_not_a_case_or_a_finding(cards):
    """The card has no Finding, no concept id and no code: nothing in it can be ranked on."""
    text = json.dumps(cards)
    for token in ("RF:", "SYM:", "COND:", "concept_id", "assertion", "source", "synthea"):
        assert token not in text, token


def test_the_cli_prints_counts_never_patient_rows(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--synthea", str(FIXTURE), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "5 patients read, 4 alive on 2026-01-01" in result.stdout
    for row in ("lisinopril", "hypertension", "p-0001", "bronchitis", "Smoking"):
        assert row.lower() not in result.stdout.lower(), row
    assert (tmp_path / "patients.json").exists()


def test_the_cli_says_why_when_there_is_no_synthea_output(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--synthea", str(tmp_path), "--out", str(tmp_path / "o")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 1 and "no patients.csv" in result.stderr


def test_the_data_folder_is_never_committed():
    assert "data/" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    default = builder.DEFAULT_OUT.relative_to(REPO_ROOT).as_posix()
    assert default.startswith("data/interim/")


# --------------------------------------------------------------------------- #
# The view functions
# --------------------------------------------------------------------------- #


def test_an_absent_or_unreadable_file_gives_none_and_never_raises(tmp_path):
    assert load_patient_cards(tmp_path / "missing.json") is None
    assert load_patient_cards(tmp_path) is None, "a folder"
    for name, content in {
        "empty.json": "",
        "garbage.json": "{not json",
        "list.json": "[1, 2, 3]",
        "no_patients.json": '{"as_of": "2026-01-01"}',
        "wrong_type.json": '{"patients": "many"}',
    }.items():
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        assert load_patient_cards(path) is None, name
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00bad")
    assert load_patient_cards(binary) is None
    assert load_patient_cards(None) is None  # type: ignore[arg-type]


def test_a_valid_file_gives_its_cards_and_drops_what_is_not_one(tmp_path):
    path = tmp_path / "patients.json"
    path.write_text(
        json.dumps({"patients": [{"id": "abc12345"}, 7, {"age": 3}, {"id": ""}, {"id": "z"}]}),
        encoding="utf-8",
    )
    assert [c["id"] for c in load_patient_cards(path)] == ["abc12345", "z"]
    path.write_text(json.dumps({"patients": []}), encoding="utf-8")
    assert load_patient_cards(path) == []


def test_the_builders_file_loads(tmp_path):
    builder.build(FIXTURE, AS_OF, tmp_path)
    loaded = load_patient_cards(tmp_path / "patients.json")
    assert [c["id"] for c in loaded] == ["p-0001", "p-0002", "p-0004", "p-0005"]


def test_the_label_names_the_patient_by_short_id_age_and_sex(cards):
    assert patient_label(cards["p-0001"]) == "p-0001 · 65 y · male"
    assert patient_label(cards["p-0002"]) == "p-0002 · 35 y · female"
    assert patient_label({}) == "unknown · age unknown · sex unknown"
    assert patient_label({"id": "ab*c[d](e)", "age": True, "sex": "x"}).startswith("abcde ·")


def test_the_note_says_it_is_context_only_and_not_advice():
    text = PATIENT_NOTE.lower()
    assert "synthea" in text and "synthetic" in text
    assert "does not enter the ranking" in text and "the findings you tick do" in text
    assert "medications are history, never advice" in text


def test_the_card_markdown_shows_each_section(cards):
    text = patient_card_markdown(cards["p-0001"])
    assert text.startswith("**p\\-0001 · 65 y · male**") and "history as of 2026\\-01\\-01" in text
    for heading in (
        "**Active conditions**",
        "**Risk factors the history supports**",
        f"**{MEDICATION_HEADING}**",
        "**On record by that day**: 1 procedures, 4 observations.",
        "No risk-factor mapping for 1 of 3 active entries.",
    ):
        assert heading in text, heading
    assert MEDICATION_HEADING == "Medication history (not advice)"
    assert "- lisinopril 10 MG Oral Tablet" in text
    assert "- Smoking: absent · since 2023\\-01\\-01" in text
    assert "- Hypertension: present · since 2010\\-01\\-01" in text
    assert "_none recorded_" not in text
    empty = patient_card_markdown(cards["p-0005"])
    assert "_none recorded_" in empty and "_none active_" in empty


def test_the_card_markdown_escapes_what_the_file_holds():
    card = {
        "id": "abc12345",
        "age": 50,
        "sex": "F",
        "conditions": [
            {
                "description": "Cough [click](http://evil.test) *bold* $x$ <b>",
                "since": "2020-01-01",
            },
            {"description": "# heading | cell", "since": "2021-01-01", "finding": True},
        ],
        "medications": ["Drug `code` [a](b)"],
        "risk_factors": [{"label": "_Smoking_", "present": True, "since": None}],
    }
    text = patient_card_markdown(card)
    assert "[click](" not in text and "\\[click\\]\\(http://evil.test\\)" in text
    assert "\\*bold\\*" in text and "\\$x\\$" in text and "<b>" not in text and "\\<b\\>" in text
    assert "`code`" not in text and "\\`code\\`" in text and "[a](b)" not in text
    assert "\n# heading" not in text and "\\# heading \\| cell" in text
    assert "\\_Smoking\\_: present" in text


def test_a_cut_list_says_how_many_more_there_are(tmp_path):
    out = tmp_path / "out"
    builder.build(crowded_folder(tmp_path), AS_OF, out)
    (card,) = load_patient_cards(out / "patients.json")
    text = patient_card_markdown(card)
    assert "- …and 9 more" in text, "conditions: 12 of 21"
    assert "- …and 5 more" in text, "medications: 10 of 15"
    assert text.count("…and") == 2, "the unmapped ones are counted, not listed again"
    assert "No risk-factor mapping for 21 of 21 active entries." in text
    assert len(card["conditions"]) + 9 == card["conditions_total"]


def test_a_malformed_card_is_drawn_not_raised():
    for card in (
        {"id": "x"},
        {"id": "x", "conditions": "none", "medications": 4, "risk_factors": [None, 3]},
        {"id": "x", "conditions": [{"since": "2020"}, "text", None], "procedures": "many"},
        {"id": "x", "conditions": [{"description": "a"}] * 500, "conditions_total": "all"},
    ):
        text = patient_card_markdown(card)
        assert "Active conditions" in text and MEDICATION_HEADING in text
    many = patient_card_markdown({"id": "x", "conditions": [{"description": "a"}] * 500})
    assert many.count("- a") <= 25, "the page never draws more rows than a card should hold"


# --------------------------------------------------------------------------- #
# The page: the panel can neither read nor change the case (the source)
# --------------------------------------------------------------------------- #

PANEL = "render_patient_history"
SOURCE = APP.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def panel_function() -> ast.FunctionDef:
    (func,) = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == PANEL]
    return func


def calls_of(name: str) -> list[ast.Call]:
    """Every call of ``name(...)`` or ``obj.name(...)`` in the page."""
    found = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == name) or (
                isinstance(func, ast.Attribute) and func.attr == name
            ):
                found.append(node)
    return found


def mentions(nodes: list[ast.AST]) -> set[str]:
    """Every name, attribute and string in ``nodes``, lowered: what an expression can reach."""
    seen: set[str] = set()
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                seen.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                seen.add(node.attr.lower())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.add(node.value.lower())
    return seen


def test_the_case_and_the_ranking_are_built_from_the_form_alone():
    consumers = [*calls_of("build_case"), *calls_of("expand_case"), *calls_of("run")]
    assert len(calls_of("build_case")) == 1 and len(calls_of("expand_case")) == 1
    runs = [c for c in calls_of("run") if "pipeline" in mentions([c.func])]
    assert len(runs) == 1, "the page ranks in one place"
    panel = {
        "patient",
        "history",
        PANEL,
        "history_patient",
        "patient_cards",
        "chosen",
        "cards",
        "patient_card_markdown",
        "patient_label",
    }
    for call in consumers:
        reached = mentions([*call.args, *[k.value for k in call.keywords]])
        assert not any(
            word in text for text in reached for word in ("patient", "history", "synthea")
        ), (call.lineno, reached)
        assert not reached & panel, (call.lineno, reached & panel)


def test_the_panel_is_one_function_that_returns_nothing_and_is_called_for_its_drawing():
    func = panel_function()
    assert not [n for n in ast.walk(func) if isinstance(n, ast.Return) and n.value is not None]
    assert not [n for n in ast.walk(func) if isinstance(n, ast.Global | ast.Nonlocal)]
    assert not func.args.args, "it is given nothing"
    calls = [c for c in calls_of(PANEL)]
    assert len(calls) == 1
    statements = [n for n in TREE.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert [s for s in statements if s.value is calls[0]], "a bare statement: the value is dropped"
    assigned = [
        n.value
        for n in ast.walk(TREE)
        if isinstance(n, ast.Assign | ast.AnnAssign | ast.AugAssign) and n.value is not None
    ]
    assert not [v for v in assigned if PANEL in mentions([v])], "its value is never kept"


def test_the_panel_reaches_neither_the_case_nor_the_form():
    func = panel_function()
    segment = ast.get_source_segment(SOURCE, func) or ""
    assert "session_state" not in segment, "the panel reads and writes no session state"
    names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
    shut_out = {
        "case",
        "ranked",
        "result",
        "rows",
        "built",
        "options",
        "labels",
        "examples",
        "FORM",
        "build_case",
        "expand_case",
        "candidate_rows",
        "next_steps",
        "apply_example",
        "load",
    }
    assert not names & shut_out, names & shut_out
    page_calls = {
        node.func.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    }
    assert page_calls <= {"caption", "expander", "selectbox", "markdown"}, page_calls


def test_the_panels_widget_key_is_not_a_form_key():
    form = next(
        n.value
        for n in TREE.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FORM"
    )
    assert isinstance(form, ast.Dict)
    form_keys = {k.value for k in form.keys if isinstance(k, ast.Constant)}
    assert form_keys == {
        "age",
        "sex",
        "findings_present",
        "findings_denied",
        "risks_present",
        "risks_denied",
    }
    assert SOURCE.count('"history_patient"') == 1, "named once, as the selectbox's own key"
    assert "history_patient" not in form_keys


def test_the_page_imports_no_graph_of_the_patient_history():
    for path in (APP, REPO_ROOT / "app" / "view.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            if isinstance(node, ast.Import):
                modules += [a.name for a in node.names]
            assert not [m for m in modules if m.startswith("src.patient_kg")], path.name


def test_the_page_has_no_bare_string_for_streamlit_to_draw():
    """Streamlit draws a string statement at the top level of the script (its "magic"), so a
    docstring-style note after an assignment would show on the page. Only the module's own
    docstring is exempt."""
    bare = [
        n.lineno
        for n in TREE.body[1:]
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
    ]
    assert not bare, f"a string statement at line(s) {bare}: use a comment"


def test_the_panel_says_how_to_build_the_cards_when_there_are_none():
    func = panel_function()
    segment = ast.get_source_segment(SOURCE, func) or ""
    assert "scripts/build_patient_display.py" in segment
    assert "Patient history (Synthea, display only)" in segment
    assert "PATIENT_NOTE" in segment and "patient_card_markdown" in segment


# --------------------------------------------------------------------------- #
# The page: the same ranking with and without a patient (driven headlessly)
# --------------------------------------------------------------------------- #

needs_the_app = pytest.mark.skipif(
    importlib.util.find_spec("streamlit") is None or not all(p.exists() for p in REAL),
    reason="needs Streamlit and data/ and models/ (not in CI)",
)
FORM_KEYS = ("age", "sex", "findings_present", "findings_denied", "risks_present", "risks_denied")


def run_page(monkeypatch, cards_file: Path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NIGHTINGALE_GRAPH", "networkx")  # never reach for Aura in a test
    monkeypatch.setenv("NIGHTINGALE_PATIENT_DISPLAY", str(cards_file))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, app.exception
    return app


def what_the_case_and_ranking_show(app) -> dict:
    """Everything the form fed and the ranking produced, ready to compare."""
    return {
        "form": {key: app.session_state[key] for key in FORM_KEYS},
        "table": app.dataframe[0].value.to_dict("list"),
        "red_flags": [e.value for e in app.error],
        "notices": [i.value for i in app.info],
        "subheaders": [s.value for s in app.subheader],
    }


@needs_the_app
def test_the_ranking_is_the_same_with_and_without_a_patient_chosen(monkeypatch, tmp_path):
    builder.build(FIXTURE, AS_OF, tmp_path)
    app = run_page(monkeypatch, tmp_path / "patients.json")
    assert any("Patient history (Synthea, display only)" in e.label for e in app.expander)

    apart = {}
    for example in ("GC-001", "GC-003"):
        app.selectbox(key="example").select(example).run()
        assert not app.exception, app.exception
        apart[example] = what_the_case_and_ranking_show(app)
        assert apart[example]["table"]["Condition"], "a ranking was shown"
        for index in (1, 2, 4, 0):  # three patients, then none again
            app.selectbox(key="history_patient").select_index(index).run()
            assert not app.exception, app.exception
            assert what_the_case_and_ranking_show(app) == apart[example], (example, index)
            shown = " ".join(m.value for m in app.markdown)
            assert (MEDICATION_HEADING in shown) == (index != 0)
    # A patient who was chosen first changes nothing for a case chosen afterwards.
    fresh = run_page(monkeypatch, tmp_path / "patients.json")
    fresh.selectbox(key="history_patient").select_index(1).run()
    fresh.selectbox(key="example").select("GC-003").run()
    assert not fresh.exception, fresh.exception
    assert what_the_case_and_ranking_show(fresh) == apart["GC-003"]


@needs_the_app
def test_choosing_a_patient_shows_the_card_and_the_note(monkeypatch, tmp_path):
    builder.build(FIXTURE, AS_OF, tmp_path)
    app = run_page(monkeypatch, tmp_path / "patients.json")
    picker = app.selectbox(key="history_patient")
    assert picker.options[0] == "— none selected —"
    assert picker.options[1:] == [
        "p-0001 · 65 y · male",
        "p-0002 · 35 y · female",
        "p-0004 · 70 y · female",
        "p-0005 · 26 y · male",
    ]
    assert not any("lisinopril" in m.value for m in app.markdown), "nothing chosen, no card"
    picker.select_index(1).run()
    shown = [m.value for m in app.markdown]
    assert any("lisinopril 10 MG Oral Tablet" in m and MEDICATION_HEADING in m for m in shown)
    assert any(PATIENT_NOTE in c.value for c in app.caption)


@needs_the_app
def test_without_cards_the_page_says_how_to_build_them_and_is_otherwise_the_same(
    monkeypatch, tmp_path
):
    app = run_page(monkeypatch, tmp_path / "not_built.json")
    assert any("scripts/build_patient_display.py" in c.value for c in app.caption)
    assert not [s for s in app.selectbox if s.key == "history_patient"]
    assert not app.expander or all("Patient history" not in e.label for e in app.expander)
    app.selectbox(key="example").select("GC-003").run()
    assert not app.exception, app.exception
    assert app.dataframe[0].value.iloc[0]["Condition"] == "Aortic dissection"
    # The same ranking as with cards present.
    builder.build(FIXTURE, AS_OF, tmp_path / "built")
    with_cards = run_page(monkeypatch, tmp_path / "built" / "patients.json")
    with_cards.selectbox(key="example").select("GC-003").run()
    assert with_cards.dataframe[0].value.to_dict("list") == app.dataframe[0].value.to_dict("list")


@needs_the_app
def test_an_unreadable_file_is_the_same_as_none(monkeypatch, tmp_path):
    broken = tmp_path / "patients.json"
    broken.write_text("{ this is not json", encoding="utf-8")
    app = run_page(monkeypatch, broken)
    assert any("scripts/build_patient_display.py" in c.value for c in app.caption)
    assert not app.exception


# --------------------------------------------------------------------------- #
# Disorders first, findings apart (2026-11-10: Synthea lists social history as active conditions)
# --------------------------------------------------------------------------- #


def mixed_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "mixed"
    folder.mkdir()
    write_csv(folder, "patients.csv", "Id,BIRTHDATE,DEATHDATE,GENDER", ["p-8,1960-01-01,,M"])
    long_finding = "A finding whose description is much too long to keep whole " * 3 + "(finding)"
    write_csv(
        folder,
        "conditions.csv",
        "START,STOP,PATIENT,CODE,DESCRIPTION",
        [
            "1980-01-01,,p-8,7000001,Full-time employment (finding)",
            "1981-01-01,,p-8,7000002,Received higher education (finding)",
            "2020-05-05,,p-8,7000003,Anemia (disorder)",
            "2021-06-06,,p-8,7000004,Past pregnancy history of miscarriage (situation)",
            f"1979-01-01,,p-8,7000005,{long_finding}",
            "2022-07-07,,p-8,7000006,Acute bronchitis",
        ],
    )
    return folder


def test_disorders_come_first_and_findings_are_flagged_even_when_clipped(tmp_path):
    out = tmp_path / "out"
    builder.build(mixed_folder(tmp_path), AS_OF, out)
    (card,) = load_patient_cards(out / "patients.json")
    names = [c["description"] for c in card["conditions"]]
    assert names[:2] == [
        "Anemia (disorder)",
        "Acute bronchitis",
    ], "disorders, and an untagged one, first"
    flagged = {c["description"][:30]: c.get("finding") for c in card["conditions"]}
    assert flagged["Anemia (disorder)"] is None and flagged["Acute bronchitis"] is None
    assert sum(1 for c in card["conditions"] if c.get("finding") is True) == 4
    clipped = next(c for c in card["conditions"] if c["description"].endswith("…"))
    assert clipped.get("finding") is True, "classified on the full text, before the clip"
    assert [c["since"] for c in card["conditions"][2:]] == sorted(
        c["since"] for c in card["conditions"][2:]
    )


def test_the_card_draws_findings_under_their_own_heading():
    card = {
        "id": "abc12345",
        "age": 50,
        "sex": "M",
        "conditions": [
            {"description": "Anemia (disorder)", "since": "2020-05-05"},
            {
                "description": "Full-time employment (finding)",
                "since": "2025-10-06",
                "finding": True,
            },
        ],
        "conditions_total": 5,
        "unmapped_total": 2,
    }
    text = patient_card_markdown(card)
    head, _, rest = text.partition("**Other findings on record**")
    assert "Anemia" in head and "Full\\-time employment" not in head
    assert (
        "Full\\-time employment" in rest and "- …and 3 more" in rest
    ), "the cut is the findings' end"
    assert "No risk-factor mapping for 2 of 5 active entries." in text
    # No findings: no second heading.
    assert "Other findings" not in patient_card_markdown(
        {"id": "x", "conditions": [{"description": "a"}]}
    )
