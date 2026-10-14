"""Tests for evidence retrieval (task 3a, src/rag/), on a tiny invented corpus.

Every title, passage and PMCID here is made up (tests/rag_fixture.py); nothing calls NCBI or loads
a model: the network is a fake opener and the model a hash of words. What is checked:

* the parts: the tokenizer, BM25 against scores worked out by hand, reciprocal rank fusion's
  arithmetic, the condition-name filter, the excerpt;
* the treatment screen (FR-6.5): what counts as treatment language, the runs of clean sentences,
  the titles, and that nothing with treatment language reaches a result;
* the bundle loader refusing a wrong hash, a missing file, mismatched shapes and a malformed
  manifest, and ``open_retriever`` degrading instead of raising;
* the retriever: one passage per article, denied findings never in a query, the dense scores
  equal to the embeddings' product, the mix's fidelity;
* the pipeline: evidence on the top candidates only, never changing the ranking, ``rag`` reported
  when degraded, citations with a link, the golden cases unchanged;
* the Kaggle build, end to end and offline; the E-utilities client's rate limit and retries;
* the dashboard's evidence rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.conditions import CONDITIONS
from src.contracts import (
    Assertion,
    Candidate,
    DiagnosisResult,
    Evidence,
    Finding,
    PatientCase,
)
from src.pipeline import DiagnosisPipeline
from src.rag.bm25 import BM25Index
from src.rag.build import build_bundle, fetch_articles, golden_checks, plan_searches
from src.rag.bundle import BundleError, load_bundle, write_manifest
from src.rag.corpus import (
    Article,
    article_passages,
    article_url,
    chunk_spans,
    licence_code,
    parse_articles,
)
from src.rag.eutils import MIN_INTERVAL, EUtils, EUtilsError
from src.rag.retriever import (
    HybridRetriever,
    UnavailableRetriever,
    describe,
    mix_fidelity,
    open_retriever,
    present_concepts,
    reciprocal_rank_fusion,
)
from src.rag.screening import (
    diagnostic_spans,
    title_names_treatment,
    treatment_language,
)
from src.rag.text import excerpt, mentions_any, phrase_tokens, tokenize
from src.rag.vocabulary import (
    CONDITION_TERMS,
    compose_query_text,
    esearch_terms,
    query_vocabulary,
    resolve_condition,
)
from src.reasoning.safety import safety_check
from src.stubs import ConstantRanker, EmptyRetriever, InMemoryGraphStore, TemplateExplainer
from tests.rag_fixture import FEVER_CHECK, PASSAGES, axis, mixed, write_fake_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
GOLDEN = yaml.safe_load(
    (REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml").read_text(encoding="utf-8")
)["cases"]


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    return write_fake_bundle(tmp_path / "bundle")


@pytest.fixture
def retriever(bundle_dir: Path) -> HybridRetriever:
    return HybridRetriever(load_bundle(bundle_dir))


def finding(concept: str, assertion: Assertion = Assertion.PRESENT) -> Finding:
    return Finding(concept_id=concept, label=concept.split(":")[1], assertion=assertion)


def case(*findings: Finding, risk_factors: tuple[Finding, ...] = ()) -> PatientCase:
    return PatientCase(
        case_id="T-RAG", age=40, sex="F", findings=list(findings), risk_factors=list(risk_factors)
    )


PERICARDITIS_CASE = case(
    finding("SYM:chest_pain"),
    finding("SYM:pleuritic"),
    finding("SYM:positional_relief_sitting_forward"),
    finding("SYM:recent_viral_illness"),
    finding("SYM:fever"),
)


# --------------------------------------------------------------------------- #
# Text: the tokenizer, mentions, excerpts
# --------------------------------------------------------------------------- #


def test_the_tokenizer_follows_its_stated_rules():
    assert tokenize("Gastro-oesophageal reflux attacks, and LOSS of the pulses") == [
        "gastro",
        "esophageal",
        "reflux",
        "attack",
        "loss",
        "pulse",
    ]
    assert tokenize("Pericarditis diagnosis in type 2 diabetes") == [
        "pericarditis",
        "diagnosis",
        "type",
        "2",
        "diabete",
    ]
    assert tokenize("Ischaemia and oedema") == ["ischemia", "edema"]
    assert tokenize("No pain? Not a word.") == ["pain", "word"]


def test_a_phrase_matches_whole_tokens_only():
    stable = [phrase_tokens("stable angina")]
    assert not mentions_any(tokenize("Unstable angina at rest"), stable)
    assert mentions_any(tokenize("Chronic stable angina on exertion"), stable)
    assert mentions_any(
        tokenize("Dissection of the aorta"), [phrase_tokens("dissection of the aorta")]
    )
    assert not mentions_any(
        tokenize("aorta dissection"), [phrase_tokens("dissection of the aorta")]
    )


def test_the_excerpt_is_the_sentence_with_most_query_terms_verbatim():
    text = (
        "Chest pain has many causes in the emergency room and most are benign. "
        "Pericarditis gives pleuritic pain that eases when sitting forward. "
        "Another sentence follows here."
    )
    out = excerpt(text, tokenize("pericarditis pleuritic"), limit=80)
    assert out.startswith("…Pericarditis gives pleuritic pain")
    assert out.endswith("…")
    assert out.strip("…") in text


# --------------------------------------------------------------------------- #
# BM25 against hand-computed scores; reciprocal rank fusion
# --------------------------------------------------------------------------- #


def test_bm25_matches_scores_worked_out_by_hand():
    """Three passages, k1 = 1.2, b = 0.75, Lucene's idf. avgdl = (2 + 1 + 3) / 3 = 2."""
    index = BM25Index([["chest", "pain"], ["pain"], ["fever", "pain", "pain"]])
    # "fever" is in one passage of three: idf = ln(1 + 2.5 / 1.5) = ln(8/3). The third passage has
    # length 3: its normaliser is 1.2 · (0.25 + 0.75 · 3/2) = 1.65.
    expected = math.log(8 / 3) * 1 * 2.2 / (1 + 1.65)
    assert index.scores("fever") == pytest.approx([0.0, 0.0, expected])
    # "pain" is in all three: idf = ln(1 + 0.5 / 3.5) = ln(8/7). Normalisers 1.2, 0.75 and 1.65;
    # the third passage holds it twice.
    idf = math.log(8 / 7)
    assert index.scores("pain") == pytest.approx(
        [idf * 2.2 / (1 + 1.2), idf * 2.2 / (1 + 0.75), idf * 2 * 2.2 / (2 + 1.65)]
    )
    # A query is a set of terms; an unknown term scores nothing.
    assert index.scores("pain pain unknown") == pytest.approx(index.scores("pain"))
    assert index.scores("") == pytest.approx([0.0, 0.0, 0.0])


def test_bm25_from_texts_tokenizes_like_the_rest():
    index = BM25Index.from_texts(["Pains in the chest", "Fevers"])
    assert index.query_terms("pain FEVER absent") == ["pain", "fever"]
    assert index.doc_freq[index.vocabulary["pain"]] == 1


def test_reciprocal_rank_fusion_arithmetic_and_ties():
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    assert [i for i, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 1]])] == [1, 3, 2]
    # Equal scores: the item ranked higher in an earlier list first.
    assert [i for i, _ in reciprocal_rank_fusion([[7], [5]])] == [7, 5]
    assert reciprocal_rank_fusion([]) == []


# --------------------------------------------------------------------------- #
# The treatment screen (FR-6.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Patients were treated with colchicine.",
        "Ibuprofen 600 mg three times a day.",
        "First-line therapy is a non-steroidal agent.",
        "Metoprolol reduced the rate.",
        "Most patients can be managed as outpatients.",
        "Urgent surgical repair is needed.",
        "Catheter ablation is curative.",
        "Anticoagulation should be started.",
        "Analgesia was given.",
    ],
)
def test_treatment_language_is_recognised(text):
    assert treatment_language(text)


@pytest.mark.parametrize(
    "text",
    [
        "Recent surgery is a strong risk factor for embolism.",
        "A friction rub may be heard.",
        "Cocaine and other drug use can cause chest pain.",
        "The pain radiates to the back.",
        "A ventilation-perfusion scan can confirm the diagnosis.",
        "Symptoms began in April.",
    ],
)
def test_diagnostic_sentences_are_not_treatment_language(text):
    assert not treatment_language(text)


def test_only_runs_of_consecutive_clean_sentences_are_kept():
    a = "Pericarditis gives sharp pleuritic chest pain that eases on sitting forward in most adults."
    b = "It was treated with colchicine for three months."
    c = "A pericardial friction rub is heard in a third of patients at presentation."
    d = "Widespread ST elevation with PR depression on the electrocardiogram supports it."
    text = " ".join([a, b, c, d])
    spans = diagnostic_spans(text, min_chars=50)
    assert [text[s:e] for s, e in spans] == [a, f"{c} {d}"]
    # Never glued across the removed sentence; a run shorter than the floor is dropped.
    assert all(b not in text[s:e] for s, e in spans)
    assert [text[s:e] for s, e in diagnostic_spans(text, min_chars=120)] == [f"{c} {d}"]


def test_a_title_naming_a_treatment_is_refused_but_management_is_not():
    assert title_names_treatment("Colchicine for recurrent pericarditis")
    assert title_names_treatment("Thrombolysis in pulmonary embolism")
    assert not title_names_treatment("Diagnosis and management of acute pericarditis")


def test_no_search_asks_for_treatment_and_every_search_is_open_access():
    for cid in CONDITION_TERMS:
        for _, term in esearch_terms(cid):
            assert term.endswith('"open access"[filter]')
            assert not treatment_language(term), term


def test_the_retriever_hides_passages_with_treatment_in_the_text_or_the_title(retriever):
    assert retriever.screened_out == 2
    shown = set(retriever.eligible("COND:pericarditis").tolist())
    assert shown == {0, 1, 3}  # not 5 (its title names colchicine), not 6 (ibuprofen)
    texts = [
        e.passage + e.title for e in retriever.retrieve("pericarditis", PERICARDITIS_CASE, k=9)
    ]
    assert texts and not any(treatment_language(t) or "Colchicine" in t for t in texts)


def test_safety_check_drops_evidence_that_recommends_a_treatment():
    bad = Evidence(
        source="PMC",
        title="An invented title",
        passage="Give aspirin 300 mg.",
        citation="PMCID:PMC99990009",
    )
    good = Evidence(
        source="PMC",
        title="An invented title",
        passage="Pain eases on sitting forward.",
        citation="PMCID:PMC99990008",
    )
    titled = Evidence(
        source="PMC",
        title="Heparin in an invented trial",
        passage="Chest pain.",
        citation="PMCID:PMC99990007",
    )
    result = DiagnosisResult(
        case_id="T",
        candidates=[
            Candidate(
                condition_id="COND:pericarditis", label="Pericarditis", evidence=[bad, good, titled]
            )
        ],
    )
    checked = safety_check(result)
    assert [e.citation for e in checked.candidates[0].evidence] == ["PMCID:PMC99990008"]


def test_the_pipeline_never_passes_on_treatment_evidence():
    class Careless(EmptyRetriever):
        def retrieve(self, condition, case, k=5):
            return [
                Evidence(
                    source="PMC",
                    title="t",
                    passage="Start heparin now.",
                    citation="PMCID:PMC99990009",
                )
            ]

    pipeline = DiagnosisPipeline(
        ConstantRanker(), InMemoryGraphStore(), Careless(), TemplateExplainer()
    )
    result = pipeline.run(PERICARDITIS_CASE)
    assert all(not c.evidence for c in result.candidates)


# --------------------------------------------------------------------------- #
# The corpus: invented JATS, licences, chunking, screened passages
# --------------------------------------------------------------------------- #


def jats(
    pmcid: str,
    title: str,
    body: str,
    *,
    licence: str | None = "by/4.0",
    kind: str = "research-article",
) -> str:
    permissions = (
        f'<permissions><license xlink:href="https://creativecommons.org/licenses/{licence}/">'
        "<license-p>Open access.</license-p></license></permissions>"
        if licence
        else "<permissions><copyright-statement>All rights reserved.</copyright-statement></permissions>"
    )
    return f"""<article article-type="{kind}" xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta><journal-title-group><journal-title>Invented Journal</journal-title></journal-title-group></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmc">{pmcid.removeprefix("PMC")}</article-id>
      <title-group><article-title>{title}</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="author"><name><surname>Doe</surname><given-names>Jane A</given-names></name></contrib>
        <contrib contrib-type="author"><name><surname>Roe</surname><given-names>R</given-names></name></contrib>
      </contrib-group>
      <pub-date><year>2021</year></pub-date>
      {permissions}
      <abstract><p>{title} is described in this invented abstract, which says nothing true about anybody.<xref ref-type="bibr">1</xref></p></abstract>
    </article-meta>
  </front>
  <body>{body}</body>
</article>"""


CLEAN_BODY = """
<sec><title>Clinical presentation</title>
  <p>Pericarditis gives sharp pleuritic chest pain that eases on sitting forward in most adults.
  It was treated with colchicine for three months. A pericardial friction rub is heard in a third
  of patients at presentation. Widespread ST elevation with PR depression supports the diagnosis.</p>
  <table-wrap><table><tr><td>a table is not text</td></tr></table></table-wrap>
</sec>
<sec><title>Management</title><p>Ibuprofen was given to every invented patient.</p></sec>
<sec><title>Methods</title><p>We searched invented records.</p></sec>
"""


def articleset(*articles: str) -> str:
    return "<pmc-articleset>" + "".join(articles) + "</pmc-articleset>"


def test_jats_is_parsed_with_its_licence_and_without_treatment_or_methods_sections():
    xml = articleset(
        jats("PMC99990011", "Pericarditis in an invented cohort", CLEAN_BODY),
        jats("PMC99990012", "An invented closed article", CLEAN_BODY, licence=None),
        jats("PMC99990013", "An invented notice", CLEAN_BODY, kind="retraction"),
    )
    kept, dropped = parse_articles(xml)
    assert [a.pmcid for a in kept] == ["PMC99990011"]
    article = kept[0]
    assert article.licence == "CC BY 4.0"
    assert article.licence_url == "https://creativecommons.org/licenses/by/4.0/"
    assert (article.authors, article.year, article.journal) == (
        "Doe JA et al.",
        2021,
        "Invented Journal",
    )
    assert [title for title, _ in article.sections] == ["Abstract", "Clinical presentation"]
    assert "table" not in article.sections[1][1] and "[1]" not in article.sections[0][1]
    assert {d["pmcid"]: d["reason"] for d in dropped} == {
        "PMC99990012": "no Creative Commons licence in the XML",
        "PMC99990013": "article type retraction",
    }
    assert article.metadata()["url"] == article_url("PMC99990011")


@pytest.mark.parametrize(
    "text, code",
    [
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC BY-NC-ND 4.0"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0 1.0"),
        ("Creative Commons Attribution-NonCommercial 4.0 International License", "CC BY-NC 4.0"),
        ("This article is under a CC BY 4.0 licence.", "CC BY 4.0"),
        ("All rights reserved.", None),
        ("https://creativecommons.org/publicdomain/mark/1.0/", None),
    ],
)
def test_licence_codes(text, code):
    assert licence_code(text) == code


def whitespace_offsets(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_chunks_are_windows_of_tokens_at_word_boundaries():
    text = " ".join(f"w{i}" for i in range(10))
    spans = chunk_spans(whitespace_offsets(text), window=4, overlap=1)
    words = [text[s:e].split() for s, e in spans]
    assert all(len(w) <= 4 for w in words)
    assert words[0] == ["w0", "w1", "w2", "w3"] and words[1][0] == "w3"
    assert words[-1][-1] == "w9"
    with pytest.raises(ValueError):
        chunk_spans(whitespace_offsets(text), window=4, overlap=4)


def test_passages_are_verbatim_screened_runs_of_their_sections():
    kept, _ = parse_articles(
        articleset(jats("PMC99990011", "Pericarditis in an invented cohort", CLEAN_BODY))
    )
    passages = article_passages(kept[0], whitespace_offsets, window=12, overlap=3, min_chars=40)
    sections = dict(kept[0].sections)
    assert passages
    for passage in passages:
        assert passage["text"] in sections[passage["section"]]
        assert not treatment_language(passage["text"])
    joined = " ".join(p["text"] for p in passages)
    assert "colchicine" not in joined and "Ibuprofen" not in joined
    # Nothing spans the removed sentence: "forward in most adults." and "A pericardial" are never
    # in one passage.
    assert not any("adults." in p["text"] and "pericardial" in p["text"] for p in passages)


# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #


def test_the_vocabulary_names_every_condition_and_nothing_else():
    assert set(CONDITION_TERMS) == {c.id for c in CONDITIONS}
    for c in CONDITIONS:
        assert resolve_condition(c.label) == c.id == resolve_condition(c.id)
    with pytest.raises(ValueError):
        resolve_condition("pneumonia")  # closed-world (R-13)


def test_the_query_vocabulary_has_a_row_per_concept_and_named_check_rows():
    from src.medical_kg.crosswalk import BY_CONCEPT

    rows = query_vocabulary({"GC-T": ["SYM:fever", "SYM:fever", "SYM:pleuritic"]})
    assert {r["id"] for r in rows if r["kind"] == "concept"} == set(BY_CONCEPT)
    checks = [r for r in rows if r["kind"] == "check"]
    assert len(checks) == len(CONDITION_TERMS)
    assert checks[0]["concepts"] == ["SYM:fever", "SYM:pleuritic"]
    assert checks[0]["text"] == compose_query_text(
        checks[0]["condition"], ["SYM:fever", "SYM:pleuritic"]
    )


# --------------------------------------------------------------------------- #
# The bundle loader
# --------------------------------------------------------------------------- #


def test_the_bundle_loads_and_widens_float16(tmp_path):
    bundle = load_bundle(write_fake_bundle(tmp_path / "b16", passage_dtype="float16"))
    assert bundle.passage_vectors.dtype == np.float16
    scores = bundle.passage_scores(np.array([axis(0)]))
    assert scores.dtype == np.float32
    assert scores[0, 5] == pytest.approx(1.0, abs=1e-3)


def rewrite_manifest(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"], manifest["format"], manifest["format_version"]
    (path / "manifest.json").unlink()
    write_manifest(path, manifest)


def corrupt_hash(path: Path) -> None:
    with open(path / "passages.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"pmcid": "PMC99990001", "section": "x", "text": "added later"}\n')


def missing_file(path: Path) -> None:
    (path / "query_embeddings.npy").unlink()


def fewer_vectors(path: Path) -> None:
    vectors = np.load(path / "passage_embeddings.npy")
    np.save(path / "passage_embeddings.npy", vectors[:-1], allow_pickle=False)
    rewrite_manifest(path)


def other_dimension(path: Path) -> None:
    np.save(path / "query_embeddings.npy", np.ones((8, 4), dtype=np.float32), allow_pickle=False)
    rewrite_manifest(path)


def zero_vector(path: Path) -> None:
    vectors = np.load(path / "passage_embeddings.npy")
    vectors[2] = 0
    np.save(path / "passage_embeddings.npy", vectors, allow_pickle=False)
    rewrite_manifest(path)


def not_finite(path: Path) -> None:
    vectors = np.load(path / "passage_embeddings.npy")
    vectors[1, 1] = np.nan
    np.save(path / "passage_embeddings.npy", vectors, allow_pickle=False)
    rewrite_manifest(path)


def orphan_passage(path: Path) -> None:
    with open(path / "passages.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"pmcid": "PMC99990099", "section": "x", "text": "an orphan"}\n')
    vectors = np.load(path / "passage_embeddings.npy")
    np.save(path / "passage_embeddings.npy", np.vstack([vectors, vectors[:1]]), allow_pickle=False)
    rewrite_manifest(path)


def malformed_files(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = {name: "not a record" for name in manifest["files"]}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def unlisted_file(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"]["passages.jsonl"]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def outside_file(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["../outside.txt"] = {"sha256": "0" * 64}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def other_format(path: Path) -> None:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["format_version"] = 99
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def not_an_object(path: Path) -> None:
    (path / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")


def pickled_array(path: Path) -> None:
    np.save(
        path / "passage_embeddings.npy", np.array([{"a": 1}] * 7, dtype=object), allow_pickle=True
    )
    rewrite_manifest(path)


REFUSALS = {
    "a file changed after the manifest": (corrupt_hash, "does not match the manifest"),
    "a missing file": (missing_file, "missing"),
    "fewer vectors than passages": (fewer_vectors, "passages but"),
    "two embedding sizes": (other_dimension, "two models"),
    "a zero vector": (zero_vector, "zero vector"),
    "a value that is not finite": (not_finite, "not finite"),
    "a passage without its article": (orphan_passage, "article not in"),
    "files that are not records": (malformed_files, "sha256 and size"),
    "a required file not listed": (unlisted_file, "does not list"),
    "a file outside the bundle": (outside_file, "outside the bundle"),
    "another format": (other_format, "manifest"),
    "a manifest that is not an object": (not_an_object, "not a JSON object"),
}


@pytest.mark.parametrize("damage, message", REFUSALS.values(), ids=REFUSALS.keys())
def test_the_loader_refuses_a_bundle_that_is_not_what_its_manifest_says(
    bundle_dir, damage, message
):
    damage(bundle_dir)
    with pytest.raises(BundleError, match=message):
        load_bundle(bundle_dir)


@pytest.mark.parametrize(
    "damage", [*(d for d, _ in REFUSALS.values()), pickled_array], ids=[*REFUSALS, "a pickle"]
)
def test_open_retriever_degrades_instead_of_raising(bundle_dir, damage):
    damage(bundle_dir)
    retriever = open_retriever(bundle_dir)
    assert isinstance(retriever, UnavailableRetriever) and retriever.degraded
    assert retriever.retrieve("pericarditis", PERICARDITIS_CASE) == []


def test_no_bundle_or_none_is_degraded_with_a_reason(tmp_path):
    assert "switched off" in open_retriever(None).reason
    absent = open_retriever(tmp_path / "nothing here")
    assert absent.degraded and "no manifest.json" in absent.reason
    assert describe(absent).startswith("retrieval: none (")
    assert describe(None) == "retrieval: none"


# --------------------------------------------------------------------------- #
# The retriever
# --------------------------------------------------------------------------- #


def test_evidence_is_cited_by_pmcid_with_a_link_and_one_passage_per_article(retriever):
    evidence = retriever.retrieve("COND:pericarditis", PERICARDITIS_CASE, k=5)
    citations = [e.citation for e in evidence]
    assert set(citations) == {"PMCID:PMC99990001", "PMCID:PMC99990003"}
    assert len(citations) == len(set(citations))
    for e in evidence:
        pmcid = e.citation.removeprefix("PMCID:")
        assert e.source == "PMC" and e.url == f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        assert e.passage.strip("…") in " ".join(t for p, _, t, _ in PASSAGES if p == pmcid)
    assert retriever.source_of(citations[0])["licence"].startswith("CC BY")
    assert len(retriever.retrieve("COND:pericarditis", PERICARDITIS_CASE, k=1)) == 1


def test_only_passages_naming_the_condition_are_shown(retriever):
    # The embolism passage mentions pleuritic pain, like pericarditis's, but never pericarditis.
    assert 2 not in retriever.eligible("COND:pericarditis")
    # One passage names both conditions and may be shown under either.
    assert 3 in retriever.eligible("COND:pericarditis") and 3 in retriever.eligible(
        "COND:pulmonary_embolism"
    )
    assert retriever.retrieve("COND:nstemi_stemi", PERICARDITIS_CASE) == []  # none names it
    # The unrelated passage of an article whose title names no condition is never eligible.
    assert all(4 not in retriever.eligible(cid) for cid in CONDITION_TERMS)


def test_the_case_s_findings_steer_which_passage_of_an_article_is_shown(retriever):
    plain = retriever.rank("COND:pericarditis", [], methods=("dense",))
    fever = retriever.rank("COND:pericarditis", ["SYM:fever"], methods=("dense",))
    assert [i for i, _ in plain] == [0, 3]  # a tie between 0 and 1 goes to the earlier passage
    assert [i for i, _ in fever] == [1, 3]  # the fever passage, once fever is present


def test_denied_and_unknown_findings_never_enter_a_query(retriever, monkeypatch):
    import src.rag.retriever as module

    seen = []
    original = module.compose_query_text
    monkeypatch.setattr(
        module,
        "compose_query_text",
        lambda cid, concepts=(): seen.append(list(concepts)) or original(cid, concepts),
    )
    denied = case(
        finding("SYM:pleuritic"),
        finding("SYM:fever", Assertion.ABSENT),
        finding("SYM:leg_swelling_unilateral", Assertion.UNKNOWN),
        Finding(concept_id="DDX:E_53", label="a DDXPlus question"),
    )
    assert present_concepts(denied) == ["SYM:pleuritic"]
    retriever.retrieve("COND:pericarditis", denied)
    assert seen and all(concepts == ["SYM:pleuritic"] for concepts in seen)
    assert retriever.retrieve("COND:pericarditis", denied) == retriever.retrieve(
        "COND:pericarditis", case(finding("SYM:pleuritic"))
    )


def test_the_dense_scores_equal_the_embeddings_times_the_mixed_query(retriever):
    bundle = retriever.bundle
    vectors = np.asarray(bundle.passage_vectors, dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    indices = np.arange(len(bundle.passages))
    for concepts in (
        [],
        ["SYM:fever"],
        ["SYM:pleuritic", "SYM:leg_swelling_unilateral"],
        ["SYM:unknown"],
    ):
        query = retriever.query_vector("COND:pericarditis", concepts)
        direct = vectors @ query
        assert retriever.dense_scores("COND:pericarditis", concepts, indices) == pytest.approx(
            direct, abs=1e-6
        )
    assert retriever.query_vector("COND:pericarditis", ["SYM:fever"]) == pytest.approx(
        mixed(axis(0), axis(3)), abs=1e-6
    )
    assert (
        retriever.dense_scores("COND:gerd", [], indices) is None
    )  # no vector for it in the bundle


def test_bm25_alone_dense_alone_and_the_hybrid(retriever):
    for methods in (("bm25",), ("dense",), ("bm25", "dense")):
        ranked = retriever.rank("COND:pulmonary_embolism", ["SYM:pleuritic"], methods=methods)
        assert {i for i, _ in ranked} <= {2, 3} and ranked
    with pytest.raises(ValueError):
        retriever.rank("COND:pericarditis", [], methods=("rerank",))


def test_mix_fidelity_measures_the_mixed_vector_against_the_full_sentence(retriever):
    (record,) = mix_fidelity(retriever)
    assert record["case"] == "GC-T" and record["condition"] == "COND:pericarditis"
    assert record["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert record["overlap"] == 1.0
    assert np.allclose(retriever.query_vector("COND:pericarditis", ["SYM:fever"]), FEVER_CHECK)


def test_a_concept_without_a_vector_is_named(retriever):
    assert "SYM:chest_pain" in retriever.missing_concepts
    assert "SYM:fever" not in retriever.missing_concepts


def test_describe_names_the_corpus(retriever):
    assert describe(retriever) == (
        "retrieval: BM25 + dense (RRF) over 7 passages from 5 PMC Open Access articles, built "
        "2026-10-14"
    )


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


class Spy(EmptyRetriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, condition, case, k=5):
        self.calls.append((condition, k))
        return [
            Evidence(
                source="PMC",
                title="An invented title",
                passage="Pain at rest.",
                citation="PMCID:PMC99990001",
            )
        ]


def stub_pipeline(retriever, **kwargs) -> DiagnosisPipeline:
    return DiagnosisPipeline(
        ConstantRanker(), InMemoryGraphStore(), retriever, TemplateExplainer(), **kwargs
    )


def test_evidence_goes_to_the_top_candidates_only():
    spy = Spy()
    result = stub_pipeline(spy, top_k_evidence=2, evidence_per_candidate=4).run(PERICARDITIS_CASE)
    assert spy.calls == [(c.label, 4) for c in result.candidates[:2]]
    assert all(c.evidence for c in result.candidates[:2])
    assert not any(c.evidence for c in result.candidates[2:])
    assert "rag" not in result.degraded_components


def test_evidence_never_changes_the_ranking(retriever):
    for golden in GOLDEN:
        golden_case = PatientCase(**golden["case"])
        with_evidence = stub_pipeline(retriever, top_k_evidence=14).run(golden_case)
        without = stub_pipeline(None).run(golden_case)
        assert [c.condition_id for c in with_evidence.candidates] == [
            c.condition_id for c in without.candidates
        ]


@pytest.mark.golden
@pytest.mark.parametrize("golden", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_golden_cases_hold_with_retrieval(retriever, golden):
    """The golden expectations, through the stub pipeline with the invented bundle attached."""
    result = stub_pipeline(retriever).run(PatientCase(**golden["case"]))
    expect = golden["expect"]
    top = [c.condition_id for c in result.candidates[: expect.get("top_k", 3)]]
    for required in expect.get("must_include", []):
        assert required in top
    flagged = {c.condition_id for c in result.candidates if c.red_flag}
    for required in expect.get("red_flag_for", []):
        assert required in flagged
    if expect.get("no_red_flag"):
        assert not flagged


def test_the_pipeline_attaches_real_evidence_from_the_bundle(retriever):
    result = stub_pipeline(retriever, top_k_evidence=14).run(PERICARDITIS_CASE)
    by_id = {c.condition_id: c for c in result.candidates}
    assert {e.citation for e in by_id["COND:pericarditis"].evidence} == {
        "PMCID:PMC99990001",
        "PMCID:PMC99990003",
    }
    assert {e.citation for e in by_id["COND:pulmonary_embolism"].evidence} == {
        "PMCID:PMC99990002",
        "PMCID:PMC99990003",
    }
    assert by_id["COND:gerd"].evidence == []
    assert "rag" not in result.degraded_components


def test_without_a_bundle_the_result_reports_rag_and_still_ranks(tmp_path):
    result = stub_pipeline(open_retriever(tmp_path / "absent")).run(PERICARDITIS_CASE)
    assert "rag" in result.degraded_components
    assert len(result.candidates) == len(CONDITIONS)
    assert not any(c.evidence for c in result.candidates)


def test_a_failing_retriever_degrades_rather_than_crashes():
    class Broken(EmptyRetriever):
        def retrieve(self, condition, case, k=5):
            raise RuntimeError("the bundle vanished")

    result = stub_pipeline(Broken()).run(PERICARDITIS_CASE)
    assert result.degraded_components.count("rag") == 1


def test_the_configuration_builds_the_retriever_and_its_settings(bundle_dir, tmp_path):
    from _config import build_pipeline, open_retriever_from_config

    assert open_retriever_from_config({"retrieval": {"bundle": None}}).degraded
    assert open_retriever_from_config({}).degraded
    built = build_pipeline(
        {
            "ml": {"backend": "none"},
            "graph": {"backend": "stub"},
            "ranking": {"top_k_evidence": 2},
            "retrieval": {"bundle": str(bundle_dir), "passages_per_candidate": 1, "top_k": 5},
        }
    )
    assert isinstance(built["retriever"], HybridRetriever) and built["retriever"].depth == 5
    assert (built["pipeline"].top_k_evidence, built["pipeline"].evidence_per_candidate) == (2, 1)
    assert built["about"]["retriever"].startswith("retrieval: BM25 + dense")
    result = built["pipeline"].run(PERICARDITIS_CASE)
    assert all(len(c.evidence) <= 1 for c in result.candidates)


def test_the_shipped_configuration_has_retrieval_off_by_default():
    from _config import load_config

    retrieval = load_config()["retrieval"]
    assert retrieval["bundle"] is None and retrieval["rrf_k"] == 60
    assert retrieval["embedding_model"] == "pritamdeka/S-PubMedBert-MS-MARCO"


# --------------------------------------------------------------------------- #
# The dashboard's evidence rows (app/view.py)
# --------------------------------------------------------------------------- #


def test_the_dashboard_shows_each_passage_with_its_citation_link_and_licence(retriever):
    from app.view import evidence_markdown, evidence_note, evidence_rows

    result = stub_pipeline(retriever, top_k_evidence=14).run(PERICARDITIS_CASE)
    rows = evidence_rows(result, retriever)["COND:pericarditis"]
    first = rows[0]
    assert first["url"].startswith("https://pmc.ncbi.nlm.nih.gov/articles/PMC9999000")
    assert first["licence"] == "CC BY 4.0"
    assert first["attribution"] == "Doe J et al., Invented Journal of Test Medicine, 2020"
    text = evidence_markdown(first)
    assert f"]({first['url']})" in text and "[CC BY 4.0](https://creativecommons.org/" in text
    assert evidence_note(result) is None
    degraded = stub_pipeline(None).run(PERICARDITIS_CASE)
    assert evidence_note(degraded).startswith("No evidence bundle is loaded")
    assert evidence_rows(degraded) == {}


def test_the_dashboard_escapes_literature_text_and_links_only_to_pmc():
    from app.view import evidence_markdown, evidence_rows

    sneaky = Evidence(
        source="PMC",
        title="An *invented* title",
        passage="See [this](https://example.org) and pay $5 now.",
        citation="PMCID:PMC99990001",
        url="https://example.org/elsewhere",
    )
    result = DiagnosisResult(
        case_id="T",
        candidates=[Candidate(condition_id="COND:gerd", label="GERD", evidence=[sneaky])],
    )
    row = evidence_rows(result)["COND:gerd"][0]
    assert row["url"] == "https://pmc.ncbi.nlm.nih.gov/articles/PMC99990001/"
    text = evidence_markdown(row)
    assert "](https://example.org)" not in text and "\\[this\\]" in text and "\\$5" in text
    assert row["licence"] is None and "None" not in text


# --------------------------------------------------------------------------- #
# The Kaggle build, offline: a fake NCBI and a fake model
# --------------------------------------------------------------------------- #


class FakeNCBI:
    """Answers ESearch with fixed ids per condition and EFetch with invented JATS."""

    def __init__(self, ids: dict[str, list[str]], articles: dict[str, str]) -> None:
        self.ids = ids
        self.articles = articles
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> tuple[int, bytes]:
        self.urls.append(url)
        from urllib.parse import parse_qs, urlparse

        query = parse_qs(urlparse(url).query)
        if "esearch" in url:
            term = query["term"][0]
            found = next((ids for name, ids in self.ids.items() if name in term), [])
            body = {
                "esearchresult": {
                    "count": str(len(found)),
                    "idlist": found,
                    "querytranslation": term,
                }
            }
            return 200, json.dumps(body).encode()
        if "efetch" in url:
            wanted = query["id"][0].split(",")
            return (
                200,
                articleset(*(self.articles[i] for i in wanted if i in self.articles)).encode(),
            )
        return 404, b"unknown"


def fake_embed(texts: list[str]) -> np.ndarray:
    """Each word hashed to one of 16 axes: deterministic, normalised, no model."""
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in tokenize(text) or ["empty"]:
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 16] += 1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def fake_ncbi() -> FakeNCBI:
    peri = (
        "<sec><title>Presentation</title><p>Pericarditis in this invented cohort gave sharp "
        "pleuritic chest pain that eased on sitting forward, often after a viral illness. A "
        "friction rub was heard in a third of the invented patients at their first visit.</p></sec>"
    )
    pe = (
        "<sec><title>Presentation</title><p>Pulmonary embolism in this invented series began with "
        "sudden breathlessness and pleuritic pain, and unilateral leg swelling was common among "
        "the invented patients who had been immobilised.</p></sec>"
    )
    return FakeNCBI(
        ids={"pericarditis": ["99990022", "99990021"], "pulmonary embolism": ["99990023"]},
        articles={
            "99990021": jats("PMC99990021", "Pericarditis in an invented cohort", peri),
            "99990022": jats("PMC99990022", "Colchicine for invented pericarditis", peri),
            "99990023": jats("PMC99990023", "Pulmonary embolism in an invented series", pe),
        },
    )


def offline_eutils(opener) -> EUtils:
    return EUtils(opener=opener, sleep=lambda seconds: None, min_interval=0.0)


def test_the_whole_build_runs_offline_and_its_bundle_loads(tmp_path):
    ncbi = fake_ncbi()
    out = tmp_path / "retrieval_corpus"
    manifest = build_bundle(
        out,
        eutils=offline_eutils(ncbi),
        embed=fake_embed,
        offsets_of=whitespace_offsets,
        window=20,
        overlap=4,
        per_condition=1,
        checks=golden_checks(REPO_ROOT / "tests" / "fixtures" / "golden_cases.yaml"),
        conditions=["COND:pericarditis", "COND:pulmonary_embolism"],
        cache=tmp_path / "cache" / "fetched.jsonl",
        info={"model": "fake"},
        log=lambda message: None,
    )
    assert manifest["counts"]["articles"] == 2
    assert manifest["licences"] == {"CC BY 4.0": 2}
    skipped = [json.loads(line) for line in (out / "skipped.jsonl").read_text("utf-8").splitlines()]
    assert {"pmcid": "PMC99990022", "reason": "the title names a treatment"} in skipped
    searches = json.loads((out / "search.json").read_text("utf-8"))
    assert all(s["term"].endswith('"open access"[filter]') for s in searches)
    assert not any("api_key" in url for url in ncbi.urls)

    bundle = load_bundle(out)
    retriever = HybridRetriever(bundle)
    evidence = retriever.retrieve("COND:pericarditis", PERICARDITIS_CASE)
    assert [e.citation for e in evidence] == ["PMCID:PMC99990021"]
    articles = {a["pmcid"]: a for a in bundle.articles.values()}
    assert articles["PMC99990021"]["conditions"] == ["COND:pericarditis"]
    assert {r["kind"] for r in bundle.query_rows} == {"condition", "concept", "check"}
    fidelity = mix_fidelity(retriever)
    assert len(fidelity) == len(GOLDEN) * len(CONDITION_TERMS)
    assert all(r["overlap"] is None for r in fidelity if r["condition"] == "COND:gerd")


def test_a_resumed_fetch_does_not_fetch_again(tmp_path):
    ncbi = fake_ncbi()
    cache = tmp_path / "fetched.jsonl"
    ids = ["PMC99990021", "PMC99990023", "PMC99990029"]
    first, skipped = fetch_articles(offline_eutils(ncbi), ids, cache=cache, log=lambda m: None)
    assert set(first) == {"PMC99990021", "PMC99990023"}
    assert skipped == [{"pmcid": "PMC99990029", "reason": "not returned by EFetch"}]
    fetches = sum("efetch" in url for url in ncbi.urls)
    again, _ = fetch_articles(offline_eutils(ncbi), ids, cache=cache, log=lambda m: None)
    assert set(again) == set(first) and sum("efetch" in url for url in ncbi.urls) == fetches
    assert isinstance(again["PMC99990021"], Article)


def test_searches_fall_back_to_ncbi_s_default_order(tmp_path):
    def opener(url, timeout):
        if "sort=relevance" in url:
            return 400, b"invalid sort"
        return 200, json.dumps({"esearchresult": {"count": "1", "idlist": ["99990021"]}}).encode()

    plan, searches = plan_searches(
        offline_eutils(opener), 1, conditions=["COND:pericarditis"], log=lambda m: None
    )
    assert plan == {"COND:pericarditis": [("PMC99990021", "diagnosis")]}
    assert searches[0]["sort"] is None


class FakeTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        assert kwargs["add_special_tokens"] is False and kwargs["return_offsets_mapping"]
        return {"offset_mapping": whitespace_offsets(text)}


class FakeModel:
    """Stands in for the sentence-transformers model: a fast tokenizer's offsets and encode()."""

    tokenizer = FakeTokenizer()
    max_seq_length = 40

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        return fake_embed(list(texts))


def test_the_kaggle_script_runs_end_to_end_with_a_fake_model_and_ncbi(tmp_path, monkeypatch):
    """scripts/build_retrieval_corpus.py as the notebook's smoke cell runs it, with the model and
    NCBI replaced: the wiring is checked here, not first on Kaggle."""
    import build_retrieval_corpus as script

    monkeypatch.setattr(
        script, "load_model", lambda name, revision, device: (FakeModel(), "abc123")
    )
    monkeypatch.setattr(script, "EUtils", lambda email=None: offline_eutils(fake_ncbi()))
    out = tmp_path / "smoke_bundle"
    assert script.main(["--out", str(out), "--smoke", "--device", "cpu"]) == 0
    manifest = load_bundle(out).manifest
    assert manifest["model"] == "pritamdeka/S-PubMedBert-MS-MARCO"
    assert manifest["model_revision"] == "abc123" and manifest["smoke"] is True
    assert manifest["max_seq_length"] == 40 and manifest["chunking"]["window_tokens"] == 38
    assert manifest["screening"]["digest"] and manifest["commit"]
    assert (tmp_path / "smoke_bundle_cache" / "fetched.jsonl").is_file()
    assert HybridRetriever(load_bundle(out)).retrieve("COND:pulmonary_embolism", PERICARDITIS_CASE)


# --------------------------------------------------------------------------- #
# The E-utilities client: rate limit, retries, nothing personal
# --------------------------------------------------------------------------- #


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_never_more_than_three_requests_a_second():
    clock = Clock()
    times = []

    def opener(url, timeout):
        times.append(clock.now)
        return 200, b"ok"

    client = EUtils(opener=opener, clock=clock, sleep=clock.sleep)
    for _ in range(10):
        client.get("https://eutils.example/x", {"db": "pmc"})
    gaps = np.diff(times)
    assert np.all(gaps >= MIN_INTERVAL - 1e-9)
    assert all(sum(t0 <= t < t0 + 1.0 for t in times) <= 3 for t0 in times)


def test_a_busy_ncbi_is_retried_with_growing_waits_and_an_error_is_not():
    clock = Clock()
    answers = iter([(429, b"busy"), (503, b"down"), (200, b"fine")])
    client = EUtils(opener=lambda url, timeout: next(answers), clock=clock, sleep=clock.sleep)
    assert client.get("https://eutils.example/x", {}) == b"fine"
    assert client.requests == 3 and clock.now >= 2.0 + 4.0
    refused = EUtils(opener=lambda url, timeout: (400, b"bad"), sleep=lambda s: None)
    with pytest.raises(EUtilsError):
        refused.get("https://eutils.example/x", {})
    assert refused.requests == 1


def test_no_key_is_sent_and_an_email_only_when_given():
    urls = []

    def opener(url, timeout):
        urls.append(url)
        return 200, b"ok"

    EUtils(opener=opener, sleep=lambda s: None).get("https://eutils.example/x", {"db": "pmc"})
    EUtils(email="owner@example.org", opener=opener, sleep=lambda s: None).get(
        "https://eutils.example/x", {"db": "pmc"}
    )
    assert "tool=nightingale" in urls[0] and "email" not in urls[0] and "api_key" not in urls[0]
    assert "email=owner%40example.org" in urls[1]
