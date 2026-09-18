"""The cardiac knowledge graph, built from all of its sources (task 1b).

Three sources, merged in this order (:func:`src.medical_kg.loader.merge_graphs`):

1. **DDXPlus** ``release_conditions.json``, the backbone: the 13 trainable conditions linked to
   evidence *questions*, each edge weight 1.0 (:func:`src.medical_kg.loader.load_ddxplus_kg`).
2. **Hand-authored** aortic dissection, from the ADD-RS and the IRAD registry
   (:mod:`src.medical_kg.hand_authored`).
3. **BODHI-S**, when its files are given: answer-level findings with likelihoods, for MI,
   pericarditis, PE and GERD (:mod:`src.medical_kg.bodhi_s`).

Every edge keeps its ``source``, so the graph can be reported, and ablated, by source (R-12):
:func:`src.medical_kg.loader.only_sources` keeps a subset.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from src.ddxplus import concept_id
from src.medical_kg.hand_authored import load_hand_authored_kg
from src.medical_kg.loader import KnowledgeGraph, load_ddxplus_kg, merge_graphs

__all__ = ["build_cardiac_kg", "question_labels"]


def question_labels(vocabulary_path: Path) -> dict[str, str]:
    """``DDX:E_nn`` → English label, for every DDXPlus question in the decoded vocabulary.

    The other sources use it to label the DDXPlus question nodes they add.
    """
    vocabulary: Mapping[str, dict] = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
    return {
        concept_id(code): entry.get("label_en") or code
        for code, entry in vocabulary.items()
        if entry.get("kind") == "question"
    }


def build_cardiac_kg(conditions_path: Path, vocabulary_path: Path) -> KnowledgeGraph:
    """DDXPlus plus the hand-authored aortic dissection, merged into one graph.

    Args:
        conditions_path: ``data/interim/ddxplus_chestpain_conditions.json``.
        vocabulary_path: ``data/interim/ddxplus_evidences.json``.
    """
    ddxplus = load_ddxplus_kg(conditions_path, vocabulary_path)
    labels = question_labels(vocabulary_path)
    return merge_graphs(ddxplus, load_hand_authored_kg(labels))
