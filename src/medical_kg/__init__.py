"""Nightingale medical_kg module — the cardiac knowledge graph (P1).

- ``loader``: builds a backend-neutral ``KnowledgeGraph`` from the DDXPlus condition KB.
- ``networkx_store``: ``NetworkXGraphStore``, the in-memory ``GraphStore`` (risk R-06).
- ``crosswalk``: links the hand-authored ``SYM:*`` / ``RF:*`` concepts to DDXPlus evidence.
"""
