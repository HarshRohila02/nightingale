"""Nightingale medical_kg module — the cardiac knowledge graph (P1).

- ``loader``: builds a backend-neutral ``KnowledgeGraph`` from the DDXPlus condition KB.
- ``hand_authored``: aortic dissection from the ADD-RS and the IRAD registry.
- ``bodhi_s``: BODHI-S enrichment for MI, pericarditis, PE and GERD.
- ``cardiac_kg``: ``build_cardiac_kg`` merges the sources into one graph.
- ``networkx_store``: ``NetworkXGraphStore``, the in-memory ``GraphStore`` (risk R-06).
- ``neo4j_store``: ``Neo4jGraphStore``, the graph on AuraDB, and ``open_graph_store``,
  which falls back to NetworkX when Aura cannot be reached.
- ``crosswalk``: links the hand-authored ``SYM:*`` / ``RF:*`` concepts to DDXPlus evidence.
"""
