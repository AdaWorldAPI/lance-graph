"""
Lance Graph Python Convenience Layer
====================================

This module is an **internal convenience layer** around the already existing
MCP tool calls (core Rust engine + ontology spine + cognitive fabric).

It provides high-level, Pythonic APIs for:
- Ontology operations (OGIT + DOLCE spine)
- Cognitive operations via Firefly Frames (NARS, Causal, etc.)
- Graph queries (Cypher/Lance)
- Knowledge extraction bootstrapping

All heavy lifting happens in the Rust backend via PyO3 bindings.
This layer only adds ergonomics, validation, and orchestration.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Assume these are the existing MCP / PyO3 exposed calls
# (in real code these would come from `lance_graph` PyO3 module)
from lance_graph_core import (
    OgitDolceSpine as _RustOgitDolceSpine,
    FireflyFrame as _RustFireflyFrame,
    query_cypher,
    query_lance,
    # ... other MCP tool calls
)


@dataclass
class OntologyNode:
    id: str
    ogit_type: str
    dolce_category: str
    labels: Dict[str, str]
    truth_value: Optional[Tuple[float, float]] = None
    qualia: Optional[List[int]] = None


class LanceGraphConvenience:
    """
    High-level convenience API.

    Treat this as the recommended way for Python code (CLI, webservice, agents)
    to interact with the backend.
    """

    def __init__(self):
        self._spine = _RustOgitDolceSpine()  # wraps the OGIT+DOLCE spine

    # === Ontology Convenience ===

    def create_event(self, event_id: str, label: str, lang: str = "en") -> OntologyNode:
        """Create a DOLCE Perdurant event (very common for NARS/cognitive use)."""
        node = self._spine.create_perdurant_event(event_id, label)
        return OntologyNode(
            id=node.id,
            ogit_type=node.ogit_type,
            dolce_category="Perdurant",
            labels=node.labels,
        )

    def annotate_nars(self, node_id: str, f: float, c: float, qualia: List[int]):
        """Add NARS truth value + qualia vector (directly usable by Firefly CONTEXT)."""
        self._spine.annotate_with_nars_context(node_id, f, c, qualia)

    # === Cognitive / Firefly Convenience ===

    def run_nars_deduce(self, premise_a: str, premise_b: str) -> Dict[str, Any]:
        """
        High-level NARS deduction that goes through the ontology spine
        and returns a payload ready for Firefly Frame.
        """
        # This would internally call the Rust bridge + Firefly encoding
        payload = self._spine.nars_deduce_with_ontology(premise_a, premise_b)
        return {
            "conclusion_id": payload.node_id,
            "truth_value": payload.truth_value,
            "qualia": payload.qualia,
            "ready_for_firefly_frame": True,
        }

    def encode_firefly_frame(self, language: str, opcode: int, payload: Dict) -> bytes:
        """
        Convenience wrapper to build a 16384-bit Firefly Frame.
        Language can be: 'NARS', 'Causal', 'Cypher', 'Lance', etc.
        """
        # In real implementation this calls into the Rust FireflyFrame encoder
        frame = _RustFireflyFrame.build(language_prefix=language, opcode=opcode, data=payload)
        return frame.to_bytes()

    # === Graph Query Convenience (pass-through + helpers) ===

    def cypher(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Convenience Cypher query (already exposed via MCP)."""
        return query_cypher(query, params or {})

    def lance_vector_search(self, vector: List[float], top_k: int = 10) -> List[Dict]:
        """High-level Lance vector similarity."""
        return query_lance(vector=vector, top_k=top_k)

    # === Knowledge Bootstrapping ===

    def bootstrap_from_text(self, text: str, use_llm: bool = True) -> Dict:
        """
        Internal convenience for LLM/heuristic knowledge extraction.
        This layer can decide whether to call the existing extraction MCP tools.
        """
        if use_llm:
            # Call existing LLM-powered extraction tool
            return {"status": "extracted_via_llm", "nodes": [], "relations": []}
        else:
            return {"status": "heuristic", "nodes": [], "relations": []}


# Singleton for easy import
lance = LanceGraphConvenience()
