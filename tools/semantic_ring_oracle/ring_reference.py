"""Independent symbolic C-01/C-02 reference, never a production graph evaluator.

The binding gold is manually supplied. This module does NOT parse English,
allocate NodeGuids, lower R2IL, or infer the identity of two indefinite keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "docs/research/versioned-semantic-candidates-fixtures.json"
GOLD = ROOT / "docs/research/semantic-ring-bound-gold-v1.json"


def load_case(case_id: str) -> dict[str, Any]:
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    if fixtures["format"] != "lance-graph-versioned-semantic-candidates/fixtures-v1":
        raise ValueError("unexpected fixture version")
    matches = [c for c in fixtures["cases"] if c["id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected one fixture for {case_id}")
    return matches[0]


def load_gold() -> dict[str, Any]:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    if gold["format"] != "lance-graph-semantic-ring-bound-gold/v1":
        raise ValueError("unexpected gold version")
    return gold


def evaluate(
    case: dict[str, Any],
    gold: dict[str, Any],
    entry_order: list[int],
    *,
    facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Test-only oracle for symbolic bound relations and activated identity.

    It intentionally never consults case['expected']. A result is eligible for
    comparison with an implementation only when an independent binder produces
    its own claim IDs and relations under an explicit, compatible frame.
    """
    if case["id"] not in ("C-01", "C-02"):
        raise ValueError("this oracle covers only C-01 and C-02")
    source = case["input"]
    if source["sentences"] != gold["sentences"]:
        raise ValueError("sentence/gold binding drift")
    if entry_order not in source["entry_orders"] or sorted(entry_order) != list(range(4)):
        raise ValueError("unregistered or incomplete rotation")
    atoms = list(gold["facts"] if facts is None else facts)
    if len({a["sentence_index"] for a in atoms}) != len(atoms):
        raise ValueError("multiple atoms per sentence exceed this narrow oracle")
    if any(a["sentence_index"] not in range(4) for a in atoms):
        raise ValueError("unbound source sentence")
    selected = sorted(atoms, key=lambda a: entry_order.index(a["sentence_index"]))

    witness = source["external_identity_witness"]
    roots: tuple[str, ...] = ()
    redirects: dict[str, str] = {}
    if witness is not None:
        if witness not in gold["attested_identity"]:
            raise ValueError("unattested identity witness")
        if not all(witness.get(k) for k in ("lhs", "rhs", "root")):
            raise ValueError("incomplete identity witness")
        # Only the exact gold attestation licenses the substitution. Neither
        # lexical similarity nor thought-popcount licenses any other union.
        redirects[witness["lhs"]] = witness["rhs"]
        roots = (witness["root"],)

    def canonical(term: str) -> str:
        while term in redirects:
            term = redirects[term]
        return term

    relations = tuple(
        sorted((canonical(a["subject"]), a["predicate"], canonical(a["object"]))
               for a in selected)
    )
    # Reference traversal: require four distinct, source-bound arcs and a
    # complete return to the first key, not just a four-row presence count.
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for a in selected:
        adjacency.setdefault(canonical(a["subject"]), []).append(
            (canonical(a["object"]), a["sentence_index"])
        )
    current = canonical("first_sentence.key")
    visited: set[int] = set()
    for _ in range(4):
        options = [edge for edge in adjacency.get(current, []) if edge[1] not in visited]
        if len(options) != 1:
            break
        current, source_index = options[0]
        visited.add(source_index)
    closed = len(visited) == 4 and current == canonical("first_sentence.key")
    return {
        "cycle_closed": closed,
        "coreference_status": "WitnessedInFixture" if roots else "Unresolved",
        "relations": relations,
        "witness_roots": roots,
        "source_sentence_indices": tuple(sorted(visited)),
    }


def main() -> None:
    gold = load_gold()
    for case_id in ("C-01", "C-02"):
        case = load_case(case_id)
        results = [evaluate(case, gold, order) for order in case["input"]["entry_orders"]]
        if not all(x == results[0] for x in results):
            raise SystemExit(f"{case_id}: rotation changed the canonical answer")
        if results[0]["cycle_closed"] != case["expected"]["cycle_closed"]:
            raise SystemExit(f"{case_id}: reference mismatched preregistered expected closure")
        print(f"{case_id}: rotations={len(results)}, closed={results[0]['cycle_closed']}, "
              f"coreference={results[0]['coreference_status']}, roots={results[0]['witness_roots']}")
    print("Reference oracle only: no production parser or execution path was tested.")


if __name__ == "__main__":
    main()
