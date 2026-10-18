"""Independent symbolic C-01/C-02 reference, never a production graph evaluator.

Identity is the fixed coordinate. A sealed version witnesses that the
coreference claim was recorded at that coordinate. Source roots remain a
separate evidential property of the version receipt, not another ID.
This harness does NOT parse language, mint NodeGuids, or execute R2IL.
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
    if gold["format"] != "lance-graph-semantic-ring-bound-gold/v2":
        raise ValueError("unexpected gold version")
    return gold


def _validated_ref(ref: dict[str, Any]) -> tuple[str, str, int]:
    if not isinstance(ref, dict) or set(ref) != {"dataset", "branch", "version"}:
        raise ValueError("read reference must pin dataset, branch and version")
    dataset, branch, version = ref["dataset"], ref["branch"], ref["version"]
    if not isinstance(dataset, str) or not dataset or not isinstance(branch, str) or not branch:
        raise ValueError("read reference has unbound dataset/branch")
    if type(version) is not int or version < 0:
        raise ValueError("read reference version must be an explicit nonnegative integer")
    return dataset, branch, version


def evaluate(
    case: dict[str, Any],
    gold: dict[str, Any],
    entry_order: list[int],
    *,
    facts: list[dict[str, Any]] | None = None,
    read_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A narrow independent reference using pinned versions, never case.expected.

    The C-01/C-02 legacy 'external_identity_witness' inputs stay frozen as
    originally preregistered. They do NOT control admission. Instead, the
    recorded claim at its fixed address becomes visible at its sealed version.
    """
    if case["id"] not in ("C-01", "C-02"):
        raise ValueError("this oracle covers only C-01/C-02")
    if case["input"]["sentences"] != gold["sentences"]:
        raise ValueError("sentence/gold binding drift")
    if entry_order not in case["input"]["entry_orders"] or sorted(entry_order) != list(range(4)):
        raise ValueError("unregistered or incomplete rotation")
    if read_ref is None:
        read_ref = gold["read_heads"][case["id"]]
    dataset, branch, version = _validated_ref(read_ref)

    atoms = list(gold["facts"] if facts is None else facts)
    if len({a["sentence_index"] for a in atoms}) != len(atoms):
        raise ValueError("multiple atoms per sentence exceed this narrow oracle")
    if any(a["sentence_index"] not in range(4) for a in atoms):
        raise ValueError("unbound source sentence")
    selected = sorted(atoms, key=lambda a: entry_order.index(a["sentence_index"]))

    redirects: dict[str, str] = {}
    witnessed: list[tuple[str, str, str, int]] = []
    roots: set[str] = set()
    for assertion in gold["attested_identity"]:
        lhs, rhs = assertion["lhs"], assertion["rhs"]
        # This symbol is a fixed, manually bound *claim* address, never a
        # newly minted witness ID. The assertion is located by address.
        if assertion["claim_address"] != f"corefers({lhs},{rhs})":
            raise ValueError("identity assertion at wrong claim address")
        a_dataset, a_branch, a_version = _validated_ref(assertion["sealed_in"])
        if (a_dataset, a_branch) != (dataset, branch) or a_version > version:
            continue
        if lhs in redirects and redirects[lhs] != rhs:
            raise ValueError("conflicting visible identity assertions")
        redirects[lhs] = rhs
        witnessed.append((assertion["claim_address"], a_dataset, a_branch, a_version))
        roots.update(assertion["source_roots"])

    def canonical(term: str) -> str:
        seen: set[str] = set()
        while term in redirects:
            if term in seen:
                raise ValueError("cycle in identity assertions")
            seen.add(term)
            term = redirects[term]
        return term

    relations = tuple(
        sorted((canonical(a["subject"]), a["predicate"], canonical(a["object"]))
               for a in selected)
    )
    # The test-only traversal checks that the recorded four bound SPO facts
    # are a complete cycle, not whether a parser or Datalog join found them.
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for atom in selected:
        adjacency.setdefault(canonical(atom["subject"]), []).append(
            (canonical(atom["object"]), atom["sentence_index"])
        )
    current = canonical("first_sentence.key")
    visited: set[int] = set()
    for _ in range(4):
        options = [edge for edge in adjacency.get(current, []) if edge[1] not in visited]
        if len(options) != 1:
            break
        current, sentence_id = options[0]
        visited.add(sentence_id)
    closed = len(visited) == 4 and current == canonical("first_sentence.key")
    return {
        "cycle_closed": closed,
        "coreference_status": "VersionWitnessed" if witnessed else "Unresolved",
        "relations": relations,
        "witness_coordinates": tuple(sorted(witnessed)),
        "source_roots": tuple(sorted(roots)),
        "source_sentence_indices": tuple(sorted(visited)),
        "observed_ref": (dataset, branch, version),
    }


def main() -> None:
    gold = load_gold()
    for case_id in ("C-01", "C-02"):
        case = load_case(case_id)
        results = [evaluate(case, gold, order) for order in case["input"]["entry_orders"]]
        if not all(x == results[0] for x in results):
            raise SystemExit(f"{case_id}: rotation changed canonical result")
        if results[0]["cycle_closed"] != case["expected"]["cycle_closed"]:
            raise SystemExit(f"{case_id}: closure mismatched original pre-registration")
        print(f"{case_id}: rotations={len(results)}, closed={results[0]['cycle_closed']}, "
              f"ref={results[0]['observed_ref']}, witness={results[0]['witness_coordinates']}")
    print("Reference only: no production parser, R2IL joins or evidence gate tested.")


if __name__ == "__main__":
    main()
