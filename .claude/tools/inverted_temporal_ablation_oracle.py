#!/usr/bin/env python3
"""D-SAG-CF-1: deliberately small, independent *Boolean provenance* oracle.

Run: python3 .claude/tools/inverted_temporal_ablation_oracle.py
No foreign reasoner, CE64 update, SCM/do(), probabilistic inference,
temporal reconstruction, or Quack implementation is claimed here.
The finite acyclic fixture is a parity baseline, NOT the production engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import unittest

Term = frozenset[str]
Proofs = dict[str, frozenset[Term]]


@dataclass(frozen=True)
class Rule:
    id: str
    premises: tuple[str, ...]
    conclusion: str


@dataclass(frozen=True)
class World:
    # Claim -> source root. An alias of the same observation uses the SAME root.
    observations: tuple[tuple[str, str], ...]
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class Shadow:
    lost_claims: frozenset[str]
    gained_claims: frozenset[str]
    changed_support: frozenset[str]
    removed_proofs: Proofs
    added_proofs: Proofs


def _validate_acyclic(world: World) -> None:
    """Fail closed instead of pretending an unbounded recursive proof is exact."""
    adjacency: dict[str, set[str]] = {}
    ids: set[str] = set()
    for rule in world.rules:
        if rule.id in ids:
            raise ValueError("rule ids must be unique and versioned")
        ids.add(rule.id)
        for premise in rule.premises:
            adjacency.setdefault(premise, set()).add(rule.conclusion)
    states: dict[str, int] = {}

    def dfs(node: str) -> None:
        if states.get(node) == 1:
            raise ValueError("recursive/cyclic provenance requires another oracle")
        if states.get(node) == 2:
            return
        states[node] = 1
        for child in sorted(adjacency.get(node, ())):
            dfs(child)
        states[node] = 2

    for node in sorted(adjacency):
        dfs(node)


def evaluate(
    world: World,
    cut_roots: frozenset[str] = frozenset(),
    cut_rules: frozenset[str] = frozenset(),
) -> Proofs:
    """Compute all minimal Boolean proof terms on a tiny DAG, from scratch.

    Terms are sets of original source IDs AND rule IDs. Alternatives are a
    set of terms. A shared root appears in every proof that depends on it.
    No numerical NARS revision, timestamps or causal certification occurs.
    """
    _validate_acyclic(world)
    roots: dict[str, set[Term]] = {}
    for claim, root in world.observations:
        if root not in cut_roots:
            roots.setdefault(claim, set()).add(frozenset((f"obs:{root}",)))

    # Finite DAG, no negative premises: monotone closure terminates.
    while True:
        before = sum(len(v) for v in roots.values())
        for rule in world.rules:
            if rule.id in cut_rules:
                continue
            supports = [roots.get(p, set()) for p in rule.premises]
            # A zero-premise rule has no supports and still proves its
            # conclusion with the rule-only term.
            if any(not s for s in supports):
                continue
            for chosen in product(*supports):
                evidence = frozenset.union(*chosen, frozenset((f"rule:{rule.id}",)))
                roots.setdefault(rule.conclusion, set()).add(evidence)
        after = sum(len(v) for v in roots.values())
        if after == before:
            break
    return {k: frozenset(v) for k, v in roots.items()}


def ablate(
    world: World,
    *,
    roots: frozenset[str] = frozenset(),
    rules: frozenset[str] = frozenset(),
) -> Shadow:
    factual = evaluate(world)
    counterfactual = evaluate(world, roots, rules)
    factual_keys = set(factual)
    cf_keys = set(counterfactual)
    touched = factual_keys | cf_keys
    removed = {
        k: frozenset(factual.get(k, frozenset()) - counterfactual.get(k, frozenset()))
        for k in sorted(touched)
        if factual.get(k, frozenset()) - counterfactual.get(k, frozenset())
    }
    added = {
        k: frozenset(counterfactual.get(k, frozenset()) - factual.get(k, frozenset()))
        for k in sorted(touched)
        if counterfactual.get(k, frozenset()) - factual.get(k, frozenset())
    }
    return Shadow(
        lost_claims=frozenset(factual_keys - cf_keys),
        gained_claims=frozenset(cf_keys - factual_keys),
        changed_support=frozenset(
            k for k in touched if factual.get(k) != counterfactual.get(k)
        ),
        removed_proofs=removed,
        added_proofs=added,
    )


def two_paths() -> World:
    return World(
        (("A", "a"), ("B", "b")),
        (Rule("R1@1", ("A",), "C"), Rule("R2@1", ("B",), "C")),
    )


def hidden_root() -> World:
    return World(
        (("U", "u"), ("D", "d")),
        (
            Rule("makeA@1", ("U",), "A"),
            Rule("makeB@1", ("U",), "B"),
            Rule("R1@1", ("A",), "C"),
            Rule("R2@1", ("B",), "C"),
            Rule("R3@1", ("D",), "C"),
        ),
    )


class ProbeTests(unittest.TestCase):
    def test_f0_pristine_replay_is_exact(self) -> None:
        w = hidden_root()
        self.assertEqual(evaluate(w), evaluate(w))

    def test_f1_cut_a_keeps_independent_b(self) -> None:
        s = ablate(two_paths(), roots=frozenset(("a",)))
        self.assertNotIn("C", s.lost_claims)
        self.assertIn("C", s.changed_support)
        self.assertEqual(len(s.removed_proofs["C"]), 1)

    def test_f2_cut_rule_not_its_input(self) -> None:
        s = ablate(two_paths(), rules=frozenset(("R1@1",)))
        self.assertIn("C", s.changed_support)
        self.assertNotIn("A", s.changed_support)
        self.assertNotIn("C", s.lost_claims)

    def test_f3_joint_cut_removes_all_support(self) -> None:
        s = ablate(two_paths(), rules=frozenset(("R1@1", "R2@1")))
        self.assertIn("C", s.lost_claims)

    def test_f4_shared_assumption_collapses_two_paths(self) -> None:
        s = ablate(hidden_root(), roots=frozenset(("u",)))
        self.assertIn("A", s.lost_claims)
        self.assertIn("B", s.lost_claims)
        self.assertNotIn("C", s.lost_claims)  # D independently supports C
        self.assertEqual(len(s.removed_proofs["C"]), 2)

    def test_f5_aliased_observation_cut_by_source(self) -> None:
        w = World(
            (("A1", "shared"), ("A2", "shared"), ("B", "independent")),
            (
                Rule("R1@1", ("A1",), "C"),
                Rule("R2@1", ("A2",), "C"),
                Rule("R3@1", ("B",), "C"),
            ),
        )
        s = ablate(w, roots=frozenset(("shared",)))
        self.assertEqual(s.lost_claims, frozenset(("A1", "A2")))
        self.assertNotIn("B", s.changed_support)
        self.assertNotIn("C", s.lost_claims)

    def test_f6_irrelevant_source_is_noop(self) -> None:
        self.assertEqual(
            ablate(two_paths(), roots=frozenset(("does-not-exist",))).changed_support,
            frozenset(),
        )

    def test_f7_contingency_exposes_pivotality(self) -> None:
        w = two_paths()
        self.assertNotIn("C", ablate(w, roots=frozenset(("a",))).lost_claims)
        self.assertIn("C", ablate(w, roots=frozenset(("a", "b"))).lost_claims)

    def test_f9_rule_cut_never_changes_observations(self) -> None:
        w = hidden_root()
        s = ablate(w, rules=frozenset(("makeA@1", "makeB@1")))
        self.assertNotIn("U", s.changed_support)
        self.assertNotIn("D", s.changed_support)
        self.assertIn("A", s.lost_claims)
        self.assertIn("B", s.lost_claims)

    def test_refuse_cycle_without_losing_evidence(self) -> None:
        w = World((("A", "a"),), (Rule("R@1", ("A",), "B"),
                                     Rule("S@1", ("B",), "A")))
        with self.assertRaisesRegex(ValueError, "cyclic"):
            evaluate(w)

    def test_zero_premise_rule_proves_and_its_cut_removes(self) -> None:
        w = World((), (Rule("R@1", (), "C"),))
        self.assertEqual(evaluate(w)["C"], frozenset((frozenset(("rule:R@1",)),)))
        s = ablate(w, rules=frozenset(("R@1",)))
        self.assertIn("C", s.lost_claims)

    def test_refuse_duplicate_rule_identity(self) -> None:
        w = World((("A", "a"),),
                  (Rule("R@1", ("A",), "C"), Rule("R@1", ("A",), "D")))
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate(w)

    def test_support_and_membership_are_distinct(self) -> None:
        s = ablate(hidden_root(), roots=frozenset(("d",)))
        self.assertNotIn("C", s.lost_claims)
        self.assertIn("C", s.changed_support)
        self.assertEqual(len(s.removed_proofs["C"]), 1)


if __name__ == "__main__":
    unittest.main()
