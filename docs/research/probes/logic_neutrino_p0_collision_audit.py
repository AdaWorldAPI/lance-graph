#!/usr/bin/env python3
"""P0 semantic twins: a test-only candidate selector, NOT a compiler.

Run: python3 docs/research/probes/logic_neutrino_p0_collision_audit.py
No third-party dependencies. No imports from production Rust code.
The CE64 reader in this file is an independent scalar bit oracle; the
separate Rust test exercises the real CausalEdge64 orientation accessor.
"""

from dataclasses import dataclass, replace
from enum import Enum, auto
import unittest


# A deliberately colliding local candidate position, NOT a registered opcode.
LEFT_CORNER: int = 0x2A
RECIPIENT: int = 2  # CE64 Orientation::Recipient, bits 43..45.
ORIENTATION_SHIFT: int = 43
ORIENTATION_MASK: int = 0b111


class Domain(Enum):
    RAILS = auto()
    CE64 = auto()
    RIVER = auto()


class Rule(Enum):
    RAILS_INCOMPLETE_COUNT = auto()
    CE64_RECIPIENT_SELECTION = auto()
    RIVER_EXTRA_TURN = auto()


class Refusal(Enum):
    UNWITNESSED = auto()
    WRONG_DOMAIN = auto()
    STALE_SOURCE = auto()
    WRONG_CLASS_READING = auto()
    WRONG_RECEIVER = auto()
    WRONG_ROLE = auto()
    UNKNOWN_RULESET = auto()
    INSUFFICIENT_COAL = auto()
    WRONG_MANEUVER = auto()


RULE_DOMAIN = {
    Rule.RAILS_INCOMPLETE_COUNT: Domain.RAILS,
    Rule.CE64_RECIPIENT_SELECTION: Domain.CE64,
    Rule.RIVER_EXTRA_TURN: Domain.RIVER,
}
RULE_READING = {
    Rule.RAILS_INCOMPLETE_COUNT: "rails-association-v1",
    Rule.CE64_RECIPIENT_SELECTION: "ce64-orientation-v1",
    Rule.RIVER_EXTRA_TURN: "mississippi-queen-classic-v1",
}


@dataclass(frozen=True)
class Context:
    domain: Domain
    class_reading: str
    source_revision: int = 7
    declared_revision: int = 7
    witnessed: bool = True
    receiver: str = ""
    scope_declared: bool = False
    ce64_word: int = 0
    coal: int = 0
    sixty_degree_turns: int = 0
    ruleset: str = ""


def candidates(first_construct: int) -> tuple[Rule, ...]:
    """Indexing only. No identity, grammar, authorization or execution."""
    if first_construct != LEFT_CORNER:
        return ()
    return tuple(Rule)


def admit(rule: Rule, ctx: Context) -> Refusal | None:
    """Test-local gate. None means permitted by THIS tiny fixture only."""
    if not ctx.witnessed:
        return Refusal.UNWITNESSED
    if RULE_DOMAIN[rule] is not ctx.domain:
        return Refusal.WRONG_DOMAIN
    if ctx.source_revision != ctx.declared_revision:
        return Refusal.STALE_SOURCE
    if ctx.class_reading != RULE_READING[rule]:
        return Refusal.WRONG_CLASS_READING

    if rule is Rule.RAILS_INCOMPLETE_COUNT:
        if ctx.receiver != "Checklist::Item" or not ctx.scope_declared:
            return Refusal.WRONG_RECEIVER
    elif rule is Rule.CE64_RECIPIENT_SELECTION:
        if ((ctx.ce64_word >> ORIENTATION_SHIFT) & ORIENTATION_MASK) != RECIPIENT:
            return Refusal.WRONG_ROLE
    elif rule is Rule.RIVER_EXTRA_TURN:
        if ctx.ruleset != "classic":
            return Refusal.UNKNOWN_RULESET
        if ctx.sixty_degree_turns != 3:
            return Refusal.WRONG_MANEUVER
        # Independent fixture pins one free 60-degree turn. A 180-degree
        # rotation has three 60-degree steps, hence exactly two coal.
        if ctx.coal < 2:
            return Refusal.INSUFFICIENT_COAL
    return None


class P0SemanticTwins(unittest.TestCase):
    def setUp(self) -> None:
        self.rails = Context(
            domain=Domain.RAILS,
            class_reading=RULE_READING[Rule.RAILS_INCOMPLETE_COUNT],
            receiver="Checklist::Item",
            scope_declared=True,
        )
        self.role = Context(
            domain=Domain.CE64,
            class_reading=RULE_READING[Rule.CE64_RECIPIENT_SELECTION],
            ce64_word=(0x00123456 | (RECIPIENT << ORIENTATION_SHIFT)),
        )
        self.river = Context(
            domain=Domain.RIVER,
            class_reading=RULE_READING[Rule.RIVER_EXTRA_TURN],
            ruleset="classic",
            sixty_degree_turns=3,
            coal=6,
        )

    def test_selector_only_proposes_all_three_rules(self) -> None:
        self.assertEqual(set(candidates(LEFT_CORNER)), set(Rule))
        self.assertEqual(candidates(LEFT_CORNER ^ 1), ())
        # One byte cannot choose a semantic domain or execute a rule.
        self.assertEqual(len(candidates(LEFT_CORNER)), 3)

    def test_rails_receiver_twin_and_missing_scope(self) -> None:
        rule = Rule.RAILS_INCOMPLETE_COUNT
        self.assertIsNone(admit(rule, self.rails))
        self.assertEqual(
            admit(rule, replace(self.rails, receiver="Audit::Item")),
            Refusal.WRONG_RECEIVER,
        )
        self.assertEqual(
            admit(rule, replace(self.rails, scope_declared=False)),
            Refusal.WRONG_RECEIVER,
        )

    def test_ce64_orientation_twin_and_epi5_independence(self) -> None:
        rule = Rule.CE64_RECIPIENT_SELECTION
        self.assertIsNone(admit(rule, self.role))
        core = replace(self.role, ce64_word=self.role.ce64_word & ~(7 << 43))
        self.assertEqual(admit(rule, core), Refusal.WRONG_ROLE)
        # Epi5 is stored in bits 59..63, not a substitute for role.
        changed_epi5 = replace(self.role, ce64_word=self.role.ce64_word | (9 << 59))
        self.assertIsNone(admit(rule, changed_epi5))
        self.assertEqual(
            admit(rule, replace(changed_epi5, ce64_word=core.ce64_word | (9 << 59))),
            Refusal.WRONG_ROLE,
        )

    def test_coal_twin_for_fixed_180_degree_turn(self) -> None:
        rule = Rule.RIVER_EXTRA_TURN
        self.assertIsNone(admit(rule, self.river))
        self.assertEqual(
            admit(rule, replace(self.river, coal=0)),
            Refusal.INSUFFICIENT_COAL,
        )
        self.assertEqual(
            admit(rule, replace(self.river, coal=1)),
            Refusal.INSUFFICIENT_COAL,
        )
        self.assertIsNone(admit(rule, replace(self.river, coal=2)))
        self.assertEqual(
            admit(rule, replace(self.river, ruleset="unknown")),
            Refusal.UNKNOWN_RULESET,
        )

    def test_source_revision_and_class_reading_are_part_of_meaning(self) -> None:
        for rule, positive in [
            (Rule.RAILS_INCOMPLETE_COUNT, self.rails),
            (Rule.CE64_RECIPIENT_SELECTION, self.role),
            (Rule.RIVER_EXTRA_TURN, self.river),
        ]:
            with self.subTest(rule=rule):
                self.assertIsNone(admit(rule, positive))
                self.assertEqual(
                    admit(rule, replace(positive, source_revision=8)),
                    Refusal.STALE_SOURCE,
                )
                self.assertEqual(
                    admit(rule, replace(positive, class_reading="wrong")),
                    Refusal.WRONG_CLASS_READING,
                )
                self.assertEqual(
                    admit(rule, replace(positive, witnessed=False)),
                    Refusal.UNWITNESSED,
                )

    def test_foreign_rule_domain_is_not_an_alias(self) -> None:
        self.assertEqual(
            admit(Rule.CE64_RECIPIENT_SELECTION, self.rails),
            Refusal.WRONG_DOMAIN,
        )
        self.assertEqual(
            admit(Rule.RAILS_INCOMPLETE_COUNT, self.river),
            Refusal.WRONG_DOMAIN,
        )

    def test_mutant_prefix_only_admission_is_falsified(self) -> None:
        # A deliberately broken implementation admits a rule solely because
        # the left-corner byte matches. The fixture MUST distinguish it.
        def broken_prefix_only(first: int, rule: Rule) -> bool:
            return rule in candidates(first)

        bad_twins = [
            (Rule.RAILS_INCOMPLETE_COUNT, replace(self.rails, receiver="Audit::Item")),
            (Rule.CE64_RECIPIENT_SELECTION, replace(self.role, ce64_word=0x00123456)),
            (Rule.RIVER_EXTRA_TURN, replace(self.river, coal=0)),
        ]
        for rule, bad in bad_twins:
            with self.subTest(rule=rule):
                self.assertTrue(broken_prefix_only(LEFT_CORNER, rule))
                self.assertIsNotNone(admit(rule, bad))


if __name__ == "__main__":
    unittest.main(verbosity=2)
