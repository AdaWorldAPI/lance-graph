"""Cold P0 probe: u8 candidate selection is not semantic admission.

Pure stdlib, deliberately outside the Rust production crates. Synthetic rules are
not OGAR codebook entries. No Rails, CE64, game, or R2IL execution is claimed.
Run: python3 -m unittest -v logic_neutrino_p0.py
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import unittest

SOURCE_REV = 9
CLASSVIEW_REV = 3
ORIENTATION_SHIFT = 43
EPI_SHIFT = 59
ROLE_RECIPIENT = 2


@dataclass(frozen=True)
class Rule:
    family: str
    position: int  # u8 scoped to family, NEVER a global opcode or recipe ID
    prefix: str
    receiver: str = ""
    min_coal: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.position <= 255:
            raise ValueError("rule position must be u8")


# Illustrative read-only candidate index, not a copy of any real registry.
RULES = (
    Rule("rails", 1, "items.incomplete.count", receiver="Checklist"),
    Rule("rails", 2, "items.incomplete.count", receiver="Ticket"),
    Rule("ce64", 1, "relation", receiver="Recipient"),
    Rule("mq", 1, "navigate", min_coal=2),
)


@dataclass(frozen=True)
class Observed:
    family: str
    prefix: str
    receiver: str = ""
    edge_word: int = 0
    coal: int = 0
    source_rev: int = SOURCE_REV
    classview_rev: int = CLASSVIEW_REV


def candidates(event: Observed) -> tuple[Rule, ...]:
    """S2: only a family-qualified left-corner/prefix lookup; NO admission."""
    return tuple(r for r in RULES if r.family == event.family and r.prefix == event.prefix)


def refusal(event: Observed, rule: Rule, *, skip: str = "") -> str | None:
    """S3: independent typed checks. Never performs a DO effect."""
    if rule not in candidates(event):
        return "not-a-candidate"
    if skip != "revision" and (event.source_rev != SOURCE_REV or event.classview_rev != CLASSVIEW_REV):
        return "stale-witness"
    if event.family == "rails" and skip != "receiver" and event.receiver != rule.receiver:
        return "receiver-mismatch"
    if event.family == "ce64":
        # Epi5 6,7 are reserved certification codes; do not rebind them.
        if ((event.edge_word >> EPI_SHIFT) & 0b11111) >= 24:
            return "reserved-epistemic-reading"
        care = 0b111 << ORIENTATION_SHIFT
        pattern = ROLE_RECIPIENT << ORIENTATION_SHIFT
        if skip != "orientation" and (event.edge_word ^ pattern) & care:
            return "orientation-mismatch"
    if event.family == "mq" and skip != "coal" and event.coal < rule.min_coal:
        return "resource-insufficient"
    return None


def guarded_candidates(event: Observed, *, skip: str = "") -> tuple[tuple[str, int], ...]:
    return tuple((r.family, r.position) for r in candidates(event) if refusal(event, r, skip=skip) is None)


def independent_oracle(event: Observed) -> tuple[tuple[str, int], ...]:
    """Procedural fixture oracle; does not consult RULES, candidates or refusal."""
    if event.source_rev != 9 or event.classview_rev != 3:
        return ()
    if event.family == "rails" and event.prefix == "items.incomplete.count":
        if event.receiver == "Checklist":
            return (("rails", 1),)
        if event.receiver == "Ticket":
            return (("rails", 2),)
    if event.family == "ce64" and event.prefix == "relation":
        orientation = (event.edge_word // (2**43)) % 8
        epistemic_code = (event.edge_word // (2**59)) % 32
        if orientation == 2 and epistemic_code < 24:
            return (("ce64", 1),)
    if event.family == "mq" and event.prefix == "navigate" and event.coal >= 2:
        return (("mq", 1),)
    return ()


def edge(role: int, epi: int = 0) -> int:
    """Minimal packed fixture: only CE64 orientation + Epi5 bits are set."""
    assert 0 <= role < 8 and 0 <= epi < 32
    return (role << ORIENTATION_SHIFT) | (epi << EPI_SHIFT)


class P0CandidateGate(unittest.TestCase):
    def test_exhaustive_small_matrix_matches_independent_oracle(self) -> None:
        observations = [
            Observed("rails", "items.incomplete.count", receiver=receiver, source_rev=source, classview_rev=cv)
            for receiver, source, cv in itertools.product(("Checklist", "Ticket", "Unknown"), (9, 10), (3, 4))
        ]
        observations += [
            Observed("ce64", "relation", edge_word=edge(role, epi), source_rev=source, classview_rev=cv)
            for role, epi, source, cv in itertools.product(range(8), range(32), (9, 10), (3, 4))
        ]
        observations += [
            Observed("mq", "navigate", coal=coal, source_rev=source, classview_rev=cv)
            for coal, source, cv in itertools.product((0, 1, 2, 6), (9, 10), (3, 4))
        ]
        for observed in observations:
            with self.subTest(family=observed.family, receiver=observed.receiver, word=observed.edge_word, coal=observed.coal, source=observed.source_rev, classview=observed.classview_rev):
                self.assertEqual(guarded_candidates(observed), independent_oracle(observed))
        self.assertEqual(len(observations), 12 + 1024 + 16)

    def test_one_byte_is_not_an_identity_or_admission(self) -> None:
        self.assertEqual(RULES[0].position, RULES[2].position)
        self.assertNotEqual(RULES[0].family, RULES[2].family)
        ambiguous = Observed("rails", "items.incomplete.count", receiver="Checklist")
        self.assertEqual(len(candidates(ambiguous)), 2)
        self.assertEqual(guarded_candidates(ambiguous), (("rails", 1),))
        with self.assertRaises(ValueError):
            Rule("bad", 256, "anything")

    def test_epistemic_topology_cannot_change_orientation(self) -> None:
        for epi in range(24):
            self.assertEqual(guarded_candidates(Observed("ce64", "relation", edge_word=edge(2, epi))), (("ce64", 1),))
            self.assertEqual(guarded_candidates(Observed("ce64", "relation", edge_word=edge(0, epi))), ())
        self.assertEqual(refusal(Observed("ce64", "relation", edge_word=edge(2, 24)), RULES[2]), "reserved-epistemic-reading")

    def test_fail_closed_reason_is_specific(self) -> None:
        cases = (
            (Observed("rails", "items.incomplete.count", receiver="Unknown"), RULES[0], "receiver-mismatch"),
            (Observed("rails", "items.incomplete.count", receiver="Checklist", classview_rev=4), RULES[0], "stale-witness"),
            (Observed("ce64", "relation", edge_word=edge(0)), RULES[2], "orientation-mismatch"),
            (Observed("mq", "navigate", coal=0), RULES[3], "resource-insufficient"),
        )
        for event, rule, expected in cases:
            self.assertEqual(refusal(event, rule), expected)

    def test_ablation_each_guard_is_necessary(self) -> None:
        bad = (
            (Observed("rails", "items.incomplete.count", receiver="Checklist"), "receiver"),
            (Observed("rails", "items.incomplete.count", receiver="Checklist", source_rev=10), "revision"),
            (Observed("ce64", "relation", edge_word=edge(0)), "orientation"),
            (Observed("mq", "navigate", coal=0), "coal"),
        )
        for event, removed_guard in bad:
            self.assertNotEqual(guarded_candidates(event, skip=removed_guard), independent_oracle(event), removed_guard)


if __name__ == "__main__":
    unittest.main()
