"""C-01/C-02 falsifiers, including silence and mutation controls."""

import copy
import unittest

from ring_reference import evaluate, load_case, load_gold


class RingReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = load_gold()
        cls.positive = load_case("C-01")
        cls.negative = load_case("C-02")

    def run_case(self, case, **kw):
        return evaluate(case, self.gold, case["input"]["entry_orders"][0], **kw)

    def test_witnessed_closure_is_rotation_invariant(self):
        readings = [evaluate(self.positive, self.gold, order)
                    for order in self.positive["input"]["entry_orders"]]
        self.assertTrue(all(x == readings[0] for x in readings))
        self.assertTrue(readings[0]["cycle_closed"])
        self.assertEqual(readings[0]["witness_roots"], ("independent_coref_witness_1",))
        self.assertEqual(len(readings[0]["relations"]), 4)

    def test_unwitnessed_ring_remains_open_under_every_rotation(self):
        for order in self.negative["input"]["entry_orders"]:
            with self.subTest(order=order):
                result = evaluate(self.negative, self.gold, order)
                self.assertFalse(result["cycle_closed"])
                self.assertEqual(result["coreference_status"], "Unresolved")
                self.assertEqual(result["witness_roots"], ())
                self.assertEqual(len(result["relations"]), 4)

    def test_inactive_attestation_is_not_implicit(self):
        candidate = copy.deepcopy(self.positive)
        candidate["input"]["external_identity_witness"] = None
        self.assertFalse(self.run_case(candidate)["cycle_closed"])

    def test_spoofed_root_does_not_close_cycle(self):
        candidate = copy.deepcopy(self.positive)
        candidate["input"]["external_identity_witness"]["root"] = "copied_fake_witness"
        with self.assertRaisesRegex(ValueError, "unattested"):
            self.run_case(candidate)

    def test_wrong_identity_endpoints_are_refused(self):
        candidate = copy.deepcopy(self.positive)
        candidate["input"]["external_identity_witness"]["rhs"] = "door"
        with self.assertRaisesRegex(ValueError, "unattested"):
            self.run_case(candidate)

    def test_missing_last_claim_breaks_cycle_even_with_witness(self):
        truncated = copy.deepcopy(self.gold["facts"][:-1])
        self.assertFalse(self.run_case(self.positive, facts=truncated)["cycle_closed"])

    def test_changed_predicate_changes_semantics_not_only_topology(self):
        baseline = self.run_case(self.positive)
        altered = copy.deepcopy(self.gold["facts"])
        altered[-1]["predicate"] = "destroys"
        mutant = self.run_case(self.positive, facts=altered)
        self.assertTrue(mutant["cycle_closed"], "unlabeled geometry still cycles")
        self.assertNotEqual(mutant["relations"], baseline["relations"],
                            "different predicates cannot retain claim identity")

    def test_different_last_referent_does_not_close_with_unrelated_witness(self):
        changed = copy.deepcopy(self.gold["facts"])
        changed[-1]["object"] = "another.key"
        self.assertFalse(self.run_case(self.positive, facts=changed)["cycle_closed"])

    def test_duplicate_source_binding_refused(self):
        doubled = copy.deepcopy(self.gold["facts"])
        doubled[1]["sentence_index"] = 0
        with self.assertRaisesRegex(ValueError, "multiple atoms"):
            self.run_case(self.positive, facts=doubled)

    def test_unregistered_rotation_refused(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            evaluate(self.positive, self.gold, [3, 2, 1, 0])

    def test_fixture_expected_fields_do_not_drive_oracle(self):
        candidate = copy.deepcopy(self.negative)
        candidate["expected"]["cycle_closed"] = True
        self.assertFalse(self.run_case(candidate)["cycle_closed"])


if __name__ == "__main__":
    unittest.main()
