"""Version-address C-01/C-02 falsifiers and independent reference controls."""

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
        self.assertEqual(readings[0]["coreference_status"], "VersionWitnessed")
        self.assertEqual(readings[0]["witness_coordinates"], (
            ("corefers(last_sentence.key,first_sentence.key)", "symbolic-ring", "main", 17),
        ))
        self.assertEqual(readings[0]["source_roots"], ("independent_coref_witness_1",))

    def test_pre_witness_version_remains_open_under_every_rotation(self):
        for order in self.negative["input"]["entry_orders"]:
            with self.subTest(order=order):
                result = evaluate(self.negative, self.gold, order)
                self.assertFalse(result["cycle_closed"])
                self.assertEqual(result["coreference_status"], "Unresolved")
                self.assertEqual(result["witness_coordinates"], ())
                self.assertEqual(len(result["relations"]), 4)

    def test_a_later_version_closes_same_C02_without_rewriting_identities(self):
        future = {"dataset": "symbolic-ring", "branch": "main", "version": 17}
        result = self.run_case(self.negative, read_ref=future)
        baseline = self.run_case(self.positive)
        self.assertTrue(result["cycle_closed"])
        self.assertEqual(result["relations"], baseline["relations"])
        self.assertEqual(result["witness_coordinates"], baseline["witness_coordinates"])

    def test_legacy_external_witness_field_does_not_control_admission(self):
        candidate = copy.deepcopy(self.positive)
        candidate["input"]["external_identity_witness"]["root"] = "spoofed-root"
        self.assertTrue(self.run_case(candidate)["cycle_closed"])
        self.assertEqual(self.run_case(candidate)["source_roots"], (
            "independent_coref_witness_1",))

    def test_declaring_a_root_does_not_bypass_older_read_horizon(self):
        old = {"dataset": "symbolic-ring", "branch": "main", "version": 16}
        self.assertFalse(self.run_case(self.positive, read_ref=old)["cycle_closed"])

    def test_dataset_or_branch_mismatch_cannot_witness_identity(self):
        for ref in (
            {"dataset": "other-dataset", "branch": "main", "version": 17},
            {"dataset": "symbolic-ring", "branch": "other-branch", "version": 17},
        ):
            with self.subTest(ref=ref):
                self.assertFalse(self.run_case(self.positive, read_ref=ref)["cycle_closed"])

    def test_wrong_claim_address_is_refused(self):
        gold = copy.deepcopy(self.gold)
        gold["attested_identity"][0]["claim_address"] = "invented.witness.id"
        with self.assertRaisesRegex(ValueError, "wrong claim address"):
            evaluate(self.positive, gold, self.positive["input"]["entry_orders"][0])

    def test_missing_or_unbounded_version_ref_is_refused(self):
        for ref in ({"dataset": "symbolic-ring", "branch": "main"},
                    {"dataset": "symbolic-ring", "branch": "main", "version": -1},
                    {"dataset": "symbolic-ring", "branch": "main", "version": None}):
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "reference"):
                self.run_case(self.positive, read_ref=ref)

    def test_missing_last_claim_breaks_cycle_even_with_witness(self):
        self.assertFalse(self.run_case(self.positive,
                                       facts=copy.deepcopy(self.gold["facts"][:-1]))["cycle_closed"])

    def test_changed_predicate_changes_semantics_not_only_topology(self):
        baseline = self.run_case(self.positive)
        altered = copy.deepcopy(self.gold["facts"])
        altered[-1]["predicate"] = "destroys"
        mutant = self.run_case(self.positive, facts=altered)
        self.assertTrue(mutant["cycle_closed"], "unlabeled topology still cycles")
        self.assertNotEqual(mutant["relations"], baseline["relations"])

    def test_different_last_referent_stays_open(self):
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

    def test_expected_fields_do_not_drive_oracle(self):
        candidate = copy.deepcopy(self.negative)
        candidate["expected"]["cycle_closed"] = True
        self.assertFalse(self.run_case(candidate)["cycle_closed"])

    def test_source_roots_are_evidential_metadata_not_witness_identity(self):
        gold = copy.deepcopy(self.gold)
        gold["attested_identity"][0]["source_roots"] = [
            "source-A", "source-A", "source-B",
        ]
        result = evaluate(self.positive, gold, self.positive["input"]["entry_orders"][0])
        self.assertTrue(result["cycle_closed"])
        self.assertEqual(result["source_roots"], ("source-A", "source-B"))
        self.assertEqual(result["witness_coordinates"],
                         self.run_case(self.positive)["witness_coordinates"])


if __name__ == "__main__":
    unittest.main()
