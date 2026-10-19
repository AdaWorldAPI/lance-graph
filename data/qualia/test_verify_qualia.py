"""Negative controls for the two qualia verifiers.

Run as `python3 -O -m unittest discover -s data/qualia -p 'test_*.py'`: the
`-O` flag strips `assert`, so these tests also prove the verifiers do not rely
on it. Each test mutates an in-memory copy; no file is written.
"""
import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


anchors_v = load("verify_musical_anchors")
enrich_v = load("verify_music_enrichment")
CAL = json.loads((ROOT / "calibrated_219.json").read_text(encoding="utf-8"))
ANCHORS = json.loads((ROOT / "calibrated_219.musical_anchors.json").read_text(encoding="utf-8"))
ENRICHED = json.loads((ROOT / "calibrated_219.music_enriched.json").read_text(encoding="utf-8"))


class AnchorsVerifier(unittest.TestCase):
    def test_committed_archive_passes(self):
        anchors_v.verify(CAL, ANCHORS)

    def test_altered_musical_value_is_refused(self):
        bad = copy.deepcopy(ANCHORS)
        bad[0]["harmonic_bias"][0] = bad[0]["harmonic_bias"][0] + " (edited)"
        with self.assertRaises(anchors_v.VerificationError):
            anchors_v.verify(CAL, bad)

    def test_emptied_musical_field_is_refused(self):
        bad = copy.deepcopy(ANCHORS)
        bad[5]["neighbors"] = []
        with self.assertRaises(anchors_v.VerificationError):
            anchors_v.verify(CAL, bad)

    def test_label_mismatch_is_refused(self):
        bad = copy.deepcopy(ANCHORS)
        bad[3]["label"] = "not the label"
        with self.assertRaises(anchors_v.VerificationError):
            anchors_v.verify(CAL, bad)


class EnrichmentVerifier(unittest.TestCase):
    def test_committed_sidecar_passes(self):
        enrich_v.verify(CAL, ENRICHED, ANCHORS)

    def test_well_formed_but_different_value_is_refused(self):
        bad = copy.deepcopy(ENRICHED)
        bad["entries"][7]["melodic_motions"] = ["a plausible but wrong motion"]
        with self.assertRaises(enrich_v.VerificationError):
            enrich_v.verify(CAL, bad, ANCHORS)

    def test_empty_sidecar_is_refused(self):
        bad = copy.deepcopy(ENRICHED)
        bad["entries"] = []
        with self.assertRaises(enrich_v.VerificationError):
            enrich_v.verify(CAL, bad, ANCHORS)


if __name__ == "__main__":
    unittest.main()
