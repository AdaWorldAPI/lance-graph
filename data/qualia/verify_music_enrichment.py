#!/usr/bin/env python3
"""Archival consistency fixture, not production runtime code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
canonical = json.loads((ROOT / "calibrated_219.json").read_text(encoding="utf-8"))
enriched = json.loads((ROOT / "calibrated_219.music_enriched.json").read_text(encoding="utf-8"))
original = canonical["entries"]
extra = enriched["entries"]
assert canonical["count"] == enriched["count"] == len(original) == len(extra) == 219
ids = [x["id"] for x in original]
extra_ids = [x["id"] for x in extra]
assert len(set(ids)) == len(set(extra_ids)) == 219, "duplicate ID"
assert set(ids) == set(extra_ids), "sidecar changes ID inventory"
assert enriched["join_key"] == "id"
for entry in extra:
    assert set(entry) == {"id", "melodic_motions", "harmonic_bias", "truth_anchor_candidates", "neighbors"}
    for field in ("melodic_motions", "harmonic_bias", "truth_anchor_candidates", "neighbors"):
        assert isinstance(entry[field], list) and entry[field], (entry["id"], field)
        assert all(isinstance(v, str) and v.strip() for v in entry[field]), (entry["id"], field)
print("PASS: 219/219 IDs match; musical enrichment is additive and complete.")
