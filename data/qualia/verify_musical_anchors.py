#!/usr/bin/env python3
"""Read-only schema and identity check for the archived musical anchors.

Usage: python3 data/qualia/verify_musical_anchors.py
No dependencies, no files modified.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
with (ROOT / "calibrated_219.json").open(encoding="utf-8") as f:
    calibrated = json.load(f)
with (ROOT / "calibrated_219.musical_anchors.json").open(encoding="utf-8") as f:
    original = json.load(f)

assert calibrated.get("schema") == "ada.qualia.calibrated/1"
assert calibrated.get("count") == 219
assert len(calibrated["axes_17d"]) == 17
assert len(calibrated["entries"]) == len(original) == 219

def by_id(rows):
    result = {}
    for row in rows:
        assert "id" in row, "missing id"
        assert row["id"] not in result, f"duplicate id: {row['id']}"
        result[row["id"]] = row
    return result

a = by_id(calibrated["entries"])
b = by_id(original)
assert set(a) == set(b), f"ID set difference: {set(a) ^ set(b)}"

for ident, src in b.items():
    dst = a[ident]
    for field in ("label", "family", "subfamily"):
        assert src[field] == dst[field], f"{ident}: {field} differs"
    assert src["vector"] == dst["vector_17d"], f"{ident}: 17D vector differs"
    assert set(src["vector"]) == set(calibrated["axes_17d"]), f"{ident}: axes differ"
    assert src["qualia"] == dst["qualia_tags"], f"{ident}: qualia tags differ"
    for field in ("melodic_motions", "harmonic_bias", "truth_anchor_candidates", "neighbors"):
        assert isinstance(src[field], list), f"{ident}: missing {field}"

print("OK: 219/219 IDs and primary fields identical; musical annotations retained.")
