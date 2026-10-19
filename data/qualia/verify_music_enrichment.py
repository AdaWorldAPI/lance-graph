#!/usr/bin/env python3
"""Archival consistency fixture, not production runtime code.

Every check raises explicitly instead of using `assert`, so it still runs
under `python -O` / `PYTHONOPTIMIZE`, which strips assert statements.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ("melodic_motions", "harmonic_bias", "truth_anchor_candidates", "neighbors")


class VerificationError(Exception):
    pass


def check(condition, message):
    if not condition:
        raise VerificationError(message)


def verify(canonical, enriched, anchors):
    original = canonical["entries"]
    extra = enriched["entries"]
    check(
        canonical["count"] == enriched["count"] == len(original) == len(extra) == 219,
        "counts differ from 219",
    )
    ids = [x["id"] for x in original]
    extra_ids = [x["id"] for x in extra]
    check(len(set(ids)) == len(set(extra_ids)) == 219, "duplicate ID")
    check(set(ids) == set(extra_ids), "sidecar changes ID inventory")
    check(enriched["join_key"] == "id", "join_key is not id")
    by_id = {a["id"]: a for a in anchors}
    for entry in extra:
        check(set(entry) == {"id", *FIELDS}, f"{entry.get('id')}: unexpected keys {sorted(entry)}")
        for field in FIELDS:
            values = entry[field]
            check(isinstance(values, list) and values, f"{entry['id']}: {field} missing or empty")
            check(
                all(isinstance(v, str) and v.strip() for v in values),
                f"{entry['id']}: {field} has a blank or non-string value",
            )
            # The sidecar must carry exactly the archived annotations, not
            # merely well-formed lists.
            check(
                values == by_id[entry["id"]][field],
                f"{entry['id']}: {field} differs from calibrated_219.musical_anchors.json",
            )


def main():
    canonical = json.loads((ROOT / "calibrated_219.json").read_text(encoding="utf-8"))
    enriched = json.loads((ROOT / "calibrated_219.music_enriched.json").read_text(encoding="utf-8"))
    anchors = json.loads((ROOT / "calibrated_219.musical_anchors.json").read_text(encoding="utf-8"))
    try:
        verify(canonical, enriched, anchors)
    except VerificationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("PASS: 219/219 IDs match; musical enrichment is additive, complete and equal to the archived annotations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
