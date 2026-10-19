#!/usr/bin/env python3
"""Read-only schema and identity check for the archived musical anchors.

Usage: python3 data/qualia/verify_musical_anchors.py
No dependencies, no files modified.

Every check raises explicitly instead of using `assert`, so it still runs
under `python -O` / `PYTHONOPTIMIZE`, which strips assert statements.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MUSICAL_FIELDS = ("melodic_motions", "harmonic_bias", "truth_anchor_candidates", "neighbors")

# SHA-256 of the four archival fields of all 219 records, as canonical JSON
# ({id: {field: value}}, sorted keys, compact separators, UTF-8). These fields
# are this file's unique archival content, so a type check alone would accept
# an altered or emptied annotation. Recompute only when the archive is
# deliberately replaced, and say so in the commit.
MUSICAL_FIELDS_SHA256 = "027f6753cbe17f14109d45aefb679b1baf363e5333a87cb720f112d3c49147dc"


class VerificationError(Exception):
    pass


def check(condition, message):
    if not condition:
        raise VerificationError(message)


def by_id(rows):
    result = {}
    for row in rows:
        check("id" in row, "missing id")
        check(row["id"] not in result, f"duplicate id: {row['id']}")
        result[row["id"]] = row
    return result


def musical_digest(rows):
    payload = {ident: {f: row[f] for f in MUSICAL_FIELDS} for ident, row in rows.items()}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify(calibrated, original):
    check(calibrated.get("schema") == "ada.qualia.calibrated/1", "unexpected schema")
    check(calibrated.get("count") == 219, "count is not 219")
    check(len(calibrated["axes_17d"]) == 17, "axes_17d is not 17 long")
    check(len(calibrated["entries"]) == len(original) == 219, "entry counts differ from 219")

    a = by_id(calibrated["entries"])
    b = by_id(original)
    check(set(a) == set(b), f"ID set difference: {set(a) ^ set(b)}")

    for ident, src in b.items():
        dst = a[ident]
        for field in ("label", "family", "subfamily"):
            check(src[field] == dst[field], f"{ident}: {field} differs")
        check(src["vector"] == dst["vector_17d"], f"{ident}: 17D vector differs")
        check(set(src["vector"]) == set(calibrated["axes_17d"]), f"{ident}: axes differ")
        check(src["qualia"] == dst["qualia_tags"], f"{ident}: qualia tags differ")
        for field in MUSICAL_FIELDS:
            values = src.get(field)
            check(isinstance(values, list) and values, f"{ident}: {field} missing or empty")
            check(
                all(isinstance(v, str) and v.strip() for v in values),
                f"{ident}: {field} has a blank or non-string value",
            )

    digest = musical_digest(b)
    check(
        digest == MUSICAL_FIELDS_SHA256,
        f"musical annotations changed: sha256 {digest} != pinned {MUSICAL_FIELDS_SHA256}",
    )


def main():
    with (ROOT / "calibrated_219.json").open(encoding="utf-8") as f:
        calibrated = json.load(f)
    with (ROOT / "calibrated_219.musical_anchors.json").open(encoding="utf-8") as f:
        original = json.load(f)
    try:
        verify(calibrated, original)
    except VerificationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("OK: 219/219 IDs and primary fields identical; musical annotations retained and match the pinned digest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
