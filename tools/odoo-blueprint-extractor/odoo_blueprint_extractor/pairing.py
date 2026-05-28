"""Curated-vs-extracted pairing scanner (D-ODOO-EXT-5).

Walks ``l{1..15}.rs`` curated lane modules and ``extracted/*.rs`` source-
extracted modules under the given odoo_blueprint source directory, parses
``OdooEntity`` const declarations via stdlib ``re``, and builds a cross-
reference table: one ``OdooEntityPairing`` row per ``model_name`` that
appears in BOTH the curated set AND the extracted set.

Public API
----------
- :func:`scan_blueprint_dir` — scan both sides, return pairing data.
- :func:`emit_pairing_rs`    — render the pairing table as a Rust source file.
- :func:`emit_audit_json`    — render the mismatch audit as JSON.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_block_end(src: str, start: int) -> int:
    """Return the index of the closing ``}`` for the brace-delimited block
    whose opening ``{`` is at *start*.  Returns -1 if not found."""
    depth = 0
    i = start
    n = len(src)
    while i < n:
        c = src[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


# Matches both ``pub const NAME: OdooEntity = OdooEntity {`` and the
# module-private ``const NAME: OdooEntity = OdooEntity {`` forms used in
# several curated lane files (l3.rs, l5.rs, l7.rs, …).
_CONST_RE = re.compile(r'(?:pub\s+)?const (\w+): OdooEntity = OdooEntity \{')


def _scan_file(path: Path, source_tag: str) -> dict[str, list[dict[str, Any]]]:
    """Return ``{model_name: [{file, const_name, field_count, method_count}]}``
    for all ``OdooEntity`` const declarations found in *path*.

    *source_tag* is ``"curated"`` or ``"extracted"`` and is stored for
    downstream selection logic.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    src = path.read_text(encoding="utf-8")

    for m in _CONST_RE.finditer(src):
        const_name = m.group(1)
        # Skip the ENTITIES slice const — it's just an aggregator, not an entity.
        if const_name == "ENTITIES":
            continue

        brace_start = m.end() - 1  # points at the opening ``{``
        block_end = _find_block_end(src, brace_start)
        if block_end == -1:
            continue  # malformed — skip

        block = src[m.start() : block_end + 1]

        mn_match = re.search(r'model_name:\s*"([^"]+)"', block)
        if not mn_match:
            continue

        model_name = mn_match.group(1)
        field_count = len(re.findall(r'OdooField \{', block))
        method_count = len(re.findall(r'OdooMethod \{', block))

        results.setdefault(model_name, []).append(
            {
                "source": source_tag,
                "file": path.name,
                "file_stem": path.stem,
                "const_name": const_name,
                "field_count": field_count,
                "method_count": method_count,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------

#: Names of extracted files that do NOT contain ``OdooEntity`` consts and
#: must be skipped (they use their own typed surfaces instead).
_SKIP_EXTRACTED = frozenset(
    {
        "l10n_de_chart.rs",       # OdooAccountTemplate / OdooSkrChart
        "l10n_de_kennzahlen.rs",  # OdooUstvaKennzahl / OdooKennzahlKind
        "mod.rs",
        "pairing.rs",             # this file itself (guard against re-scan)
    }
)


def scan_blueprint_dir(blueprint_dir: Path) -> dict[str, Any]:
    """Scan curated lane modules and extracted modules under *blueprint_dir*.

    *blueprint_dir* should point to
    ``crates/lance-graph-ontology/src/odoo_blueprint/``.

    Returns a dict::

        {
          "curated":   {model_name: [{file, file_stem, const_name, field_count, method_count}]},
          "extracted": {model_name: [{file, file_stem, const_name, field_count, method_count}]},
        }
    """
    # -- Scan curated lane modules (l1.rs … l15.rs) --
    curated: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(blueprint_dir.glob("l*.rs")):
        for mn, entries in _scan_file(f, "curated").items():
            curated.setdefault(mn, []).extend(entries)

    # -- Scan extracted modules --
    extracted: dict[str, list[dict[str, Any]]] = {}
    ext_dir = blueprint_dir / "extracted"
    for f in sorted(ext_dir.glob("*.rs")):
        if f.name in _SKIP_EXTRACTED:
            continue
        for mn, entries in _scan_file(f, "extracted").items():
            extracted.setdefault(mn, []).extend(entries)

    return {"curated": curated, "extracted": extracted}


# ---------------------------------------------------------------------------
# Pairing selection logic
# ---------------------------------------------------------------------------

def _best_curated(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the canonical curated entry for a given *model_name*.

    Rules (in priority order):
    1. Prefer the entry with the most ``field_count + method_count`` (handles
       files that use indirect const references — those count as 0 inline).
    2. On a tie: alphabetically first ``const_name``.

    Most model_names have exactly one curated entry; extension fragments in
    different files share the model_name but have distinct const names.
    """
    return sorted(
        entries,
        key=lambda e: (-(e["field_count"] + e["method_count"]), e["const_name"]),
    )[0]


def _best_extracted(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the canonical extracted entry for a given *model_name*.

    Rule: entry with the most ``field_count + method_count`` (richest coverage).
    On a tie, pick alphabetically-first ``file`` then alphabetically-first
    ``const_name``.
    """
    return sorted(
        entries,
        key=lambda e: (-(e["field_count"] + e["method_count"]), e["file"], e["const_name"]),
    )[0]


def build_pairings(scan: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the sorted list of pairing records from a *scan* result.

    Returns a list of dicts sorted by *model_name*::

        [
          {
            "model_name": "account.account",
            "curated":   {"file": "l11.rs", "file_stem": "l11",
                          "const_name": "ACCOUNT_ACCOUNT",
                          "field_count": 9, "method_count": 4},
            "extracted": {"file": "account.rs", "file_stem": "account",
                          "const_name": "EXT_ACCOUNT_ACCOUNT",
                          "field_count": 28, "method_count": 71},
            "deltas":    {"field_delta": 19, "method_delta": 67,
                          "note": "curated is a savant-relevant subset; extracted is full ORM"},
          },
          …
        ]
    """
    curated = scan["curated"]
    extracted = scan["extracted"]
    overlap = sorted(set(curated) & set(extracted))

    pairings = []
    for mn in overlap:
        cur = _best_curated(curated[mn])
        ext = _best_extracted(extracted[mn])
        field_delta = ext["field_count"] - cur["field_count"]
        method_delta = ext["method_count"] - cur["method_count"]
        pairings.append(
            {
                "model_name": mn,
                "curated": cur,
                "extracted": ext,
                "deltas": {
                    "field_delta": field_delta,
                    "method_delta": method_delta,
                    "note": "curated is a savant-relevant subset; extracted is full ORM",
                },
            }
        )
    return pairings


# ---------------------------------------------------------------------------
# Rust emitter
# ---------------------------------------------------------------------------

def _rust_ref(entry: dict[str, Any], side: str) -> str:
    """Build the absolute Rust path for a pairing entry reference.

    ``pairing.rs`` lives inside ``extracted/`` and uses
    ``use crate::odoo_blueprint::*;``.  The lane consts live at
    ``crate::odoo_blueprint::<stem>::<CONST>`` and the extracted consts live at
    ``crate::odoo_blueprint::extracted::<stem>::<CONST>``.
    """
    stem = entry["file_stem"]
    const = entry["const_name"]
    if side == "curated":
        return f"&crate::odoo_blueprint::{stem}::{const}"
    else:
        return f"&crate::odoo_blueprint::extracted::{stem}::{const}"


def emit_pairing_rs(pairings: list[dict[str, Any]]) -> str:
    """Render the pairing table as a complete ``pairing.rs`` Rust source file."""
    n = len(pairings)

    lines = [
        "//! Auto-generated curated-vs-extracted pairing table (D-ODOO-EXT-5).",
        "//!",
        "//! For every `model_name` that appears in BOTH a curated lane module",
        "//! (`l{1..15}`) AND a source-extracted module (`extracted::*`), this",
        "//! file records the `(curated_const, extracted_const)` reference pair.",
        "//!",
        "//! Curated stays canonical on conflict (per `odoo-business-logic-blueprint-v1`",
        "//! §\"merge ordering\"). Mismatches (field/method count deltas) are",
        "//! recorded out-of-tree in `/tmp/pairings.json` for human review.",
        "//! Plan: `.claude/plans/odoo-source-extraction-v1.md`.",
        "",
        "use crate::odoo_blueprint::*;",
        "",
        "/// One pairing: a model_name that has both a human-curated lane entity",
        "/// (`OdooConfidence::Curated`) and at least one source-extracted entity",
        "/// (`OdooConfidence::Extracted`).",
        "#[derive(Debug, Clone, Copy)]",
        "pub struct OdooEntityPairing {",
        "    pub model_name: &'static str,",
        "    /// Pointer to the curated lane const (canonical reference).",
        "    pub curated: &'static OdooEntity,",
        "    /// Pointer to the extracted const (the source-truth backing).",
        "    pub extracted: &'static OdooEntity,",
        "}",
        "",
        "pub static CURATED_EXTRACTED_PAIRS: &[OdooEntityPairing] = &[",
    ]

    for p in pairings:
        mn = p["model_name"]
        cur = p["curated"]
        ext = p["extracted"]
        cur_ref = _rust_ref(cur, "curated")
        ext_ref = _rust_ref(ext, "extracted")
        field_delta = p["deltas"]["field_delta"]
        method_delta = p["deltas"]["method_delta"]
        lines += [
            f"    // {mn}  (curated: {cur['const_name']} in {cur['file']}"
            f"  |  extracted: {ext['const_name']} in {ext['file']})",
            f"    // delta: fields={field_delta:+d}, methods={method_delta:+d}",
            "    OdooEntityPairing {",
            f'        model_name: "{mn}",',
            f"        curated: {cur_ref},",
            f"        extracted: {ext_ref},",
            "    },",
        ]

    lines += [
        "];",
        "",
        "#[cfg(test)]",
        "mod tests {",
        "    use super::*;",
        "    use crate::odoo_blueprint::OdooConfidence;",
        "",
        "    #[test]",
        "    fn pairing_table_is_well_formed() {",
        "        for pair in CURATED_EXTRACTED_PAIRS {",
        "            assert_eq!(",
        "                pair.model_name,",
        "                pair.curated.model_name,",
        "                \"Curated entity model_name mismatch for {}\",",
        "                pair.model_name,",
        "            );",
        "            assert_eq!(",
        "                pair.model_name,",
        "                pair.extracted.model_name,",
        "                \"Extracted entity model_name mismatch for {}\",",
        "                pair.model_name,",
        "            );",
        "            assert_eq!(",
        "                pair.curated.provenance.confidence,",
        "                OdooConfidence::Curated,",
        "                \"Curated confidence wrong for {}\",",
        "                pair.model_name,",
        "            );",
        "            assert_eq!(",
        "                pair.extracted.provenance.confidence,",
        "                OdooConfidence::Extracted,",
        "                \"Extracted confidence wrong for {}\",",
        "                pair.model_name,",
        "            );",
        "        }",
        "    }",
        "",
        "    #[test]",
        "    fn pairing_table_has_expected_size() {",
        f"        // EXT-5 inventory: {n} model_name overlaps across TIER-1.",
        "        // Adjust if the actual count differs; commit body should explain drift.",
        "        assert!(",
        "            CURATED_EXTRACTED_PAIRS.len() >= 40,",
        "            \"Pairing table thinner than expected: {}\",",
        "            CURATED_EXTRACTED_PAIRS.len(),",
        "        );",
        "    }",
        "}",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit JSON emitter
# ---------------------------------------------------------------------------

def emit_audit_json(pairings: list[dict[str, Any]]) -> str:
    """Render the mismatch audit as a JSON string."""
    audit_pairings = []
    for p in pairings:
        audit_pairings.append(
            {
                "model_name": p["model_name"],
                "curated": {
                    "file": p["curated"]["file"],
                    "const_name": p["curated"]["const_name"],
                    "field_count": p["curated"]["field_count"],
                    "method_count": p["curated"]["method_count"],
                },
                "extracted": {
                    "file": "extracted/" + p["extracted"]["file"],
                    "const_name": p["extracted"]["const_name"],
                    "field_count": p["extracted"]["field_count"],
                    "method_count": p["extracted"]["method_count"],
                },
                "deltas": p["deltas"],
            }
        )

    doc = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_pairings": len(pairings),
        "pairings": audit_pairings,
    }
    return json.dumps(doc, indent=2)
