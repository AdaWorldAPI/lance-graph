"""Smoke test: parse uom addon and assert the emitted Rust is well-formed.

Run with:
    python tests/test_smoke_uom.py
or:
    python -m pytest tests/ -v

Assertions:
1. Emitted Rust contains `pub const EXT_UOM_UOM: OdooEntity`
2. Contains `model_name: "uom.uom"`
3. Contains `kind: OdooEntityKind::Model`
4. Contains `confidence: OdooConfidence::Extracted`
5. Output is syntactically plausible (balanced braces)
6. Fallback rate <= 5%
"""

import sys
import os

# Allow running from anywhere by inserting the package parent to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from pathlib import Path

from odoo_blueprint_extractor.audit.fallback_log import FallbackLog
from odoo_blueprint_extractor.parsers.classes import parse_addon
from odoo_blueprint_extractor.emitters.module import emit_module


ADDONS_ROOT = Path("/home/user/odoo/addons")
ADDON_NAME = "uom"
ADDON_DIR = ADDONS_ROOT / ADDON_NAME


def _count_braces(text: str):
    """Return (open_count, close_count) for curly braces."""
    return text.count("{"), text.count("}")


def run_tests() -> bool:
    """Run all smoke tests; return True if all pass, False otherwise."""
    failures = []

    # ── Parse ──────────────────────────────────────────────────────────────
    if not ADDON_DIR.is_dir():
        print(f"SKIP: addon directory not found: {ADDON_DIR}")
        return True  # Not a failure on CI without Odoo source

    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)

    if not entities:
        failures.append("No entities parsed from uom addon")
    else:
        rust = emit_module(entities, ADDON_NAME, str(ADDON_DIR))

        # Test 1 — EXT_UOM_UOM const present
        if "pub const EXT_UOM_UOM: OdooEntity" not in rust:
            failures.append(
                f"Test 1 FAIL: 'pub const EXT_UOM_UOM: OdooEntity' not found\n"
                f"  (first 400 chars of output):\n{rust[:400]}"
            )
        else:
            print("Test 1 PASS: pub const EXT_UOM_UOM: OdooEntity found")

        # Test 2 — model_name field
        if 'model_name: "uom.uom"' not in rust:
            failures.append(
                f"Test 2 FAIL: 'model_name: \"uom.uom\"' not found in output"
            )
        else:
            print('Test 2 PASS: model_name: "uom.uom" found')

        # Test 3 — kind field
        if "kind: OdooEntityKind::Model" not in rust:
            failures.append(
                "Test 3 FAIL: 'kind: OdooEntityKind::Model' not found in output"
            )
        else:
            print("Test 3 PASS: kind: OdooEntityKind::Model found")

        # Test 4 — confidence
        if "confidence: OdooConfidence::Extracted" not in rust:
            failures.append(
                "Test 4 FAIL: 'confidence: OdooConfidence::Extracted' not found"
            )
        else:
            print("Test 4 PASS: confidence: OdooConfidence::Extracted found")

        # Test 5 — balanced braces
        opens, closes = _count_braces(rust)
        if opens != closes:
            failures.append(
                f"Test 5 FAIL: unbalanced braces — {opens} open vs {closes} close"
            )
        else:
            print(f"Test 5 PASS: balanced braces ({opens} open, {closes} close)")

        # Test 6 — fallback rate
        total_fields = sum(len(e.fields) for e in entities)
        total_methods = sum(len(e.methods) for e in entities)
        rate = log.fallback_rate(total_fields)
        if rate > 0.05:
            failures.append(
                f"Test 6 FAIL: fallback rate {rate:.1%} > 5% "
                f"(other_fields={log.summary()['other_fields']}, "
                f"helper_methods={log.summary()['helper_methods']}, "
                f"total_fields={total_fields}, total_methods={total_methods})"
            )
        else:
            print(
                f"Test 6 PASS: fallback rate {rate:.1%} <= 5% "
                f"(other_fields={log.summary()['other_fields']}, "
                f"helper_methods={log.summary()['helper_methods']})"
            )

        # Print summary info
        print(f"\n--- Summary ---")
        print(f"Entities: {len(entities)}")
        for e in entities:
            print(
                f"  {e.model_name}: {len(e.fields)} fields, "
                f"{len(e.methods)} methods, "
                f"{len(e.constraints)} constraints, "
                f"state_machine={'yes' if e.state_machine else 'no'}, "
                f"regulation_iris={e.regulation_iris}"
            )
        print(f"Audit: {log.summary()}")

    if failures:
        print("\n=== FAILURES ===")
        for f in failures:
            print(f"  {f}")
        return False

    print("\nALL TESTS PASS")
    return True


# pytest compatibility — expose individual test functions
def test_uom_entities_found():
    if not ADDON_DIR.is_dir():
        return  # Skip silently
    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)
    assert entities, "No entities parsed from uom addon"


def test_uom_const_name():
    if not ADDON_DIR.is_dir():
        return
    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)
    rust = emit_module(entities, ADDON_NAME, str(ADDON_DIR))
    assert "pub const EXT_UOM_UOM: OdooEntity" in rust


def test_uom_model_name():
    if not ADDON_DIR.is_dir():
        return
    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)
    rust = emit_module(entities, ADDON_NAME, str(ADDON_DIR))
    assert 'model_name: "uom.uom"' in rust


def test_uom_entity_kind():
    if not ADDON_DIR.is_dir():
        return
    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)
    rust = emit_module(entities, ADDON_NAME, str(ADDON_DIR))
    assert "kind: OdooEntityKind::Model" in rust


def test_uom_confidence():
    if not ADDON_DIR.is_dir():
        return
    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)
    rust = emit_module(entities, ADDON_NAME, str(ADDON_DIR))
    assert "confidence: OdooConfidence::Extracted" in rust


def test_uom_balanced_braces():
    if not ADDON_DIR.is_dir():
        return
    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)
    rust = emit_module(entities, ADDON_NAME, str(ADDON_DIR))
    assert rust.count("{") == rust.count("}")


def test_uom_fallback_rate():
    if not ADDON_DIR.is_dir():
        return
    log = FallbackLog()
    entities = parse_addon(ADDON_DIR, ADDON_NAME, log)
    total_fields = sum(len(e.fields) for e in entities)
    total_methods = sum(len(e.methods) for e in entities)
    rate = log.fallback_rate(total_fields)
    assert rate <= 0.05, f"Fallback rate {rate:.1%} > 5%"


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
