"""Extract GoBD audit-trail wiring from l10n_de/models/res_company.py.

Verifies:
  - `_compute_force_restrictive_audit_trail` is present and sets
    `force_restrictive_audit_trail |= company.country_code == 'DE'`
  - The base fields `restrictive_audit_trail` + `force_restrictive_audit_trail`
    exist in account/models/company.py

This module does NOT generate new Rust — the GoBD wiring const is emitted
by xml_kennzahl.py alongside the UStVA Kennzahlen. This module provides
a verification/analysis helper that can be called for auditing purposes.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional


def _find_field_defs(source: str) -> List[Dict]:
    """Find fields.Boolean / fields.Char definitions in source."""
    tree = ast.parse(source)
    fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    value = node.value
                    if (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute)
                        and isinstance(value.func.value, ast.Name)
                        and value.func.value.id == "fields"
                    ):
                        fields.append({
                            "name": target.id,
                            "field_type": value.func.attr,
                            "lineno": node.lineno,
                        })
    return fields


def _find_compute_method(source: str, method_name: str) -> Optional[Dict]:
    """Find a method definition and return its line range + source text."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            end = getattr(node, "end_lineno", node.lineno)
            # Extract source lines for body inspection
            lines = source.splitlines()
            body_lines = lines[node.lineno - 1:end]
            body_source = "\n".join(body_lines)
            return {
                "name": method_name,
                "lineno": node.lineno,
                "end_lineno": end,
                "source": body_source,
            }
    return None


def verify_gobd_wiring(l10n_de_dir: Path, account_dir: Path) -> Dict:
    """Verify GoBD wiring exists in both account and l10n_de.

    Returns a summary dict with:
      - base_field_found: bool
      - force_field_found: bool
      - l10n_de_compute_found: bool
      - l10n_de_compute_lineno: int
      - trigger_pattern: str
    """
    result = {
        "base_field_found": False,
        "force_field_found": False,
        "l10n_de_compute_found": False,
        "l10n_de_compute_lineno": None,
        "trigger_pattern": "country_code == 'DE'",
    }

    # Check account/models/company.py for base fields
    account_company = account_dir / "models" / "company.py"
    if account_company.exists():
        source = account_company.read_text(encoding="utf-8")
        fields = _find_field_defs(source)
        field_names = {f["name"] for f in fields}
        result["base_field_found"] = "restrictive_audit_trail" in field_names
        result["force_field_found"] = "force_restrictive_audit_trail" in field_names

    # Check l10n_de/models/res_company.py for the DE-specific compute
    l10n_de_company = l10n_de_dir / "models" / "res_company.py"
    if l10n_de_company.exists():
        source = l10n_de_company.read_text(encoding="utf-8")
        method = _find_compute_method(source, "_compute_force_restrictive_audit_trail")
        if method:
            body = method.get("source", "")
            has_trigger = "country_code" in body and "DE" in body
            result["l10n_de_compute_found"] = has_trigger
            result["l10n_de_compute_lineno"] = method.get("lineno") if has_trigger else None
            result["l10n_de_trigger_found"] = has_trigger

    return result
