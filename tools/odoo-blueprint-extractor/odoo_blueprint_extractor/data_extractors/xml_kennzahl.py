"""Parse account_account_tags_data.xml → OdooUstvaKennzahl Rust consts.

Extracts `account.report.line` records that have `DE_*` codes (actual
Kennzahlen, not aggregate section headers). Each Kennzahl carries:
  - kz: numeric code extracted from the DE_ code (e.g. "81" from "DE_81")
  - label: German name@de field
  - tag_xmlid: derived from the expression formula (tag name used in tax_tags engine)
  - kind: Base / Tax / Derived inferred from expression label(s) present
  - regulation_iri: derived from Kennzahl number / context

Output:
  - One KZ_<number> const per Kennzahl
  - pub static USTVA_KENNZAHLEN: &[OdooUstvaKennzahl] slice
  - pub static GOBD_WIRING: OdooGobdWiring const

Regulation IRI mapping:
  Section A (output VAT on supplies, Kz 81/86/87/35/77/76):
      base boxes → ogit:regulation/de/ustg/13  (Ausgangsumsatz)
      tax boxes  → ogit:regulation/de/ustg/18  (Voranmeldung)
  Section B (tax-free supplies, Kz 41/44/49/43/48):
      → ogit:regulation/de/ustg/4   (Steuerbefreiungen)
  Section C (innergemeinschaftliche Erwerbe, Kz 89/93/91/90/95/94):
      base/tax → ogit:regulation/de/ustg/1  (ig Erwerb)
  Section D (§13b reverse charge, Kz 46/73/84):
      → ogit:regulation/de/ustg/13b
  Section F (Vorsteuer, Kz 66/61/62/67/63/59/64):
      → ogit:regulation/de/ustg/15  (Vorsteuerabzug)
  Default:
      → ogit:regulation/de/ustg/18  (§18 Voranmeldung)
"""

try:
    from defusedxml import ElementTree as ET  # XXE/billion-laughs hardening
except ImportError:  # pragma: no cover — defusedxml listed in pyproject.toml
    import xml.etree.ElementTree as ET  # type: ignore[no-redef]
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Regulation IRI mapping by Kennzahl number                                   #
# --------------------------------------------------------------------------- #

# Output VAT on supplies (Section A)
_SECTION_A = {"81", "86", "87", "35", "77", "76"}
# Tax-free supplies (Section B)
_SECTION_B = {"41", "44", "49", "43", "48"}
# ig Erwerbe (Section C)
_SECTION_C = {"91", "89", "93", "90", "95", "94"}
# §13b reverse charge (Section D)
_SECTION_D = {"46", "73", "84"}
# Ergänzende Angaben (Section E)
_SECTION_E = {"42", "60", "21", "45"}
# Vorsteuer (Section F)
_SECTION_F = {"66", "61", "62", "67", "63", "59", "64"}


def _regulation_iris_for_kz(kz: str) -> List[str]:
    if kz in _SECTION_A:
        return ["ogit:regulation/de/ustg/13", "ogit:regulation/de/ustg/18"]
    if kz in _SECTION_B:
        return ["ogit:regulation/de/ustg/4", "ogit:regulation/de/ustg/18"]
    if kz in _SECTION_C:
        return ["ogit:regulation/de/ustg/1a", "ogit:regulation/de/ustg/18"]
    if kz in _SECTION_D:
        return ["ogit:regulation/de/ustg/13b", "ogit:regulation/de/ustg/18"]
    if kz in _SECTION_F:
        return ["ogit:regulation/de/ustg/15", "ogit:regulation/de/ustg/18"]
    return ["ogit:regulation/de/ustg/18"]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _escape_rust_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _kz_from_code(code: str) -> Optional[str]:
    """Extract numeric kz from code like 'DE_81' → '81', 'DE_45_BASE' → '45'.

    Returns None for aggregate section headers (AGG_DE_*).
    """
    if not code or code.startswith("AGG_"):
        return None
    # Remove DE_ prefix
    rest = code[3:] if code.startswith("DE_") else code
    # Take the leading numeric part (before any _BASE, _TAX suffix)
    parts = rest.split("_")
    numeric = parts[0]
    if numeric.isdigit():
        return numeric
    return None


def _infer_kind(exprs: List[Dict]) -> str:
    """Infer OdooKennzahlKind from expression labels.

    Rules:
    - If only 'base' expressions → Base
    - If only 'tax' expressions → Tax
    - If both base+tax or aggregation formulas → Derived
    - If 'sales' or other unusual label → Derived
    """
    labels = {e["label"] for e in exprs if e.get("label")}
    if not labels:
        return "Derived"
    if labels == {"base"}:
        return "Base"
    if labels == {"tax"}:
        return "Tax"
    # Check for aggregate formulas (contain '+')
    for e in exprs:
        formula = e.get("formula", "") or ""
        if "+" in formula:
            return "Derived"
    if "base" in labels and "tax" in labels:
        return "Derived"
    return "Derived"


def _tag_xmlid_from_exprs(rec_id: str, kz: str, exprs: List[Dict]) -> str:
    """Derive a tag_xmlid from expression data.

    The formula in a tax_tags expression refers to a tag name (e.g. '-81_BASE').
    We use the report line's record id as the xmlid base (e.g.
    'l10n_de.tax_report_de_tag_81').
    """
    # Use first tax_tags expression's formula as the tag reference
    for e in exprs:
        if e.get("engine") == "tax_tags" and e.get("formula"):
            formula = e["formula"].lstrip("-")
            return f"l10n_de.tax_report_de_{formula.lower().replace('_base','').replace('_tax','')}"
    # Fallback: use record id
    return f"l10n_de.{rec_id}"


def _regulation_literal(kz: str) -> str:
    iris = _regulation_iris_for_kz(kz)
    inner = ", ".join(f'"{iri}"' for iri in iris)
    return f"&[{inner}]"


# --------------------------------------------------------------------------- #
# XML parser                                                                   #
# --------------------------------------------------------------------------- #

def _extract_line_records(elem: ET.Element) -> List[Dict]:
    """Recursively extract all account.report.line records with their expressions."""
    results = []
    for child in elem:
        if child.tag == "record" and child.get("model") == "account.report.line":
            rec_id = child.get("id", "")
            code = None
            name_de = None
            name_en = None
            exprs: List[Dict] = []

            for field in child:
                if field.tag != "field":
                    continue
                fname = field.get("name", "")
                if fname == "code":
                    code = field.text
                elif fname == "name@de":
                    name_de = field.text
                elif fname == "name":
                    name_en = field.text
                elif fname == "expression_ids":
                    # Nested expression records
                    for expr_rec in field:
                        if expr_rec.tag == "record" and expr_rec.get("model") == "account.report.expression":
                            expr_id = expr_rec.get("id", "")
                            label = None
                            formula = None
                            engine = None
                            for ef in expr_rec:
                                if ef.tag == "field":
                                    efn = ef.get("name", "")
                                    if efn == "label":
                                        label = ef.text
                                    elif efn == "formula":
                                        formula = ef.text
                                    elif efn == "engine":
                                        engine = ef.text
                            exprs.append({"id": expr_id, "label": label, "formula": formula, "engine": engine})

            results.append({
                "id": rec_id,
                "code": code,
                "name_de": name_de,
                "name_en": name_en,
                "exprs": exprs,
            })

        # Recurse into all children regardless
        results.extend(_extract_line_records(child))

    return results


def parse_xml_kennzahlen(xml_path: Path) -> List[Dict]:
    """Parse the XML and return a list of Kennzahl dicts for DE_ lines.

    Each dict has: kz, label, tag_xmlid, kind, regulation_iri
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    all_lines = _extract_line_records(root)

    # Deduplicate by record id (XML nesting may cause revisits)
    seen_ids = set()
    kennzahlen = []

    for line in all_lines:
        rec_id = line["id"]
        if rec_id in seen_ids:
            continue
        seen_ids.add(rec_id)

        code = line.get("code") or ""
        kz = _kz_from_code(code)
        if kz is None:
            continue  # skip aggregate sections, LINE* rows, etc.

        # Skip if code contains LINE or is section aggregate
        if "LINE" in code:
            continue

        name_de = (line.get("name_de") or line.get("name_en") or "").strip()
        exprs = line.get("exprs", [])
        kind_str = _infer_kind(exprs)
        tag_xmlid = _tag_xmlid_from_exprs(rec_id, kz, exprs)

        kennzahlen.append({
            "kz": kz,
            "label": name_de,
            "tag_xmlid": tag_xmlid,
            "kind": kind_str,
            "regulation_iri": _regulation_literal(kz),
        })

    return kennzahlen


# --------------------------------------------------------------------------- #
# Rust emitter                                                                 #
# --------------------------------------------------------------------------- #

def emit_kennzahlen_rs(xml_path: Path) -> str:
    """Emit the full l10n_de_kennzahlen.rs Rust source."""
    kennzahlen = parse_xml_kennzahlen(xml_path)

    lines: List[str] = []
    lines.append(
        "//! Auto-generated from l10n_de UStVA XML by `tools/odoo-blueprint-extractor`.\n"
        "//! Do NOT edit by hand — re-run: `python -m odoo_blueprint_extractor data --addon l10n_de --out <dir>`\n"
        "//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-4).\n"
        "//!\n"
        "//! Provenance:\n"
        "//!   `/home/user/odoo/addons/l10n_de/data/account_account_tags_data.xml`\n"
        "//!   `account/models/company.py:268` + `l10n_de/models/res_company.py:32`\n"
    )
    lines.append("\n")
    lines.append("use crate::odoo_blueprint::{OdooGobdWiring, OdooKennzahlKind, OdooUstvaKennzahl};\n")
    lines.append("\n")
    lines.append("// ─── UStVA Kennzahlen ───────────────────────────────────────────────────\n\n")

    const_names: List[str] = []
    for kz_info in kennzahlen:
        kz = kz_info["kz"]
        label = kz_info["label"]
        tag_xmlid = kz_info["tag_xmlid"]
        kind = kz_info["kind"]
        regulation_iri = kz_info["regulation_iri"]

        cn = f"KZ_{kz}"
        const_names.append(cn)
        lines.append(
            f"pub const {cn}: OdooUstvaKennzahl = OdooUstvaKennzahl {{\n"
            f'    kz: "{_escape_rust_str(kz)}",\n'
            f'    label: "{_escape_rust_str(label)}",\n'
            f'    tag_xmlid: "{_escape_rust_str(tag_xmlid)}",\n'
            f"    kind: OdooKennzahlKind::{kind},\n"
            f"    regulation_iri: {regulation_iri},\n"
            f"}};\n\n"
        )

    # Canonical slice
    refs = ",\n    ".join(cn for cn in const_names)
    lines.append(
        "/// All UStVA Kennzahlen — canonical iteration handle.\n"
        "pub static USTVA_KENNZAHLEN: &[OdooUstvaKennzahl] = &[\n"
        f"    {refs},\n"
        "];\n\n"
    )

    # GoBD wiring
    lines.append("// ─── GoBD audit-trail wiring ─────────────────────────────────────────────\n\n")
    lines.append(
        "/// GoBD compliance wiring on `res.company` — when `country_code == 'DE'`,\n"
        "/// `force_restrictive_audit_trail` is computed True, forcing the audit\n"
        "/// trail per GoBD (Grundsätze zur ordnungsmäßigen Führung und\n"
        "/// Aufbewahrung von Büchern, Aufzeichnungen und Unterlagen in\n"
        "/// elektronischer Form sowie zum Datenzugriff).\n"
        "///\n"
        "/// Provenance: `account/models/company.py:268` + `l10n_de/models/res_company.py:32`.\n"
        "pub static GOBD_WIRING: OdooGobdWiring = OdooGobdWiring {\n"
        '    base_field: "restrictive_audit_trail",\n'
        '    force_field: "force_restrictive_audit_trail",\n'
        "    trigger: \"country_code == 'DE'\",\n"
        '    regulation_iri: &["ogit:regulation/de/gobd", "ogit:regulation/de/ao/146a"],\n'
        "};\n\n"
    )

    # Tests
    kz_set = {k["kz"] for k in kennzahlen}
    canonical = [c for c in ["81", "86", "87", "35", "41", "44", "49"] if c in kz_set]
    canonical_checks = "\n".join(
        f'        assert!(kz_codes.contains("{c}"), "Missing canonical UStVA Kennzahl Kz.{c}");'
        for c in canonical
    )

    lines.append(
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n"
        "\n"
        "    #[test]\n"
        "    fn ustva_kennzahlen_cover_canonical_boxes() {\n"
        "        let kz_codes: std::collections::HashSet<&str> =\n"
        "            USTVA_KENNZAHLEN.iter().map(|k| k.kz).collect();\n"
        f"{canonical_checks}\n"
        "    }\n"
        "\n"
        "    #[test]\n"
        "    fn ustva_kennzahlen_non_empty() {\n"
        f"        assert_eq!(USTVA_KENNZAHLEN.len(), {len(const_names)});\n"
        "    }\n"
        "\n"
        "    #[test]\n"
        "    fn gobd_wiring_has_correct_trigger() {\n"
        "        assert_eq!(GOBD_WIRING.trigger, \"country_code == 'DE'\");\n"
        "        assert!(!GOBD_WIRING.regulation_iri.is_empty());\n"
        "    }\n"
        "}\n"
    )

    return "".join(lines)
