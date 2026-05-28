"""Parse account.account-de_skr0{3,4}.csv → OdooAccountTemplate Rust consts.

Columns observed in both files:
    id, code, name, tag_ids, account_type, reconcile, non_trade, tax_ids, name@de

Output:
  - One EXT_SKR03_<CODE> / EXT_SKR04_<CODE> const per row.
  - pub static SKR03_CHART / SKR04_CHART slice consts.

Regulation IRI derivation (from account_type):
  income / income_other          → ogit:regulation/de/ustg/13  (Umsatzsteuer)
  expense / expense_depreciation → ogit:regulation/de/ustg/15  (Vorsteuer)
  asset_*                        → ogit:regulation/de/hgb/266  (Bilanzgliederung Aktiva)
  liability_*                    → ogit:regulation/de/hgb/266  (Bilanzgliederung Passiva)
  equity*                        → ogit:regulation/de/hgb/266
  default                        → ogit:regulation/de/hgb/238  (Buchführungspflicht)
"""

import csv
from pathlib import Path
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# Regulation IRI mapping                                                       #
# --------------------------------------------------------------------------- #

def _regulation_iris(account_type: str) -> List[str]:
    at = account_type.lower()
    if at.startswith("income"):
        return ["ogit:regulation/de/ustg/13"]
    if at.startswith("expense"):
        return ["ogit:regulation/de/ustg/15"]
    if at.startswith("asset") or at.startswith("liability") or at.startswith("equity"):
        return ["ogit:regulation/de/hgb/266"]
    return ["ogit:regulation/de/hgb/238"]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _escape_rust_str(s: str) -> str:
    """Escape a string for inclusion in a Rust string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _const_name(prefix: str, code: str) -> str:
    """EXT_SKR03_8400 from ('EXT_SKR03_', '8400')."""
    safe = code.replace("-", "_").replace(".", "_")
    return f"{prefix}{safe}"


def _tag_xmlids_literal(tag_ids_cell: str) -> str:
    """Convert 'l10n_de.tag_x,l10n_de.tag_y' to &["l10n_de.tag_x", "l10n_de.tag_y"]."""
    if not tag_ids_cell or not tag_ids_cell.strip():
        return "&[]"
    tags = [t.strip() for t in tag_ids_cell.split(",") if t.strip()]
    if not tags:
        return "&[]"
    inner = ", ".join(f'"{_escape_rust_str(t)}"' for t in tags)
    return f"&[{inner}]"


def _regulation_literal(account_type: str) -> str:
    iris = _regulation_iris(account_type)
    inner = ", ".join(f'"{iri}"' for iri in iris)
    return f"&[{inner}]"


# --------------------------------------------------------------------------- #
# Per-row emitter                                                              #
# --------------------------------------------------------------------------- #

def _emit_const(const_name: str, row: dict, chart_variant: str) -> str:
    code = _escape_rust_str(row.get("code", ""))
    # Prefer German name; fall back to English name
    name_de = row.get("name@de", "").strip()
    name_en = row.get("name", "").strip()
    name = name_de if name_de else name_en
    account_type = row.get("account_type", "").strip()
    tag_ids_cell = row.get("tag_ids", "").strip()

    return (
        f"pub const {const_name}: OdooAccountTemplate = OdooAccountTemplate {{\n"
        f'    code: "{code}",\n'
        f'    name: "{_escape_rust_str(name)}",\n'
        f'    account_type: "{_escape_rust_str(account_type)}",\n'
        f"    tag_xmlids: {_tag_xmlids_literal(tag_ids_cell)},\n"
        f"    chart: OdooSkrChart::{chart_variant},\n"
        f"    regulation_iri: {_regulation_literal(account_type)},\n"
        f"}};\n"
    )


# --------------------------------------------------------------------------- #
# CSV parser                                                                   #
# --------------------------------------------------------------------------- #

def parse_csv_chart(csv_path: Path, chart: str) -> Tuple[List[Tuple[str, dict]], str]:
    """Parse a SKR CSV file.

    Returns (rows, chart_variant) where rows is [(const_name, row_dict), ...].
    chart must be 'skr03' or 'skr04'.
    """
    if chart == "skr03":
        prefix = "EXT_SKR03_"
        chart_variant = "Skr03"
    else:
        prefix = "EXT_SKR04_"
        chart_variant = "Skr04"

    rows: List[Tuple[str, dict]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("code", "").strip()
            if not code:
                continue  # skip rows without a code
            cn = _const_name(prefix, code)
            rows.append((cn, dict(row)))
    return rows, chart_variant


# --------------------------------------------------------------------------- #
# Full emitter                                                                 #
# --------------------------------------------------------------------------- #

def emit_chart_rs(skr03_path: Path, skr04_path: Path) -> str:
    """Emit the full l10n_de_chart.rs Rust source."""
    rows03, variant03 = parse_csv_chart(skr03_path, "skr03")
    rows04, variant04 = parse_csv_chart(skr04_path, "skr04")

    lines: List[str] = []
    lines.append(
        "//! Auto-generated from l10n_de SKR03/SKR04 CSVs by `tools/odoo-blueprint-extractor`.\n"
        "//! Do NOT edit by hand — re-run: `python -m odoo_blueprint_extractor data --addon l10n_de --out <dir>`\n"
        "//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-4).\n"
        "//!\n"
        "//! Provenance:\n"
        "//!   `/home/user/odoo/addons/l10n_de/data/template/account.account-de_skr03.csv`\n"
        "//!   `/home/user/odoo/addons/l10n_de/data/template/account.account-de_skr04.csv`\n"
    )
    lines.append("\n")
    lines.append("use crate::odoo_blueprint::{OdooAccountTemplate, OdooSkrChart};\n")
    lines.append("\n")

    # Emit SKR03 consts
    lines.append("// ─── SKR03 account template consts ─────────────────────────────────────\n\n")
    skr03_names: List[str] = []
    for cn, row in rows03:
        lines.append(_emit_const(cn, row, variant03))
        lines.append("\n")
        skr03_names.append(cn)

    # Emit SKR04 consts
    lines.append("// ─── SKR04 account template consts ─────────────────────────────────────\n\n")
    skr04_names: List[str] = []
    for cn, row in rows04:
        lines.append(_emit_const(cn, row, variant04))
        lines.append("\n")
        skr04_names.append(cn)

    # Emit slice consts
    lines.append("// ─── Canonical iteration handles ────────────────────────────────────────\n\n")

    skr03_refs = ",\n    ".join(n for n in skr03_names)
    lines.append(
        "/// All SKR03 account templates — canonical iteration handle.\n"
        f"pub static SKR03_CHART: &[OdooAccountTemplate] = &[\n"
        f"    {skr03_refs},\n"
        "];\n\n"
    )

    skr04_refs = ",\n    ".join(n for n in skr04_names)
    lines.append(
        "/// All SKR04 account templates — canonical iteration handle.\n"
        f"pub static SKR04_CHART: &[OdooAccountTemplate] = &[\n"
        f"    {skr04_refs},\n"
        "];\n\n"
    )

    # Tests
    lines.append(
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n"
        "\n"
        "    #[test]\n"
        "    fn skr03_chart_has_expected_size() {\n"
        f"        assert_eq!(SKR03_CHART.len(), {len(skr03_names)});\n"
        "    }\n"
        "\n"
        "    #[test]\n"
        "    fn skr04_chart_has_expected_size() {\n"
        f"        assert_eq!(SKR04_CHART.len(), {len(skr04_names)});\n"
        "    }\n"
        "\n"
        "    #[test]\n"
        "    fn skr03_chart_entries_have_codes() {\n"
        "        for entry in SKR03_CHART {\n"
        "            assert!(!entry.code.is_empty(), \"SKR03 entry missing code\");\n"
        "        }\n"
        "    }\n"
        "\n"
        "    #[test]\n"
        "    fn skr04_chart_entries_have_codes() {\n"
        "        for entry in SKR04_CHART {\n"
        "            assert!(!entry.code.is_empty(), \"SKR04 entry missing code\");\n"
        "        }\n"
        "    }\n"
        "}\n"
    )

    return "".join(lines)
