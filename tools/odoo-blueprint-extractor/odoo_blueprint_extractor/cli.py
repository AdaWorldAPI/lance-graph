"""CLI implementation for odoo-blueprint-extractor.

Usage (ORM extraction — original subcommand):
    python -m odoo_blueprint_extractor \\
        --addons /home/user/odoo/addons \\
        --addon uom \\
        --out -           # stdout
        [--audit /tmp/fallback.json]

Usage (data extraction — D-ODOO-EXT-4):
    python -m odoo_blueprint_extractor data \\
        --addon l10n_de \\
        --out crates/lance-graph-ontology/src/odoo_blueprint/extracted/
        [--addons /home/user/odoo/addons]

Usage (curated-vs-extracted pairing — D-ODOO-EXT-5):
    python -m odoo_blueprint_extractor pair \\
        --crate crates/lance-graph-ontology \\
        --out crates/lance-graph-ontology/src/odoo_blueprint/extracted/pairing.rs \\
        --audit /tmp/pairings.json
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .audit.fallback_log import FallbackLog
from .emitters.module import emit_module
from .parsers.classes import parse_addon


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m odoo_blueprint_extractor",
        description="Extract Odoo ORM classes as OdooEntity Rust consts (D-ODOO-EXT-1).",
    )
    # Subparsers — 'data' is EXT-4; 'pair' is EXT-5; original args are the default (no subcommand)
    subparsers = p.add_subparsers(dest="subcommand")

    # ---- 'pair' subcommand (EXT-5) ----
    pair_p = subparsers.add_parser(
        "pair",
        help="Build curated-vs-extracted pairing table (D-ODOO-EXT-5).",
    )
    pair_p.add_argument(
        "--crate",
        required=True,
        metavar="DIR",
        help="Path to the lance-graph-ontology crate root (e.g. crates/lance-graph-ontology).",
    )
    pair_p.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output path for the generated pairing.rs file.",
    )
    pair_p.add_argument(
        "--audit",
        metavar="JSON",
        default=None,
        help="Write mismatch audit (field/method count deltas) to this JSON file.",
    )

    # ---- 'data' subcommand (EXT-4) ----
    data_p = subparsers.add_parser(
        "data",
        help="Extract CSV/XML data files → typed Rust consts (D-ODOO-EXT-4).",
    )
    data_p.add_argument(
        "--addon",
        required=True,
        metavar="NAME",
        help="Addon name whose data files to extract (currently only 'l10n_de').",
    )
    data_p.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="Output directory for emitted .rs files.",
    )
    data_p.add_argument(
        "--addons",
        metavar="DIR",
        default="/home/user/odoo/addons",
        help="Path to the Odoo addons root (default: /home/user/odoo/addons).",
    )

    # ---- original ORM extraction args (no subcommand) ----
    p.add_argument(
        "--addons",
        metavar="DIR",
        help="Path to the Odoo addons root (e.g. /home/user/odoo/addons).",
    )
    p.add_argument(
        "--addon",
        metavar="NAME",
        help="Addon name to extract (e.g. uom).",
    )
    p.add_argument(
        "--out",
        metavar="PATH|-",
        help="Output path: '-' for stdout, or a directory for file output (EXT-2).",
    )
    p.add_argument(
        "--audit",
        metavar="JSON",
        default=None,
        help="Write fallback/audit log to this JSON file.",
    )
    return p


def _run_pair_subcommand(args: argparse.Namespace) -> None:
    """Run the 'pair' subcommand — curated-vs-extracted pairing table (EXT-5)."""
    from .pairing import build_pairings, emit_audit_json, emit_pairing_rs, scan_blueprint_dir

    crate_dir = Path(args.crate)
    blueprint_dir = crate_dir / "src" / "odoo_blueprint"
    if not blueprint_dir.is_dir():
        sys.exit(f"ERROR: odoo_blueprint dir not found: {blueprint_dir}")

    out_path = Path(args.out)
    audit_path: Optional[str] = args.audit

    # Scan
    scan = scan_blueprint_dir(blueprint_dir)
    pairings = build_pairings(scan)

    n = len(pairings)
    curated_count = len(scan["curated"])
    extracted_count = len(scan["extracted"])
    print(
        f"# Pairing: curated={curated_count} model_names, "
        f"extracted={extracted_count} model_names, "
        f"overlap={n} pairings",
        file=sys.stderr,
    )

    # Emit Rust
    rust_src = emit_pairing_rs(pairings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rust_src, encoding="utf-8")
    print(f"Written: {out_path}", file=sys.stderr)

    # Emit audit JSON
    if audit_path:
        audit_json = emit_audit_json(pairings)
        Path(audit_path).write_text(audit_json, encoding="utf-8")
        print(f"Audit written: {audit_path}", file=sys.stderr)


def _run_data_subcommand(args: argparse.Namespace) -> None:
    """Run the 'data' subcommand — CSV/XML extraction for l10n_de (EXT-4)."""
    from .data_extractors.csv_chart import emit_chart_rs
    from .data_extractors.xml_kennzahl import emit_kennzahlen_rs

    addon_name: str = args.addon
    addons_root = Path(args.addons)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    addon_dir = addons_root / addon_name
    if not addon_dir.is_dir():
        sys.exit(f"ERROR: addon directory not found: {addon_dir}")

    if addon_name == "l10n_de":
        # SKR03 + SKR04 chart
        skr03_csv = addon_dir / "data" / "template" / "account.account-de_skr03.csv"
        skr04_csv = addon_dir / "data" / "template" / "account.account-de_skr04.csv"
        if not skr03_csv.exists():
            sys.exit(f"ERROR: SKR03 CSV not found: {skr03_csv}")
        if not skr04_csv.exists():
            sys.exit(f"ERROR: SKR04 CSV not found: {skr04_csv}")

        chart_rs = emit_chart_rs(skr03_csv, skr04_csv)
        chart_file = out_dir / "l10n_de_chart.rs"
        chart_file.write_text(chart_rs, encoding="utf-8")
        print(f"Written: {chart_file}", file=sys.stderr)

        # UStVA Kennzahlen + GoBD wiring
        xml_path = addon_dir / "data" / "account_account_tags_data.xml"
        if not xml_path.exists():
            sys.exit(f"ERROR: XML not found: {xml_path}")

        kennzahlen_rs = emit_kennzahlen_rs(xml_path)
        kennzahlen_file = out_dir / "l10n_de_kennzahlen.rs"
        kennzahlen_file.write_text(kennzahlen_rs, encoding="utf-8")
        print(f"Written: {kennzahlen_file}", file=sys.stderr)
    else:
        sys.exit(f"ERROR: 'data' subcommand only supports l10n_de (got: {addon_name!r})")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Route to 'pair' subcommand if requested
    if args.subcommand == "pair":
        _run_pair_subcommand(args)
        return

    # Route to 'data' subcommand if requested
    if args.subcommand == "data":
        _run_data_subcommand(args)
        return

    # Original ORM extraction path
    if not args.addons or not args.addon or not args.out:
        parser.print_help()
        sys.exit(1)

    addons_root = Path(args.addons)
    addon_name: str = args.addon
    out: str = args.out
    audit_path: Optional[str] = args.audit

    addon_dir = addons_root / addon_name
    if not addon_dir.is_dir():
        sys.exit(f"ERROR: addon directory not found: {addon_dir}")

    log = FallbackLog()

    entities = parse_addon(addon_dir, addon_name, log)

    if not entities:
        print(
            f"# WARNING: no OdooEntity found in {addon_dir}",
            file=sys.stderr,
        )

    rust_src = emit_module(entities, addon_name, str(addon_dir))

    if out == "-":
        print(rust_src)
    else:
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{addon_name}.rs"
        out_file.write_text(rust_src, encoding="utf-8")
        print(f"Written: {out_file}", file=sys.stderr)

    # Print audit summary
    summary = log.summary()
    if summary["total"] > 0 or audit_path:
        print(
            f"# Audit: {summary['other_fields']} fields ::Other, "
            f"{summary['helper_methods']} methods ::Helper, "
            f"{summary['skipped_classes']} skipped classes",
            file=sys.stderr,
        )

    if audit_path:
        log.flush(audit_path)
        print(f"# Fallback log written to: {audit_path}", file=sys.stderr)
