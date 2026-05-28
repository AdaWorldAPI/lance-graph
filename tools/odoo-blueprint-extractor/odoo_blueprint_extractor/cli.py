"""CLI implementation for odoo-blueprint-extractor.

Usage:
    python -m odoo_blueprint_extractor \\
        --addons /home/user/odoo/addons \\
        --addon uom \\
        --out -           # stdout
        [--audit /tmp/fallback.json]
"""

import argparse
import os
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
    p.add_argument(
        "--addons",
        required=True,
        metavar="DIR",
        help="Path to the Odoo addons root (e.g. /home/user/odoo/addons).",
    )
    p.add_argument(
        "--addon",
        required=True,
        metavar="NAME",
        help="Addon name to extract (e.g. uom).",
    )
    p.add_argument(
        "--out",
        required=True,
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
