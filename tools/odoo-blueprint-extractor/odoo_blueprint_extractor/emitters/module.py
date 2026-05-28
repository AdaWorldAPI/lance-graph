"""Wrap multiple OdooEntity consts into one Rust source file.

Header:
    //! Auto-generated from <addon_path>/ by `tools/odoo-blueprint-extractor`.
    //! Do NOT edit by hand — re-run the extractor.
    //! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-2).

    use crate::odoo_blueprint::*;

Then one `pub const EXT_*` block per entity, separated by blank lines.
"""

from typing import Any, List

from .rust import emit_entity


def emit_module(
    entities: List[Any],
    addon_name: str,
    addon_path: str,
) -> str:
    """Return a complete Rust source file as a string.

    *entities*   — list of ParsedClass instances.
    *addon_name* — short addon name (e.g. 'uom').
    *addon_path* — absolute path to the addon root (for the header comment).
    """
    parts: List[str] = []

    # File-level doc comment
    parts.append(
        f"//! Auto-generated from {addon_path}/ by `tools/odoo-blueprint-extractor`.\n"
        f"//! Do NOT edit by hand — re-run the extractor.\n"
        f"//! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-2)."
    )
    parts.append("")
    parts.append("use crate::odoo_blueprint::*;")
    parts.append("")

    if not entities:
        parts.append(
            f"// WARNING: no OdooEntity found in addon '{addon_name}'."
        )
    else:
        for entity in entities:
            parts.append(emit_entity(entity, addon_path))
            parts.append("")  # blank line between consts

    return "\n".join(parts)
