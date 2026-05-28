"""Wrap multiple OdooEntity consts into one Rust source file.

Header:
    //! Auto-generated from <addon_path>/ by `tools/odoo-blueprint-extractor`.
    //! Do NOT edit by hand — re-run the extractor.
    //! Plan: `.claude/plans/odoo-source-extraction-v1.md` (D-ODOO-EXT-2).

    use crate::odoo_blueprint::*;

Then one `pub const EXT_*` block per entity, separated by blank lines.

Deduplication: when multiple Odoo classes share the same model_name (common
for _inherit extension classes spread across files), we keep the *richest*
entry — the one with the most fields + methods.  This is the correct
semantic: each Python file that adds `_inherit = 'some.model'` is adding
incremental fields/methods; the merged picture is the richest class.
Extension-fragment merging (D-ODOO-EXT-5) will do a proper additive pass;
this is a compile-safe dedup to prevent duplicate const names.
"""

from typing import Any, Dict, List, Tuple

from .rust import emit_entity


def _richness(entity: Any) -> int:
    """Score an entity by number of fields + methods (higher = richer)."""
    return len(entity.fields) + len(entity.methods)


def _dedup_by_model_name(entities: List[Any]) -> Tuple[List[Any], int]:
    """Return (deduped_list, n_dropped) keeping the richest per model_name."""
    seen: Dict[str, Any] = {}
    for ent in entities:
        key = ent.model_name or ent.class_name
        if key not in seen or _richness(ent) > _richness(seen[key]):
            seen[key] = ent
    dropped = len(entities) - len(seen)
    # Preserve stable ordering (insertion order of first-seen key)
    order: List[str] = []
    for ent in entities:
        k = ent.model_name or ent.class_name
        if k not in order:
            order.append(k)
    return [seen[k] for k in order], dropped


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
        deduped, dropped = _dedup_by_model_name(entities)
        if dropped:
            parts.append(
                f"// NOTE: {dropped} duplicate model_name(s) merged (richest class kept)."
            )
            parts.append("")
        for entity in deduped:
            parts.append(emit_entity(entity, addon_path))
            parts.append("")  # blank line between consts

    return "\n".join(parts)
