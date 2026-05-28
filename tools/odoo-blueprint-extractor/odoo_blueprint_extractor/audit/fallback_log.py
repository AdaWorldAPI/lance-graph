"""Fallback / unrecognized construct logger.

Records:
- Unknown field kinds -> OdooFieldKind::Other
- Unclassified methods -> OdooMethodKind::Helper
- Skipped classes (non-Odoo bases)
- File-level errors (syntax errors, read errors)

flush(path) writes the buffered entries to a JSON file.
summary() returns counts for the CLI report line.
"""

import json
from typing import Any, Dict, List, Optional


class FallbackLog:
    """Collects fallback/audit events during a parse run."""

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._other_fields = 0
        self._helper_methods = 0
        self._skipped_classes = 0

    def record(
        self,
        addon: str,
        filepath: str,
        lineno: int,
        construct: str,
        reason: str,
    ) -> None:
        """Record a generic unrecognized construct."""
        self._entries.append(
            {
                "addon": addon,
                "file": filepath,
                "lineno": lineno,
                "construct": construct,
                "reason": reason,
            }
        )
        # Count ::Other fields
        if construct.startswith("fields.") and reason == "unknown field kind":
            self._other_fields += 1

    def record_helper_method(
        self,
        addon: str,
        filepath: str,
        lineno: int,
        method_name: str,
    ) -> None:
        """Record a method that fell through to OdooMethodKind::Helper."""
        self._entries.append(
            {
                "addon": addon,
                "file": filepath,
                "lineno": lineno,
                "construct": f"method:{method_name}",
                "reason": "classified as Helper",
            }
        )
        self._helper_methods += 1

    def record_skipped_class(
        self,
        addon: str,
        filepath: str,
        lineno: int,
        class_name: str,
    ) -> None:
        """Record a ClassDef that was skipped (no Odoo ORM base)."""
        self._entries.append(
            {
                "addon": addon,
                "file": filepath,
                "lineno": lineno,
                "construct": f"class:{class_name}",
                "reason": "skipped — non-Odoo base class",
            }
        )
        self._skipped_classes += 1

    def summary(self) -> Dict[str, int]:
        """Return summary counts for the CLI report line."""
        return {
            "other_fields": self._other_fields,
            "helper_methods": self._helper_methods,
            "skipped_classes": self._skipped_classes,
            "total": len(self._entries),
        }

    def fallback_rate(self, total_fields: int) -> float:
        """Return fallback rate — fraction of fields mapped to OdooFieldKind::Other.

        Only truly unrecognised field kinds (::Other) drive the rate. The denominator
        is total fields, so the rate reflects how well the field parser covers the
        addon's field declarations.
        """
        if total_fields == 0:
            return 0.0
        return self._other_fields / total_fields

    def flush(self, path: str) -> None:
        """Write the buffered entries to a JSON file at *path*."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": self.summary(),
                    "entries": self._entries,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
