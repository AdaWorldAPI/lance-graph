"""ClassDef visitor — extract Odoo ORM class metadata.

Maps:
    models.Model        -> OdooEntityKind::Model
    models.TransientModel -> OdooEntityKind::Transient
    models.AbstractModel  -> OdooEntityKind::Abstract

Anything else is skipped silently (non-Odoo class).
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..audit.fallback_log import FallbackLog
from .fields import extract_fields
from .methods import extract_methods
from .decorators import extract_decorators_from_methods
from .state_machine import extract_state_machine
from .constraints import extract_constraints
from .regulation import extract_regulation_iris


# Sentinel values for OdooEntityKind (used as string keys in emitter)
ODOO_ENTITY_KIND_MODEL = "Model"
ODOO_ENTITY_KIND_TRANSIENT = "Transient"
ODOO_ENTITY_KIND_ABSTRACT = "Abstract"

# Map from base class name to OdooEntityKind string
ODOO_BASE_MAP = {
    "Model": ODOO_ENTITY_KIND_MODEL,
    "TransientModel": ODOO_ENTITY_KIND_TRANSIENT,
    "AbstractModel": ODOO_ENTITY_KIND_ABSTRACT,
}


@dataclass
class ParsedClass:
    """All information extracted from one Odoo ORM ClassDef."""

    class_name: str
    kind: str  # OdooEntityKind key: "Model" | "Transient" | "Abstract"
    model_name: str
    inherit: List[str]
    description: str
    fields: list
    methods: list
    decorators: list
    state_machine: Optional[object]
    constraints: list
    regulation_iris: List[str]
    source_file: str
    line_start: int
    line_end: int


def _get_base_kind(node: ast.ClassDef) -> Optional[str]:
    """Return OdooEntityKind key if this class inherits from an Odoo ORM base, else None."""
    for base in node.bases:
        attr_name = None
        if isinstance(base, ast.Attribute):
            attr_name = base.attr
        elif isinstance(base, ast.Name):
            attr_name = base.id
        if attr_name in ODOO_BASE_MAP:
            return ODOO_BASE_MAP[attr_name]
    return None


def _extract_string_assignment(body: List[ast.stmt], name: str) -> Optional[str]:
    """Extract the string value of `name = 'value'` from a class body."""
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name:
                val = stmt.value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    return val.value
                # Handle concatenated strings (rare but possible)
                if isinstance(val, ast.JoinedStr):
                    return None  # f-string, skip
    return None


def _extract_list_or_string(body: List[ast.stmt], name: str) -> List[str]:
    """Extract `name = 'x'` or `name = ['x', 'y']` as a list of strings."""
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id == name:
                val = stmt.value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    return [val.value]
                if isinstance(val, ast.List):
                    result = []
                    for elt in val.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            result.append(elt.value)
                    return result
    return []


def parse_file(
    path: Path,
    addon_name: str,
    log: FallbackLog,
) -> List[ParsedClass]:
    """Parse one Python file and return all Odoo ORM classes found."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.record(addon_name, str(path), 0, "file_read", str(e))
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        log.record(addon_name, str(path), 0, "syntax_error", str(e))
        return []

    results = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        kind = _get_base_kind(node)
        if kind is None:
            log.record_skipped_class(addon_name, str(path), node.lineno, node.name)
            continue

        model_name = _extract_string_assignment(node.body, "_name") or ""
        inherit = _extract_list_or_string(node.body, "_inherit")
        description = _extract_string_assignment(node.body, "_description") or ""

        # If no _name but has _inherit (string form), use first inherit as model_name
        if not model_name and len(inherit) == 1:
            model_name = inherit[0]
        elif not model_name and inherit:
            model_name = inherit[0]

        parsed_fields = extract_fields(node.body, addon_name, str(path), log)
        parsed_methods = extract_methods(node.body, addon_name, str(path), log)
        parsed_decorators = extract_decorators_from_methods(node.body)
        state_machine = extract_state_machine(node.body, parsed_fields, parsed_methods)
        parsed_constraints = extract_constraints(
            node.body, parsed_methods, addon_name, str(path), log
        )

        # Collect all text for regulation scanning
        regulation_text_parts = [description]
        for f in parsed_fields:
            if f.get("help"):
                regulation_text_parts.append(f["help"])
        # Class docstring
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            regulation_text_parts.append(str(node.body[0].value.value))

        regulation_iris = extract_regulation_iris(" ".join(regulation_text_parts))

        line_end = getattr(node, "end_lineno", node.lineno)

        results.append(
            ParsedClass(
                class_name=node.name,
                kind=kind,
                model_name=model_name,
                inherit=inherit,
                description=description,
                fields=parsed_fields,
                methods=parsed_methods,
                decorators=parsed_decorators,
                state_machine=state_machine,
                constraints=parsed_constraints,
                regulation_iris=regulation_iris,
                source_file=str(path),
                line_start=node.lineno,
                line_end=line_end,
            )
        )

    return results


def parse_addon(
    addon_dir: Path,
    addon_name: str,
    log: FallbackLog,
) -> List[ParsedClass]:
    """Parse all Python files in an addon's models/ directory."""
    models_dir = addon_dir / "models"
    if not models_dir.is_dir():
        return []

    results = []
    for py_file in sorted(models_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        classes = parse_file(py_file, addon_name, log)
        results.extend(classes)

    return results
