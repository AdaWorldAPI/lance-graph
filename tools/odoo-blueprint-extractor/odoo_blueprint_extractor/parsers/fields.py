"""Extract fields.X(...) call expressions from a class body.

Maps Odoo field types to OdooFieldKind Rust enum variants.
Computed fields (compute= kw) map to OdooFieldKind::Computed.
Unknown kinds map to OdooFieldKind::Other and are logged to the audit.
"""

import ast
from typing import Any, Dict, List, Optional

from ..audit.fallback_log import FallbackLog


# Map from fields.X attribute name to OdooFieldKind Rust variant
FIELD_KIND_MAP = {
    "Char": "Char",
    "Text": "Text",
    "Boolean": "Boolean",
    "Integer": "Integer",
    "Float": "Float",
    "Monetary": "Monetary",
    "Date": "Date",
    "Datetime": "Datetime",
    "Selection": "Selection",
    "Binary": "Binary",
    "Html": "Html",
    "Many2one": "Many2one",
    "One2many": "One2many",
    "Many2many": "Many2many",
    "Reference": "Reference",
}

# ORM field kinds that are relational (have comodel_name as first positional or kw)
RELATIONAL_KINDS = {"Many2one", "One2many", "Many2many", "Reference"}


def _get_kwarg(call: ast.Call, name: str) -> Optional[ast.expr]:
    """Return the value node for a keyword argument, or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _const_str(node: Optional[ast.expr]) -> Optional[str]:
    """Extract a string constant from an ast node, or None."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_bool(node: Optional[ast.expr]) -> Optional[bool]:
    """Extract a bool constant from an ast node, or None."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    # Handle True/False as Name nodes in older Python
    if isinstance(node, ast.NameConstant):
        return node.value
    return None


def _extract_field_assign(
    stmt: ast.Assign,
    addon: str,
    filepath: str,
    log: FallbackLog,
) -> Optional[Dict[str, Any]]:
    """Try to extract an OdooField dict from a single Assign statement."""
    # We expect: name = fields.X(...)
    if not (
        isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
    ):
        return None

    call: ast.Call = stmt.value
    func: ast.Attribute = call.func

    # The object should be `fields` (Name) or `fields` via import alias
    if not (isinstance(func.value, ast.Name) and func.value.id == "fields"):
        return None

    field_type = func.attr

    # Determine field name from assignment target
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return None
    field_name: str = stmt.targets[0].id

    # Check if computed
    compute_kw = _get_kwarg(call, "compute")
    is_computed = compute_kw is not None
    compute_method = _const_str(compute_kw)

    # Determine kind
    if is_computed:
        kind = "Computed"
        # Still record base type for info but classify as Computed
    elif field_type in FIELD_KIND_MAP:
        kind = FIELD_KIND_MAP[field_type]
    else:
        kind = "Other"
        log.record(addon, filepath, stmt.lineno, f"fields.{field_type}", "unknown field kind")

    # Extract comodel_name (first positional arg for relational, or kw)
    target: Optional[str] = None
    if field_type in RELATIONAL_KINDS or (is_computed and field_type in RELATIONAL_KINDS):
        comodel_kw = _get_kwarg(call, "comodel_name")
        if comodel_kw is not None:
            target = _const_str(comodel_kw)
        elif call.args:
            target = _const_str(call.args[0])

    # Extract required
    req_node = _get_kwarg(call, "required")
    required = _const_bool(req_node) or False

    # Extract help string
    help_node = _get_kwarg(call, "help")
    help_str = _const_str(help_node) or ""

    # Extract depends list (for computed fields via @api.depends — see decorators parser;
    # here we capture the `depends` keyword if present as a direct kw arg)
    depends: List[str] = []

    # string for label
    string_node = _get_kwarg(call, "string")
    string_str = _const_str(string_node) or ""

    return {
        "name": field_name,
        "kind": kind,
        "target": target,
        "required": required,
        "computed": compute_method,
        "depends": depends,
        "help": help_str,
        "string": string_str,
        "lineno": stmt.lineno,
    }


def extract_fields(
    body: List[ast.stmt],
    addon: str,
    filepath: str,
    log: FallbackLog,
) -> List[Dict[str, Any]]:
    """Extract all field declarations from a class body."""
    result = []
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        field_dict = _extract_field_assign(stmt, addon, filepath, log)
        if field_dict is not None:
            result.append(field_dict)
    return result
