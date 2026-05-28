"""Extract @api.* decorators from class body methods and map to OdooDecoratorKind.

Each decorator is captured with its string argument targets (for depends/constrains/onchange).
"""

import ast
from typing import Any, Dict, List, Optional


# Map from api.X attribute to OdooDecoratorKind Rust variant
DECORATOR_KIND_MAP = {
    "depends": "ApiDepends",
    "constrains": "ApiConstrains",
    "onchange": "ApiOnchange",
    "model": "ApiModel",
    "model_create_multi": "ApiModelCreateMulti",
    "returns": "ApiReturns",
    "autovacuum": "ApiAutovacuum",
    "ondelete": None,  # Not in OdooDecoratorKind — skip (but still used by methods.py for classification)
    "depends_context": None,  # Not in OdooDecoratorKind — skip
}


def _extract_string_args(call: ast.Call) -> List[str]:
    """Return all positional string constant arguments from a Call node."""
    result = []
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            result.append(arg.value)
    return result


def _parse_decorator(deco: ast.expr) -> Optional[Dict[str, Any]]:
    """Try to parse one decorator into a dict with 'kind' + 'targets'."""
    # @api.X (no-arg attribute)
    if isinstance(deco, ast.Attribute):
        obj = deco.value
        if isinstance(obj, ast.Name) and obj.id == "api":
            kind = DECORATOR_KIND_MAP.get(deco.attr)
            if kind is None:
                return None  # Unknown or explicitly skipped
            return {"kind": kind, "targets": []}

    # @api.X(...) call form
    if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute):
        func_attr = deco.func
        obj = func_attr.value
        if isinstance(obj, ast.Name) and obj.id == "api":
            kind = DECORATOR_KIND_MAP.get(func_attr.attr)
            if kind is None:
                return None
            targets = _extract_string_args(deco)
            return {"kind": kind, "targets": targets}

    return None


def extract_decorators_from_methods(body: List[ast.stmt]) -> List[Dict[str, Any]]:
    """Collect all @api.* decorators across all methods in a class body.

    Returns a flat list of decorator dicts (one per decorator per method).
    The method name is included for cross-reference.
    """
    result = []
    for stmt in body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for deco in stmt.decorator_list:
            parsed = _parse_decorator(deco)
            if parsed is not None:
                parsed["method_name"] = stmt.name
                result.append(parsed)
    return result
