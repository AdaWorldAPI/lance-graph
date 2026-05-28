"""Extract FunctionDef nodes from a class body and classify by name prefix / decorator.

Classification order (first match wins):
    _compute_*             -> OdooMethodKind::Compute
    _inverse_*             -> OdooMethodKind::Inverse
    _check_* | @api.constrains -> OdooMethodKind::Constrain
    _onchange_*            -> OdooMethodKind::Onchange
    action_*               -> OdooMethodKind::Action
    _cron_*                -> OdooMethodKind::Cron
    @api.model_create_multi -> OdooMethodKind::ApiModelCreateMulti
    @api.model (alone)     -> OdooMethodKind::ApiModel
    ORM override           -> OdooMethodKind::Override
    everything else        -> OdooMethodKind::Helper
"""

import ast
from typing import Any, Dict, List, Optional

from ..audit.fallback_log import FallbackLog

# ORM method names that count as Override
ORM_OVERRIDE_NAMES = frozenset(
    {
        "create",
        "write",
        "unlink",
        "read",
        "search",
        "name_get",
        "_compute_display_name",
        "copy",
        "default_get",
        "fields_get",
        "read_group",
    }
)


def _decorator_names(func: ast.FunctionDef) -> List[str]:
    """Return a flat list of decorator names / attribute strings like 'api.depends'."""
    names = []
    for deco in func.decorator_list:
        if isinstance(deco, ast.Attribute):
            # @api.depends(...)  or  @api.model
            obj = deco.value
            if isinstance(obj, ast.Name):
                names.append(f"{obj.id}.{deco.attr}")
        elif isinstance(deco, ast.Call):
            # @api.depends('x', 'y') — the Call wraps the Attribute
            if isinstance(deco.func, ast.Attribute):
                obj = deco.func.value
                if isinstance(obj, ast.Name):
                    names.append(f"{obj.id}.{deco.func.attr}")
        elif isinstance(deco, ast.Name):
            names.append(deco.id)
    return names


def classify_method(func: ast.FunctionDef) -> str:
    """Return the OdooMethodKind string key for a FunctionDef node."""
    name = func.name
    deco_names = _decorator_names(func)

    if name.startswith("_compute_"):
        return "Compute"
    if name.startswith("_inverse_"):
        return "Inverse"
    if name.startswith("_check_") or "api.constrains" in deco_names:
        return "Constrain"
    if name.startswith("_onchange_"):
        return "Onchange"
    if name.startswith("action_"):
        return "Action"
    if name.startswith("_cron_"):
        return "Cron"
    if "api.model_create_multi" in deco_names:
        return "ApiModelCreateMulti"
    if "api.model" in deco_names:
        return "ApiModel"
    # @api.ondelete hooks are ORM lifecycle overrides
    if "api.ondelete" in deco_names:
        return "Override"
    if name in ORM_OVERRIDE_NAMES:
        return "Override"
    return "Helper"


def _infer_return_kind(func: ast.FunctionDef) -> str:
    """Heuristic: infer OdooReturnKind from method name / decorators."""
    name = func.name
    deco_names = _decorator_names(func)

    if name.startswith("action_"):
        return "Action"
    if name.startswith("_compute_"):
        return "Unit"
    if name.startswith("_check_") or "api.constrains" in deco_names:
        return "Unit"
    if name.startswith("_onchange_"):
        return "Dict"
    if name in {"create"}:
        return "Record"
    if name in {"search", "read_group"}:
        return "Recordset"
    if name in {"write", "unlink"}:
        return "Boolean"
    if name in {"read"}:
        return "Dict"
    if name in {"name_get", "_compute_display_name"}:
        return "Unit"
    # For api.returns decorator
    for deco in func.decorator_list:
        if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute):
            if deco.func.attr == "returns":
                if deco.args:
                    first = deco.args[0]
                    if isinstance(first, ast.Constant):
                        s = str(first.value).lower()
                        if "record" in s:
                            return "Record"
    return "Unit"


def extract_methods(
    body: List[ast.stmt],
    addon: str,
    filepath: str,
    log: FallbackLog,
) -> List[Dict[str, Any]]:
    """Extract all methods from a class body."""
    result = []
    for stmt in body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        kind = classify_method(stmt)
        if kind == "Helper":
            log.record_helper_method(addon, filepath, stmt.lineno, stmt.name)
        return_kind = _infer_return_kind(stmt)
        result.append(
            {
                "name": stmt.name,
                "kind": kind,
                "return_kind": return_kind,
                "lineno": stmt.lineno,
            }
        )
    return result
