"""Extract constraint declarations from Odoo class bodies.

Two sources:
1. `_sql_constraints` list assignment -> OdooConstraintKind::Sql
   Format: [(name, sql, message), ...]
2. `@api.constrains` decorated methods -> OdooConstraintKind::Python

Also handles the newer `models.Constraint(sql, message)` class-body form
introduced in Odoo 17.
"""

import ast
from typing import Any, Dict, List, Optional

from ..audit.fallback_log import FallbackLog


def _extract_sql_constraints(body: List[ast.stmt]) -> List[Dict[str, Any]]:
    """Extract _sql_constraints = [...] as a list of constraint dicts."""
    result = []
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if not isinstance(target, ast.Name) or target.id != "_sql_constraints":
                continue
            val = stmt.value
            if not isinstance(val, ast.List):
                continue
            for elt in val.elts:
                if not isinstance(elt, ast.Tuple) or len(elt.elts) < 3:
                    continue
                name_node, sql_node, msg_node = elt.elts[0], elt.elts[1], elt.elts[2]
                name = (
                    name_node.value
                    if isinstance(name_node, ast.Constant)
                    else "<unknown>"
                )
                msg = (
                    msg_node.value
                    if isinstance(msg_node, ast.Constant)
                    else "<unknown>"
                )
                result.append(
                    {
                        "kind": "Sql",
                        "condition": msg,
                        "source_method": None,
                        "name": name,
                    }
                )
    return result


def _extract_constraint_class_attrs(body: List[ast.stmt]) -> List[Dict[str, Any]]:
    """Handle the Odoo 17 `models.Constraint('CHECK ...', 'message')` class-body form.

    These appear as:
        _factor_gt_zero = models.Constraint('CHECK (relative_factor!=0)', 'message')
    """
    result = []
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        val = stmt.value
        if not isinstance(val, ast.Call):
            continue
        func = val.func
        if not isinstance(func, ast.Attribute) or func.attr != "Constraint":
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != "models":
            continue
        # Extract message (second positional arg)
        msg = ""
        if len(val.args) >= 2 and isinstance(val.args[1], ast.Constant):
            msg = str(val.args[1].value)
        elif len(val.args) >= 1 and isinstance(val.args[0], ast.Constant):
            msg = str(val.args[0].value)
        result.append(
            {
                "kind": "Sql",
                "condition": msg,
                "source_method": None,
                "name": "<attr_constraint>",
            }
        )
    return result


def _extract_python_constraints(
    body: List[ast.stmt],
    parsed_methods: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find @api.constrains decorated methods -> OdooConstraintKind::Python."""
    result = []
    for stmt in body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        for deco in stmt.decorator_list:
            # @api.constrains('x', 'y')
            if not isinstance(deco, ast.Call):
                continue
            func = deco.func
            if not isinstance(func, ast.Attribute) or func.attr != "constrains":
                continue
            obj = func.value
            if not isinstance(obj, ast.Name) or obj.id != "api":
                continue
            # Collect the field names from the decorator args
            field_names = []
            for arg in deco.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    field_names.append(arg.value)
            condition = f"Python constraint on {', '.join(field_names)}" if field_names else "Python constraint"
            result.append(
                {
                    "kind": "Python",
                    "condition": condition,
                    "source_method": stmt.name,
                }
            )
            break  # Only one @api.constrains per method
    return result


def extract_constraints(
    body: List[ast.stmt],
    parsed_methods: List[Dict[str, Any]],
    addon: str,
    filepath: str,
    log: FallbackLog,
) -> List[Dict[str, Any]]:
    """Extract all constraints from a class body."""
    result = []
    result.extend(_extract_sql_constraints(body))
    result.extend(_extract_constraint_class_attrs(body))
    result.extend(_extract_python_constraints(body, parsed_methods))
    return result
