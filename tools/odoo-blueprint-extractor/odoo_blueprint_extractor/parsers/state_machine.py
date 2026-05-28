"""Detect state machines from Odoo class bodies.

A state machine is detected when:
- A `fields.Selection` field named 'state' (or ending in '_state') exists
- The Selection has a list of (key, label) tuples as first arg

Transitions are found by scanning action_* methods for:
    self.write({'state': '<key>'})

State semantics are classified via a keyword table.
"""

import ast
from typing import Any, Dict, List, Optional, Tuple


# Map from state key (or fragment) to OdooStateSemantic Rust variant
_SEMANTIC_TABLE: List[Tuple[str, str]] = [
    ("draft", "Draft"),
    ("new", "Draft"),
    ("open", "Active"),
    ("active", "Active"),
    ("confirm", "Active"),
    ("validated", "Active"),
    ("posted", "Posted"),
    ("done", "Completed"),
    ("complete", "Completed"),
    ("paid", "Completed"),
    ("closed", "Completed"),
    ("in_progress", "InProgress"),
    ("progress", "InProgress"),
    ("running", "InProgress"),
    ("cancel", "Cancelled"),
    ("cancelled", "Cancelled"),
    ("refused", "Cancelled"),
    ("reject", "Cancelled"),
    ("expire", "Terminal"),
    ("terminal", "Terminal"),
    ("archive", "Terminal"),
]


def _classify_state_key(key: str) -> str:
    k = key.lower()
    for fragment, semantic in _SEMANTIC_TABLE:
        if fragment in k:
            return semantic
    return "Active"  # Default fallback


def _extract_selection_states(call: ast.Call) -> Optional[List[Dict[str, str]]]:
    """Extract [(key, label), ...] from a fields.Selection call's first arg."""
    if not call.args:
        return None
    first_arg = call.args[0]
    if not isinstance(first_arg, ast.List):
        return None

    states = []
    for elt in first_arg.elts:
        if not isinstance(elt, ast.Tuple) or len(elt.elts) < 2:
            continue
        key_node = elt.elts[0]
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            key = key_node.value
            states.append({"name": key, "semantic": _classify_state_key(key)})
    return states if states else None


def _find_state_write(func: ast.FunctionDef, state_keys: List[str]) -> Optional[str]:
    """Find self.write({'state': '<key>'}) in function body; return the target key."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        # self.write(...)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "write"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Dict):
            continue
        for k, v in zip(first_arg.keys, first_arg.values):
            if not isinstance(k, ast.Constant) or k.value != "state":
                continue
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                if v.value in state_keys:
                    return v.value
    return None


def extract_state_machine(
    body: List[ast.stmt],
    parsed_fields: List[Dict[str, Any]],
    parsed_methods: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detect and return a state machine dict if the class has one."""
    # Find the state field among parsed fields
    state_field_name: Optional[str] = None
    state_call_node: Optional[ast.Call] = None

    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        fname = stmt.targets[0].id
        if fname != "state" and not fname.endswith("_state"):
            continue
        val = stmt.value
        if not isinstance(val, ast.Call):
            continue
        func = val.func
        if not isinstance(func, ast.Attribute) or func.attr != "Selection":
            continue
        state_field_name = fname
        state_call_node = val
        break

    if state_field_name is None or state_call_node is None:
        return None

    states = _extract_selection_states(state_call_node)
    if not states:
        return None

    state_keys = [s["name"] for s in states]

    # Find transitions via action_* methods
    transitions: List[Dict[str, Any]] = []
    for stmt in body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        if not stmt.name.startswith("action_"):
            continue
        target_key = _find_state_write(stmt, state_keys)
        if target_key is not None:
            # Heuristic: from = the key BEFORE target in the states list (or "draft")
            target_idx = state_keys.index(target_key)
            if target_idx > 0:
                from_key = state_keys[target_idx - 1]
            else:
                from_key = state_keys[0]
            transitions.append(
                {
                    "from": from_key,
                    "to": target_key,
                    "trigger": stmt.name,
                    "guards": [],
                }
            )

    return {
        "state_field": state_field_name,
        "states": states,
        "transitions": transitions,
    }
