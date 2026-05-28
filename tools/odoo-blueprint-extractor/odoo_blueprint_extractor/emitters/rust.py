"""Format one ParsedClass as a Rust `pub const` OdooEntity literal.

Const name convention: EXT_<UPPER_SNAKE_MODEL_NAME>
  e.g. uom.uom   -> EXT_UOM_UOM
       account.move -> EXT_ACCOUNT_MOVE

Output is indented with 4 spaces and uses trailing commas — rustfmt-friendly.
"""

import re
from typing import Any, Dict, List, Optional


def _model_name_to_const(model_name: str) -> str:
    """Convert 'uom.uom' -> 'EXT_UOM_UOM'."""
    ident = re.sub(r"[^a-zA-Z0-9]", "_", model_name).upper()
    return f"EXT_{ident}"


def _rust_str(s: str) -> str:
    """Emit a Rust &'static str literal, escaping backslashes and double quotes."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _rust_opt_str(s: Optional[str]) -> str:
    if s is None:
        return "None"
    return f"Some({_rust_str(s)})"


def _rust_bool(b: bool) -> str:
    return "true" if b else "false"


def _rust_str_slice(items: List[str]) -> str:
    if not items:
        return "&[]"
    inner = ", ".join(_rust_str(i) for i in items)
    return f"&[{inner}]"


def _emit_field(f: Dict[str, Any], indent: str) -> str:
    i2 = indent + "    "
    lines = [
        f"{indent}OdooField {{",
        f"{i2}name: {_rust_str(f['name'])},",
        f"{i2}kind: OdooFieldKind::{f['kind']},",
        f"{i2}target: {_rust_opt_str(f.get('target'))},",
        f"{i2}required: {_rust_bool(f.get('required', False))},",
        f"{i2}computed: {_rust_opt_str(f.get('computed'))},",
        f"{i2}depends: {_rust_str_slice(f.get('depends', []))},",
        f"{i2}semantic_role: OdooSemanticRole::Other,",
        f"{indent}}}",
    ]
    return "\n".join(lines)


def _emit_method(m: Dict[str, Any], indent: str) -> str:
    i2 = indent + "    "
    lines = [
        f"{indent}OdooMethod {{",
        f"{i2}name: {_rust_str(m['name'])},",
        f"{i2}kind: OdooMethodKind::{m['kind']},",
        f"{i2}return_kind: OdooReturnKind::{m['return_kind']},",
        f"{i2}triggers: &[],",
        f"{indent}}}",
    ]
    return "\n".join(lines)


def _emit_decorator(d: Dict[str, Any], indent: str) -> str:
    i2 = indent + "    "
    lines = [
        f"{indent}OdooDecorator {{",
        f"{i2}kind: OdooDecoratorKind::{d['kind']},",
        f"{i2}targets: {_rust_str_slice(d.get('targets', []))},",
        f"{indent}}}",
    ]
    return "\n".join(lines)


def _emit_state(s: Dict[str, Any], indent: str) -> str:
    return (
        f"{indent}OdooState {{ "
        f"name: {_rust_str(s['name'])}, "
        f"semantic: OdooStateSemantic::{s['semantic']} }}"
    )


def _emit_transition(t: Dict[str, Any], indent: str) -> str:
    i2 = indent + "    "
    guards = t.get("guards", [])
    lines = [
        f"{indent}OdooTransition {{",
        f"{i2}from: {_rust_str(t['from'])},",
        f"{i2}to: {_rust_str(t['to'])},",
        f"{i2}trigger: {_rust_str(t['trigger'])},",
        f"{i2}guards: {_rust_str_slice(guards)},",
        f"{indent}}}",
    ]
    return "\n".join(lines)


def _emit_constraint(c: Dict[str, Any], indent: str) -> str:
    i2 = indent + "    "
    lines = [
        f"{indent}OdooConstraint {{",
        f"{i2}kind: OdooConstraintKind::{c['kind']},",
        f"{i2}condition: {_rust_str(c['condition'])},",
        f"{i2}source_method: {_rust_opt_str(c.get('source_method'))},",
        f"{indent}}}",
    ]
    return "\n".join(lines)


def _emit_state_machine(sm: Optional[Dict[str, Any]], indent: str) -> str:
    """Emit the state_machine field value (Option expression)."""
    if sm is None:
        return "None"

    i2 = indent + "    "
    i3 = i2 + "    "

    states_lines = []
    for s in sm["states"]:
        states_lines.append(_emit_state(s, i3) + ",")

    transitions_lines = []
    for t in sm["transitions"]:
        transitions_lines.append(_emit_transition(t, i3) + ",")

    states_block = "\n".join(states_lines) if states_lines else ""
    transitions_block = "\n".join(transitions_lines) if transitions_lines else ""

    sm_lines = [
        f"Some(&OdooStateMachine {{",
        f"{i2}state_field: {_rust_str(sm['state_field'])},",
        f"{i2}states: &[",
    ]
    if states_block:
        sm_lines.append(states_block)
    sm_lines.append(f"{i2}],")
    sm_lines.append(f"{i2}transitions: &[")
    if transitions_block:
        sm_lines.append(transitions_block)
    sm_lines.append(f"{i2}],")
    sm_lines.append(f"{indent}}})")
    return "\n".join(sm_lines)


def emit_entity(parsed_class: Any, addon_path: str) -> str:
    """Emit one pub const OdooEntity = OdooEntity { ... }; as a string.

    *parsed_class* is a ParsedClass instance from parsers/classes.py.
    *addon_path*   is the absolute path to the addon root.
    """
    pc = parsed_class
    const_name = _model_name_to_const(pc.model_name or pc.class_name)
    i1 = "    "   # 4 spaces
    i2 = "        "  # 8 spaces

    # Build slice literals
    fields_str = _build_slice(pc.fields, _emit_field, i2)
    methods_str = _build_slice(pc.methods, _emit_method, i2)
    decorators_str = _build_slice(pc.decorators, _emit_decorator, i2)
    constraints_str = _build_slice(pc.constraints, _emit_constraint, i2)

    # State machine
    sm_str = _emit_state_machine(pc.state_machine, i2)

    # Provenance
    # Strip /home/user/ prefix to make paths repo-relative (B2/CodeRabbit fix).
    rel_path = pc.source_file
    if rel_path and rel_path.startswith("/home/user/"):
        rel_path = rel_path[len("/home/user/"):]
    source_ref = (
        f"OdooSourceRef {{ "
        f"path: {_rust_str(rel_path)}, "
        f"line_range: ({pc.line_start}, {pc.line_end}) }}"
    )
    regulation_iri_str = _rust_str_slice(pc.regulation_iris)
    description = pc.description or ""

    lines = [
        f"pub const {const_name}: OdooEntity = OdooEntity {{",
        f"{i1}model_name: {_rust_str(pc.model_name or pc.class_name)},",
        f"{i1}kind: OdooEntityKind::{pc.kind},",
        f"{i1}description: {_rust_str(description)},",
        f"{i1}fields: &[",
        fields_str,
        f"{i1}],",
        f"{i1}methods: &[",
        methods_str,
        f"{i1}],",
        f"{i1}decorators: &[",
        decorators_str,
        f"{i1}],",
        f"{i1}state_machine: {sm_str},",
        f"{i1}constraints: &[",
        constraints_str,
        f"{i1}],",
        f"{i1}provenance: OdooProvenance {{",
        f"{i2}l_doc: \"\",",
        f"{i2}l_doc_lines: (0, 0),",
        f"{i2}odoo_source: &[{source_ref}],",
        f"{i2}confidence: OdooConfidence::Extracted,",
        f"{i2}regulation_iri: {regulation_iri_str},",
        f"{i1}}},",
        "};",
    ]
    return "\n".join(lines)


def _build_slice(
    items: List[Any],
    emitter,
    inner_indent: str,
) -> str:
    """Build the inner lines of a &[...] slice literal."""
    if not items:
        return ""
    parts = []
    for item in items:
        parts.append(emitter(item, inner_indent) + ",")
    return "\n".join(parts)
