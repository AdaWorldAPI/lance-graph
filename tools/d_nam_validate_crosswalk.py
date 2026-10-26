#!/usr/bin/env python3
"""Validate D-NAM's *documentation* crosswalk, not semantic equivalence.

No external dependencies; used at review time. Does not import or dispatch ONNX,
TFLite, MLIR, Quack, R2IL or ndarray and grants no operator admission.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / ".claude/specs/d-nam-method-crosswalk-v1.json"
DIALECTS = {"ONNX", "TFLite", "MLIR", "RelAlg"}
STATES = {"candidate", "admitted", "refused"}


def validate(obj):
    if obj.get("schema") != "d-nam-crosswalk-v1":
        raise ValueError("unsupported crosswalk schema")
    methods = obj.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("no native methods")
    ids = set()
    for method in methods:
        ident = method.get("native_id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError(f"invalid/duplicate native_id: {ident!r}")
        ids.add(ident)
        if not method.get("native_anchor") or not method.get("semantic"):
            raise ValueError(f"{ident}: missing native anchor or semantics")
        dialects = set()
        for row in method.get("readings", []):
            d = row.get("dialect")
            state = row.get("status")
            if d not in DIALECTS or d in dialects:
                raise ValueError(f"{ident}: unknown/duplicate dialect {d!r}")
            dialects.add(d)
            if state not in STATES or not row.get("expression"):
                raise ValueError(f"{ident}/{d}: incomplete reading")
            if state == "admitted" and (not row.get("version") or not row.get("oracle") or not row.get("evidence")):
                raise ValueError(f"{ident}/{d}: admitted requires pinned version, independent oracle and evidence")
            if state == "refused" and not row.get("reason"):
                raise ValueError(f"{ident}/{d}: refusal requires reason")
        if dialects != DIALECTS:
            raise ValueError(f"{ident}: expected all four dialect readings")
        if not method.get("refusal"):
            raise ValueError(f"{ident}: missing negative cases")
    if len(ids) != 4:
        raise ValueError(f"expected four seed methods, got {len(ids)}")
    return len(ids)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        count = validate(obj)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"D-NAM crosswalk REFUSE: {exc}", file=sys.stderr)
        return 1
    print(f"D-NAM crosswalk schema OK ({count} candidates; NOT semantic admission)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
