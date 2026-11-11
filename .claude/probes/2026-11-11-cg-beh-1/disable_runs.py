#!/usr/bin/env python3
"""Disable runs for beh.py (CG-BEH-1): each case applies ONE textual mutation to a temporary copy,
runs the copy's `--self-test`, and requires the test that belongs to the mutated guard to fail.

A case whose test stays green is reported GREEN (guard not load-bearing!) and the exit status is 1.
A mutation whose anchor does not occur exactly once aborts the run: a disable that does not apply
looks the same as a guard that does not bind. The unmutated copy must pass first.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

BEH = Path(__file__).with_name("beh.py")

# (test, old, new): `old` occurs exactly once in beh.py.
CASES = [
    ("test_01", 'w["opcode"] not in dispatched:', 'w["opcode"] in dispatched:'),
    ("test_02", "if witness_fnv is not None and witness_fnv != current:", "if False:"),
    ("test_03", 'arms = [d for d in disp if d["symbol"] == symbol]',
     'arms = [d for d in disp if d["symbol"] == symbol or symbol.split("::")[-1].upper() == d["opcode"]]'),
    ("test_04", "        if not us:\n", "        if False:\n"),
    ("test_05", "    if not arms:\n", "    if False:\n"),
    ("test_06", 'if exe != [e["opcode"] for e in entries]:', "if False:"),
    ("test_07", "if wi >= nwords:", "if False:"),
    ("test_08", "if lens != LENS:", "if False:"),
    ("test_09", '",".join(sorted(classes))', '",".join(classes)'),
    ("test_10", 'arms = [d for d in arms if d["arm_line"] <= line <= d["arm_end"]]', "arms = arms"),
    ("test_11", 'u["ord_int"]))', "0))"),
    ("test_12", "    if not p.is_file():\n        raise ArtifactError(\"ARTIFACT_MISSING\", str(p))\n    cols, rows",
     "    cols, rows"),
]


def run_selftest(path):
    p = subprocess.run([sys.executable, "-I", str(path), "--self-test"], capture_output=True, text=True)
    failing = sorted(set(re.findall(r"^(?:FAIL|ERROR): (test_\w+)", p.stderr + p.stdout, flags=re.M)))
    return p.returncode, failing


def main():
    src = BEH.read_text()
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.py"
        base.write_text(src)
        code, failing = run_selftest(base)
        if code != 0:
            print(f"BASELINE\tself-test fails before any mutation\t{','.join(failing)}")
            return 1
        bad = 0
        for i, (test, old, new) in enumerate(CASES):
            assert src.count(old) == 1, f"anchor for {test} occurs {src.count(old)} times"
            mut = Path(td) / f"mut_{i}.py"
            mut.write_text(src.replace(old, new))
            _, failing = run_selftest(mut)
            red = any(f.startswith(test) for f in failing)
            bad += not red
            print(f"{test}\t{'RED' if red else 'GREEN (guard not load-bearing!)'}\t{','.join(failing) or '-'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
