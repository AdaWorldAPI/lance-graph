#!/usr/bin/env python3
"""Every NEWLY ADDED plan must carry at least one D-id.

Why a property and not a count
------------------------------
A census of untracked plans is a function of the regex used to take it. The
same tree, on 2026-09-03, measured **53** (a sibling session's sweep), **75**
(the generator's own pattern) and **102** (a stricter pattern requiring a
trailing number) — identical population, identical `3DGS` family of 19, three
different totals. A gate asserting "no more than N untracked plans" would have
been wrong on the day it landed and would encode whichever regex its author
happened to hold.

So this gate asserts something regex-choice cannot move: *a plan added in this
PR cites at least one D-id.* That stops the backlog regrowing, which is the
stated goal, without requiring the backlog to be agreed on or backfilled — a
cross-session scope call nobody has made.

Why it does not fire on MODIFIED plans
--------------------------------------
Gating modifications would make every edit to a pre-existing untracked plan a
blocked PR, punishing whoever next touches an old file for a debt they did not
create. Added-only is the minimal form that prevents regrowth.

Why the pattern is READ from the generator and not copied
--------------------------------------------------------
`supersession_index.py` owns the D-id pattern; the index's coverage column is
computed with it. A second copy here would agree with it exactly until one was
edited, which is when nobody is comparing them — the drift failure this
workspace ruled on in `E-A-CITATION-IS-NOT-A-DEPENDENCY-AND-A-FORCED-COPY-NEEDS-A-GATE-1`.

`import` was the obvious way to share it and is WRONG here: that module has no
`if __name__ == "__main__"` guard, so importing it runs the whole generator and
prints the index to stdout. (Measured — the first version of this file did
exactly that.) Adding a guard would be a refactor of a CI-gated tool for one
caller's convenience, so instead the pattern is lifted from its source text.
There is exactly one definition and it is still the single source; if that line
is ever renamed or reshaped this raises, which is the correct failure — a silent
fallback to a local copy is the very drift being avoided.
"""

import pathlib
import re
import sys

_GEN = pathlib.Path(__file__).resolve().parent / "supersession_index.py"
_DEF = re.compile(r"^DID\s*=\s*re\.compile\(r'(?P<pat>.*)'\)\s*$", re.M)


def _did_pattern() -> "re.Pattern[str]":
    """The generator's own D-id pattern, read from its source."""
    m = _DEF.search(_GEN.read_text(errors="ignore"))
    if not m:
        raise SystemExit(
            f"plan-dids: could not find the `DID = re.compile(r'...')` definition in "
            f"{_GEN}. The pattern moved or was renamed. Fix this extractor rather "
            f"than copying the pattern here — a second copy is the drift this gate exists to avoid."
        )
    return re.compile(m.group("pat"))


DID = _did_pattern()


def untracked(paths: list[str]) -> list[str]:
    """Return the subset of `paths` carrying no D-id."""
    out = []
    for p in paths:
        f = pathlib.Path(p)
        if not f.is_file():
            continue  # deleted or renamed away in the same PR
        if not DID.search(f.read_text(errors="ignore")):
            out.append(p)
    return out


def main(argv: list[str]) -> int:
    paths = [a for a in argv if a.endswith(".md")]
    if not paths:
        print("plan-dids: no added plan files in this diff; nothing to check")
        return 0

    missing = untracked(paths)
    for p in paths:
        print(f"  {'MISSING D-id' if p in missing else 'ok          '}  {p}")

    if not missing:
        print(f"plan-dids: {len(paths)} added plan(s), all carry a D-id")
        return 0

    print()
    print("::error::A plan added in this PR carries no D-id.")
    print("A plan without D-ids is invisible to every discovery path: STATUS_BOARD")
    print("has nothing to hold, and the supersession index's coverage column has")
    print("nothing to count. Mint ids for its sections and add the rows, the way")
    print("#1155 did retroactively for a sibling plan.")
    print()
    print(f"Pattern (imported from supersession_index.py): {DID.pattern}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
