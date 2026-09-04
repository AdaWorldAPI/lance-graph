#!/usr/bin/env python3
"""Fail a PR that SHORTENS an append-only board file.

The protected files below are append-only logs: every session PREPENDS its
newest entry at the head. Concurrent sessions therefore collide at exactly the
same lines, by construction -- and the tempting resolution ("take mine", "take
theirs", `git checkout --ours`) silently DELETES the other session's entry. The
loss is invisible in review: the file still parses, still reads well, and the
only trace is that it got shorter.

The workspace law (root `CLAUDE.md`; the full statement plus its incident
receipts live in
`.claude/knowledge/never-truncate-a-file-you-still-need-to-read.md`):

    an append-only file that got SHORTER is always a defect.

That check has been manual (`wc -l` after every ledger write). On 2026-09-04
three sessions collided within one hour and only a hand-run `wc -l` prevented a
loss. This makes it mechanical.

WHAT IT IS NOT
--------------
This gate measures LINE COUNT ONLY. It cannot see a same-length rewrite, a
reordering, or an entry replaced by another of equal size. It catches the one
failure mode that is both the most common and the most silent -- a dropped
prepend -- and claims nothing more. A green run is not proof the file is
append-only; a red run is proof it is not.

USAGE
-----
    python3 .claude/tools/append_only_gate.py [BASE_REF]     # default origin/main
    python3 .claude/tools/append_only_gate.py --self-test

Exit 0 = no protected file shrank. Exit 1 = at least one did (or the gate could
not do its job -- an unresolvable base ref fails closed, never silently green).
"""

from __future__ import annotations

import subprocess
import sys

DEFAULT_BASE = "origin/main"

# The append-only board files, per root `CLAUDE.md` -- "The governance files are
# APPEND-ONLY (prepend new entries; never edit past entries except the
# `**Status:**` / `**Confidence:**` lines)".
#
# Deliberately NOT the whole of `.claude/board/`: several files there are
# genuinely rewritable (the generated `SUPERSESSION-INDEX.md`, working
# scratch/roadmap files), and protecting them would make the gate fire on
# correct work -- a gate that objects to everything carries exactly as much
# information as one that never fires.
# The canonical eight are the ones `.claude/BOOT.md`'s immutability table names
# and `.claude/settings.json` denies `Edit`/`Write`/`MultiEdit` on. The first
# version of this tuple SUBSTITUTED `AGENT_LOG.md` for `IDEAS.md` and kept the
# count at eight, which is exactly why the substitution looked right -- leaving
# a real 1204-line append-only ledger unguarded while `.claude/board/**`
# happily started this workflow on every PR that truncated it. Found by review,
# not by the gate. `AGENT_LOG.md` stays: CLAUDE.md's one-writer rule calls it
# append-only too, so the set is the canonical eight PLUS that one -- nine.
PROTECTED = (
    ".claude/board/LATEST_STATE.md",
    ".claude/board/EPIPHANIES.md",
    ".claude/board/PR_ARC_INVENTORY.md",
    ".claude/board/STATUS_BOARD.md",
    ".claude/board/ISSUES.md",
    ".claude/board/IDEAS.md",
    ".claude/board/TECH_DEBT.md",
    ".claude/board/AGENT_LOG.md",
    ".claude/board/INTEGRATION_PLANS.md",
)

REMEDY = """
  This is almost always a PREPEND CONFLICT resolved by picking a side.
  Two sessions each added a new entry at the head of the same append-only
  file; taking one side dropped the other session's entry entirely.

  DO NOT resolve it by keeping "the newer" or "the bigger" side.
  RESOLVE IT BY KEEPING BOTH ENTRIES -- put both new blocks at the head,
  in whatever order reads correctly, and leave every pre-existing entry
  below them untouched. Nothing below the head should have moved at all.

  To see exactly what went missing:
      git diff {base}...HEAD -- {paths}

  If a shrink is genuinely intended (a deliberate, operator-sanctioned
  removal), say so explicitly in the PR body -- and note that the storno
  convention is to REGRADE an entry in place, never to delete it.
"""


class GateError(RuntimeError):
    """The gate could not do its job. Fails closed, never silently green."""


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def resolve_base(base_ref: str) -> str:
    """Merge-base of HEAD and the base ref.

    A PR branch's base ref moves on after the branch was cut, so a straight
    `git show <base>:<path>` would compare against work the branch never saw --
    an unrelated session's later prepend would read as "grew", masking a real
    shrink. The merge-base is the file as the branch actually inherited it.
    """
    proc = _run(["git", "merge-base", "HEAD", base_ref])
    if proc.returncode != 0:
        raise GateError(
            f"cannot compute merge-base of HEAD and {base_ref!r}: "
            f"{proc.stderr.strip()}\n"
            "  (in CI this usually means a shallow checkout -- the workflow "
            "needs `fetch-depth: 0`)"
        )
    return proc.stdout.strip()


def line_count_at(rev: str, path: str) -> int | None:
    """Line count of `path` at `rev`, or None when the file does not exist there."""
    proc = _run(["git", "show", f"{rev}:{path}"])
    if proc.returncode != 0:
        return None
    return count_lines(proc.stdout)


def line_count_in_tree(path: str) -> int | None:
    """Line count of the working-tree file, or None when absent.

    Read from the tree rather than from HEAD so the gate is equally usable
    locally on uncommitted work, which is where a session can still fix a
    clobber cheaply.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return count_lines(fh.read())
    except FileNotFoundError:
        return None


def count_lines(text: str) -> int:
    """Number of lines, counting a final unterminated line.

    `str.count("\\n")` alone under-counts a file with no trailing newline by
    one, which would read as a one-line shrink on an otherwise untouched file.
    """
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def evaluate(before: int | None, after: int | None) -> tuple[str, str]:
    """Classify one file. Returns (verdict, human note).

    verdict is one of: "ok", "new", "shrank", "deleted".
    """
    if before is None and after is None:
        return "ok", "absent at base and in head (nothing to check)"
    if before is None:
        return "new", f"new file, {after} lines (absent at base -- fine)"
    if after is None:
        return "deleted", f"DELETED (was {before} lines at base)"
    if after < before:
        return "shrank", f"{before} -> {after} lines ({after - before})"
    if after == before:
        return "ok", f"{before} lines, unchanged length"
    return "ok", f"{before} -> {after} lines (+{after - before})"


def check(base_ref: str, paths=PROTECTED) -> int:
    base_sha = resolve_base(base_ref)
    print(f"append-only gate: base {base_ref} -> merge-base {base_sha[:12]}")

    failures = []
    for path in paths:
        before = line_count_at(base_sha, path)
        after = line_count_in_tree(path)
        verdict, note = evaluate(before, after)
        marker = "FAIL" if verdict in ("shrank", "deleted") else "ok  "
        print(f"  {marker}  {path}: {note}")
        if verdict in ("shrank", "deleted"):
            failures.append((path, verdict, before, after))

    if not failures:
        print(f"\nOK: no protected append-only file shrank ({len(paths)} checked).")
        return 0

    print("\n" + "=" * 72)
    print("APPEND-ONLY VIOLATION: a board file got SHORTER.")
    print("=" * 72)
    for path, verdict, before, after in failures:
        if verdict == "deleted":
            print(f"\n  {path}\n      DELETED  (base: {before} lines, head: file absent)")
        else:
            print(
                f"\n  {path}\n"
                f"      base: {before} lines\n"
                f"      head: {after} lines\n"
                f"      delta: {after - before} lines LOST"
            )
    print(
        REMEDY.format(
            base=base_ref,
            paths=" ".join(p for p, _, _, _ in failures),
        )
    )
    return 1


# --------------------------------------------------------------------------
# Self-test. Both halves are required: a gate that cannot fire and a gate that
# fires on everything carry the same information (zero). These run in-process
# against `evaluate`, the single place the verdict is decided, so they need no
# git repo and no fixtures.
# --------------------------------------------------------------------------

SELF_TEST_CASES = [
    # (name, before, after, expected_verdict)
    ("shortened file FIRES", 500, 480, "shrank"),
    ("one line lost FIRES", 500, 499, "shrank"),
    ("whole file deleted FIRES", 500, None, "deleted"),
    ("grown file is SILENT", 500, 530, "ok"),
    ("unchanged file is SILENT", 500, 500, "ok"),
    ("newly created file is SILENT", None, 42, "new"),
    ("absent both sides is SILENT", None, None, "ok"),
]


def self_test() -> int:
    print("append-only gate self-test")
    print("-" * 72)
    failed = 0
    for name, before, after, expected in SELF_TEST_CASES:
        verdict, note = evaluate(before, after)
        fires = verdict in ("shrank", "deleted")
        want_fires = expected in ("shrank", "deleted")
        ok = verdict == expected
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:<32} "
            f"before={str(before):>4} after={str(after):>4} "
            f"-> {verdict:<7} ({'fires' if fires else 'silent'}) | {note}"
        )
        if not ok:
            failed += 1
            print(f"          expected verdict {expected!r}, got {verdict!r}")

    # Anti-vacuity: the suite must contain BOTH a case that fires and a case
    # that stays silent, or it proves nothing about discrimination.
    fired = sum(1 for _, b, a, _ in SELF_TEST_CASES if evaluate(b, a)[0] in ("shrank", "deleted"))
    silent = len(SELF_TEST_CASES) - fired
    print("-" * 72)
    print(f"  discrimination: {fired} case(s) fire, {silent} case(s) stay silent")
    if fired == 0:
        print("  [FAIL] no case fires -- a gate that cannot fire is not a gate")
        failed += 1
    if silent == 0:
        print("  [FAIL] no case stays silent -- a gate that fires on everything is not a gate")
        failed += 1

    if failed:
        print(f"\nSELF-TEST FAILED: {failed} problem(s).")
        return 1
    print(f"\nSELF-TEST PASSED: {len(SELF_TEST_CASES)} cases, both halves proven.")
    return 0


def main(argv: list[str]) -> int:
    args = argv[1:]
    if args and args[0] in ("--self-test", "--selftest"):
        return self_test()
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    base_ref = args[0] if args else DEFAULT_BASE
    try:
        return check(base_ref)
    except GateError as exc:
        print(f"append-only gate: ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
