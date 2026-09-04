#!/usr/bin/env python3
"""Citation-decay gate: find `path:LINE` references that have drifted.

WHY THIS EXISTS
---------------
Plans and board files cite sources as `some/file.md:438`. `.claude/board/
EPIPHANIES.md` is APPEND-ONLY and PREPENDED to, so EVERY line number into it
shifts on every new entry. A citation therefore comes to point at a different
entry with NO edit on either side -- nothing in git blame, nothing in a diff.

Two measured instances (2026-09-04, both live on main at the time):

  1. A plan cited `EPIPHANIES.md:438` for a claim. By ratification day `:438`
     had become the heading of the ruling that RETRACTED that claim.
     A citation to a removal reads identically to a citation to the thing
     removed.
  2. The same plan cited `EPIPHANIES.md:13401`; the real target was `:13615`
     -- 214 lines off.

THE DECAY RULE (and why it is shaped this way)
----------------------------------------------
We cannot know what a citation "meant", so we never guess. We recover an
ANCHOR from the citing context and ask only a falsifiable question: does that
anchor still appear at the cited location?

  anchor extraction -- within +-ANCHOR_CONTEXT_CHARS of the citation, in
  priority order:
    1. a board id: `E-...-1` / `D-...` / `ISS-...` / `PROBE-...`
    2. a backticked symbol (>= MIN_ANCHOR_LEN chars, not the citation itself)
    3. a double-quoted phrase (>= MIN_ANCHOR_LEN chars)
  When several candidates sit in the window, the one NEAREST the citation
  wins (kind priority is only the tie-break). First-match-wins was tried and
  is wrong: on a dense page it hands every citation in a paragraph the same
  anchor, which turns one real defect into a block of false verdicts.
  The anchor must not be the cited path (or a component of it) -- a filename
  matching itself proves nothing.

  verdict:
    OK           -- anchor found within +-WINDOW_LINES of the cited line
    DECAYED      -- anchor recoverable AND absent from that window
    UNVERIFIABLE -- no anchor recoverable, or the target path does not resolve
                    in-repo, or the cited line is past EOF*

  *past-EOF is reported as UNVERIFIABLE rather than DECAYED: it is a real
   smell, but this gate only FAILS on evidence it can actually support.

Exit status is non-zero ONLY on DECAYED. UNVERIFIABLE is reported and does
not fail -- a gate that fires on everything carries exactly as much
information as one that never fires.

THE FIX THE GATE ASKS FOR
-------------------------
Do NOT correct the number. Correcting the number re-arms the same bomb on the
next prepend. Replace the line number with a stable anchor -- the heading text
or the D-id / E-id -- e.g. `EPIPHANIES.md` under `E-FOO-BAR-1` instead of
`EPIPHANIES.md:438`.

Pure stdlib. Usage:
    python3 .claude/tools/citation_decay.py [paths-or-globs ...]
    python3 .claude/tools/citation_decay.py --self-test
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import tempfile

WINDOW_LINES = 3
ANCHOR_CONTEXT_CHARS = 200
MIN_ANCHOR_LEN = 4
MAX_EXAMPLES = 10

# `path/to/file.ext:NNN` or `:NNN-MMM`. Paths may be bare basenames.
CITATION_RE = re.compile(
    # A LEADING DOT IS PART OF THE PATH, not a reason to reject the match.
    # The original lookbehind included `.`, which made every `.claude/...`
    # citation invisible -- i.e. most of the paths this workspace actually
    # cites. Found by a probe citation that the scanner silently ignored.
    # The dot is now consumed by the pattern and dropped from the lookbehind;
    # `(?<![\w/-])` still prevents matching a fragment mid-path.
    r"(?<![\w/-])((?:\.)?[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:md|rs|py|toml|yml|yaml|json|sh|txt))"
    # Trailing `.` is allowed (a sentence-final citation is ordinary prose);
    # `.<digit>` is not, so a version-ish `foo.md:1.2` never matches.
    r":(\d+)(?:-(\d+))?(?![\w-])(?!\.\d)"
)
ID_RE = re.compile(r"\b(?:E|D|ISS|PROBE|ADR|EXP)-[A-Z0-9][A-Z0-9-]{2,}\b")
BACKTICK_RE = re.compile(r"`([^`\n]{%d,})`" % MIN_ANCHOR_LEN)
QUOTED_RE = re.compile(r'"([^"\n]{%d,})"' % MIN_ANCHOR_LEN)

OK, DECAYED, UNVERIFIABLE = "OK", "DECAYED", "UNVERIFIABLE"

DEFAULT_GLOBS = [".claude/plans/*.md", ".claude/board/*.md"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def resolve_target(cited_path: str, citing_file: str, root: str) -> str | None:
    """Resolve a cited path in-repo. Bare basenames are NOT guessed."""
    if cited_path.count("/") == 0:
        return None  # a bare basename is ambiguous -> UNVERIFIABLE, never guessed
    cands = [
        os.path.join(root, cited_path),
        os.path.join(os.path.dirname(citing_file), cited_path),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def extract_anchor(text: str, cite_start: int, cite_end: int, cited_path: str):
    """Return (anchor, kind) recovered from the citation's neighbourhood."""
    lo = max(0, cite_start - ANCHOR_CONTEXT_CHARS)
    hi = min(len(text), cite_end + ANCHOR_CONTEXT_CHARS)
    ctx = text[lo:hi]
    cite_text = text[cite_start:cite_end]
    path_parts = {p.lower() for p in cited_path.split("/")} | {cited_path.lower()}

    def usable(cand: str) -> bool:
        c = cand.strip()
        if len(c) < MIN_ANCHOR_LEN:
            return False
        if c.lower() in path_parts:
            return False
        if cite_text in c or CITATION_RE.search(c):
            return False
        return True

    cands = []
    for prio, (rx, grp, kind) in enumerate(
        ((ID_RE, 0, "id"), (BACKTICK_RE, 1, "symbol"), (QUOTED_RE, 1, "phrase"))
    ):
        for m in rx.finditer(ctx):
            cand = m.group(grp).strip()
            if not usable(cand):
                continue
            # distance from the citation, in the ORIGINAL text's coordinates
            a, b = lo + m.start(), lo + m.end()
            dist = 0 if a < cite_end and b > cite_start else min(
                abs(a - cite_end), abs(cite_start - b)
            )
            cands.append((dist, prio, cand, kind))
    if not cands:
        return None, None
    cands.sort(key=lambda t: (t[0], t[1]))
    return cands[0][2], cands[0][3]


def check_anchor(target_lines: list[str], start: int, end: int, anchor: str) -> bool:
    lo = max(1, start - WINDOW_LINES)
    hi = min(len(target_lines), end + WINDOW_LINES)
    window = norm(" ".join(target_lines[lo - 1 : hi]))
    return norm(anchor) in window


class Finding:
    def __init__(self, citing, cite_line, cited, start, end, verdict, anchor, kind, why):
        self.citing, self.cite_line, self.cited = citing, cite_line, cited
        self.start, self.end = start, end
        self.verdict, self.anchor, self.kind, self.why = verdict, anchor, kind, why

    def __str__(self):
        span = f"{self.start}" if self.start == self.end else f"{self.start}-{self.end}"
        a = f' anchor={self.kind}:"{self.anchor}"' if self.anchor else ""
        return (
            f"{self.verdict:12s} {self.citing}:{self.cite_line} -> "
            f"{self.cited}:{span}{a} ({self.why})"
        )


def scan_file(path: str, root: str) -> list[Finding]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def line_of(off: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    out = []
    cache: dict[str, list[str]] = {}
    for m in CITATION_RE.finditer(text):
        cited, s, e = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        cl = line_of(m.start())
        tgt = resolve_target(cited, path, root)
        if tgt is None:
            out.append(Finding(path, cl, cited, s, e, UNVERIFIABLE, None, None,
                               "target path does not resolve in-repo"))
            continue
        anchor, kind = extract_anchor(text, m.start(), m.end(), cited)
        if anchor is None:
            out.append(Finding(path, cl, cited, s, e, UNVERIFIABLE, None, None,
                               "no anchor recoverable from citing context"))
            continue
        if tgt not in cache:
            with open(tgt, encoding="utf-8", errors="replace") as fh:
                cache[tgt] = fh.read().splitlines()
        lines = cache[tgt]
        if s > len(lines):
            out.append(Finding(path, cl, cited, s, e, UNVERIFIABLE, anchor, kind,
                               f"cited line is past EOF ({len(lines)} lines)"))
            continue
        if check_anchor(lines, s, e, anchor):
            out.append(Finding(path, cl, cited, s, e, OK, anchor, kind,
                               f"anchor present within +-{WINDOW_LINES} lines"))
        else:
            out.append(Finding(path, cl, cited, s, e, DECAYED, anchor, kind,
                               f"anchor absent within +-{WINDOW_LINES} lines"))
    return out


FIX_MESSAGE = """
::error::Citation decay detected.

THE FIX IS NOT TO CORRECT THE LINE NUMBER.
Correcting the number re-arms the same bomb on the next prepend: append-only
files (.claude/board/EPIPHANIES.md above all) are PREPENDED to, so every line
number into them shifts with no edit on either side.

Replace the line number with a STABLE ANCHOR:
  - the board id      ->  `EPIPHANIES.md` under `E-FOO-BAR-1`
  - the heading text  ->  `EPIPHANIES.md` section "the hygiene rule recursed"
A number is a coordinate in a moving frame; a heading or a D-id is an address.
"""


def added_lines(base: str, root: str) -> dict[str, set[int]] | None:
    """Line numbers ADDED or MODIFIED by `base...HEAD`, per file.

    Why this exists, and why the gate is useless without it: `EPIPHANIES.md`
    alone carries 10 pre-existing decayed citations, and this workspace's
    board-hygiene rule means nearly EVERY pull request touches it. A gate
    scoped to changed FILES would therefore fail almost every PR for decay it
    did not introduce -- and a guard that fires on everything carries exactly
    as much information as one that never fires (the `closed_class_guess`
    150/150 defect, root CLAUDE.md). Scoping to changed LINES makes it a
    no-new-decay gate: the 124-item backlog on main is a separate deliberate
    pass, not every contributor's problem.

    Returns None when the diff cannot be computed, and the caller then FAILS
    CLOSED rather than silently checking everything.
    """
    try:
        out = subprocess.run(
            ["git", "diff", "--unified=0", "--diff-filter=d", f"{base}...HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return None
    per: dict[str, set[int]] = {}
    cur: str | None = None
    for ln in out.splitlines():
        if ln.startswith("+++ b/"):
            cur = ln[6:]
            per.setdefault(cur, set())
        elif ln.startswith("@@") and cur is not None:
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            if m:
                start = int(m.group(1))
                count = int(m.group(2) or 1)
                per[cur].update(range(start, start + count))
    return per


def run(paths: list[str], root: str, only: dict[str, set[int]] | None = None) -> int:
    findings: list[Finding] = []
    for p in sorted(set(paths)):
        if os.path.isfile(p):
            got = scan_file(p, root)
            if only is not None:
                keep = only.get(os.path.relpath(p, root).replace(os.sep, "/"), set())
                got = [f for f in got if f.cite_line in keep]
            findings.extend(got)
    counts = {OK: 0, DECAYED: 0, UNVERIFIABLE: 0}
    for f in findings:
        counts[f.verdict] += 1
    decayed = [f for f in findings if f.verdict == DECAYED]
    unver = [f for f in findings if f.verdict == UNVERIFIABLE]

    print(f"citation-decay: {len(findings)} citations in {len(set(paths))} files")
    print(f"  OK           {counts[OK]}")
    print(f"  DECAYED      {counts[DECAYED]}")
    print(f"  UNVERIFIABLE {counts[UNVERIFIABLE]}  (reported, does not fail)")
    if decayed:
        print("\nDECAYED:")
        for f in decayed[:MAX_EXAMPLES]:
            print("  " + str(f))
        if len(decayed) > MAX_EXAMPLES:
            print(f"  ... and {len(decayed) - MAX_EXAMPLES} more")
    if unver:
        print(f"\nUNVERIFIABLE (first {min(MAX_EXAMPLES, len(unver))}):")
        for f in unver[:MAX_EXAMPLES]:
            print("  " + str(f))
        if len(unver) > MAX_EXAMPLES:
            print(f"  ... and {len(unver) - MAX_EXAMPLES} more")
    if decayed:
        print(FIX_MESSAGE)
        return 1
    print("\nno confirmed citation decay")
    return 0


def self_test() -> int:
    """Prove BOTH halves: it FIRES on a moved anchor, STAYS SILENT otherwise."""
    d = tempfile.mkdtemp(prefix="citation-decay-selftest-")
    sub = os.path.join(d, "board")
    os.makedirs(sub)
    target = os.path.join(sub, "TARGET.md")
    with open(target, "w") as fh:
        fh.write(
            "\n".join(
                [
                    "line 1 filler",                    # 1
                    "line 2 filler",                    # 2
                    "## E-MOVED-ANCHOR-1 heading",      # 3  <- the real home
                    "body of the moved anchor",         # 4
                    "line 5 filler",                    # 5
                    "line 6 filler",                    # 6
                    "line 7 filler",                    # 7
                    "## E-STABLE-ANCHOR-1 heading",     # 8  <- correct citation
                    "body of the stable anchor",        # 9
                    "line 10 filler",                   # 10
                    "line 11 filler",                   # 11
                    "line 12 filler",                   # 12
                ]
            )
            + "\n"
        )
    citing = os.path.join(sub, "CITING.md")
    with open(citing, "w") as fh:
        fh.write(
            # (1) FIRES: the anchor's real home is :3, the citation says :9.
            "The ruling E-MOVED-ANCHOR-1 is recorded at board/TARGET.md:9.\n"
            + "prose padding line that carries no anchor at all. " * 8 + "\n"
            # (2) SILENT: correct citation.
            "The ruling E-STABLE-ANCHOR-1 is at board/TARGET.md:8 and still is.\n"
            + "more prose padding with nothing citable in it whatsoever. " * 8 + "\n"
            # (3) SILENT (UNVERIFIABLE): nothing anchor-shaped nearby.
            "See board/TARGET.md:5 for the rest of it.\n"
            + "still more padding, deliberately anchor-free on both sides. " * 8 + "\n"
            # (4) SILENT (UNVERIFIABLE): path does not resolve in-repo.
            "And board/DOES-NOT-EXIST.md:12 for `some anchor text here`.\n"
        )
    findings = scan_file(citing, d)
    by = {}
    for f in findings:
        by[(f.cited, f.start)] = f

    print("--- self-test findings ---")
    for f in findings:
        print("  " + str(f))

    expect = [
        (("board/TARGET.md", 9), DECAYED, "moved anchor must FIRE"),
        (("board/TARGET.md", 8), OK, "correct citation must stay SILENT"),
        (("board/TARGET.md", 5), UNVERIFIABLE, "no recoverable anchor"),
        (("board/DOES-NOT-EXIST.md", 12), UNVERIFIABLE, "path not in-repo"),
    ]
    failures = []
    for key, want, why in expect:
        got = by.get(key)
        actual = got.verdict if got else "MISSING"
        mark = "PASS" if actual == want else "FAIL"
        print(f"  [{mark}] {key[0]}:{key[1]} expected {want}, got {actual}  ({why})")
        if mark == "FAIL":
            failures.append(key)
    print("--- self-test " + ("FAILED ---" if failures else "PASSED ---"))
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    root = os.getcwd()
    if "--self-test" in argv:
        return self_test()
    only = None
    if "--added-lines-only" in argv:
        i = argv.index("--added-lines-only")
        base = argv[i + 1]
        only = added_lines(base, root)
        if only is None:
            print(
                f"citation-decay: ERROR: cannot diff '{base}...HEAD' -- refusing to "
                "check every line instead (in CI this usually means a shallow "
                "checkout; the workflow needs `fetch-depth: 0`).",
                file=sys.stderr,
            )
            return 1
        argv = argv[:i] + argv[i + 2:]
    pats = [a for a in argv if not a.startswith("-")] or DEFAULT_GLOBS
    paths: list[str] = []
    for p in pats:
        hits = glob.glob(p, recursive=True)
        paths.extend(hits if hits else ([p] if os.path.isfile(p) else []))
    if not paths:
        print("citation-decay: no input files matched", file=sys.stderr)
        return 0
    return run(paths, root, only)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
