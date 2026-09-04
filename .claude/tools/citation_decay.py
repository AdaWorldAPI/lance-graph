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
    python3 .claude/tools/citation_decay.py --since BASE
    python3 .claude/tools/citation_decay.py --self-test

`--since BASE` is the CI-gating mode: it compares citation verdicts at
`merge-base(BASE, HEAD)` against verdicts at HEAD (the corpus is discovered
by globbing DEFAULT_GLOBS at each revision -- no file list is taken), and
fails ONLY on a citation that is DECAYED at HEAD and was not DECAYED at base.
This replaces an earlier `--added-lines-only BASE <files...>` mode that
filtered by which LINES a PR added -- wrong for this gate's own motivating
case, `EPIPHANIES.md`: a prepend decays citations sitting on UNCHANGED lines
in UNCHANGED files, which an added-lines filter cannot see by construction.
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


def git_merge_base(base: str, root: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "merge-base", base, "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


class _MergeBaseWorktree:
    """A detached worktree at `merge_base_sha`, cleaned up on exit.

    Deliberately the ONLY git command this module runs beyond read-only
    `merge-base`/`diff` -- `git worktree add --detach <tmpdir> <sha>` never
    touches the caller's checked-out tree, unlike `checkout`/`reset`/`clean`.
    """

    def __init__(self, sha: str, root: str):
        self.sha = sha
        self.root = root
        self.dir: str | None = None

    def __enter__(self) -> str | None:
        self.dir = tempfile.mkdtemp(prefix="citation-decay-base-")
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", self.dir, self.sha],
                cwd=self.root, capture_output=True, text=True, check=True,
            )
        except Exception:
            return None
        return self.dir

    def __exit__(self, *exc):
        if self.dir is not None:
            subprocess.run(
                ["git", "worktree", "remove", "--force", self.dir],
                cwd=self.root, capture_output=True, text=True, check=False,
            )
        return False


def collect_citations(root: str) -> dict[tuple, Finding]:
    """Every citation under DEFAULT_GLOBS at `root`, keyed by CONTENT.

    Keyed by `(citing relpath, cited path as written, cited start, cited end,
    occurrence index)` -- never by line number, since the citing line shifts
    on a prepend exactly like the cited one does. The occurrence index
    disambiguates two identical citations in one file (same cited span,
    same order of appearance), so repeated citations still pair 1:1 between
    two revisions instead of colliding into one key.
    """
    paths: list[str] = []
    for pat in DEFAULT_GLOBS:
        paths.extend(glob.glob(os.path.join(root, pat), recursive=True))
    out: dict[tuple, Finding] = {}
    for path in sorted(set(paths)):
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        occ: dict[tuple, int] = {}
        for f in scan_file(path, root):
            base_key = (rel, f.cited, f.start, f.end)
            occ[base_key] = occ.get(base_key, -1) + 1
            out[base_key + (occ[base_key],)] = f
    return out


def since_regression(base: str, root: str):
    """Compare citation verdicts at `merge-base(base, HEAD)` vs HEAD.

    Returns (new_decays, preexisting_decays, fixed, None) on success, or
    (None, None, None, error_message) when the comparison cannot be made --
    the caller then FAILS CLOSED rather than silently checking everything or
    silently passing.

    Why a regression comparison instead of a changed-lines filter: the
    citations that decay from an EPIPHANIES.md prepend sit on UNCHANGED lines
    in UNCHANGED files (only the file's LATER content moved under them) --
    an --added-lines-only filter structurally cannot see them, which is
    exactly the gap this replaces. Scoping to changed FILES instead is
    equally wrong the other way: EPIPHANIES.md alone carries pre-existing
    decays and nearly every PR touches it, so that would fail almost every
    PR for backlog it did not introduce (the `closed_class_guess` 150/150
    defect, root CLAUDE.md: a guard that fires on everything carries exactly
    as much information as one that never fires).
    """
    merge_base = git_merge_base(base, root)
    if merge_base is None:
        return None, None, None, (
            f"cannot resolve merge-base('{base}', HEAD) -- refusing to check "
            "every citation instead (in CI this usually means a shallow "
            "checkout; the workflow needs `fetch-depth: 0`)."
        )
    with _MergeBaseWorktree(merge_base, root) as base_dir:
        if base_dir is None:
            return None, None, None, (
                f"cannot create a worktree at merge-base {merge_base} -- "
                "refusing to check every citation instead."
            )
        base_map = collect_citations(base_dir)
    head_map = collect_citations(root)

    new_decays, preexisting, fixed = [], [], []
    for key, f in head_map.items():
        base_f = base_map.get(key)
        base_verdict = base_f.verdict if base_f is not None else None
        if f.verdict == DECAYED:
            (preexisting if base_verdict == DECAYED else new_decays).append(f)
        elif base_verdict == DECAYED:
            fixed.append(f)
    return new_decays, preexisting, fixed, None


def run_since(base: str, root: str) -> int:
    new_decays, preexisting, fixed, err = since_regression(base, root)
    if err is not None:
        print(f"citation-decay: ERROR: {err}", file=sys.stderr)
        return 1
    print(
        f"citation-decay --since {base}: "
        f"{len(new_decays)} new decay(s), {len(preexisting)} pre-existing "
        f"(backlog, not failing), {len(fixed)} fixed since base"
    )
    if new_decays:
        print("\nNEW DECAY (introduced or exposed since base):")
        for f in new_decays[:MAX_EXAMPLES]:
            print("  " + str(f))
        if len(new_decays) > MAX_EXAMPLES:
            print(f"  ... and {len(new_decays) - MAX_EXAMPLES} more")
    if preexisting:
        print(f"\npre-existing backlog (first {min(MAX_EXAMPLES, len(preexisting))}, not failing):")
        for f in preexisting[:MAX_EXAMPLES]:
            print("  " + str(f))
    if fixed:
        print(f"\nfixed since base ({len(fixed)}):")
        for f in fixed[:MAX_EXAMPLES]:
            print("  " + str(f))
    if new_decays:
        print(FIX_MESSAGE)
        return 1
    print("\nno new citation decay since base")
    return 0


def run(paths: list[str], root: str) -> int:
    findings: list[Finding] = []
    for p in sorted(set(paths)):
        if os.path.isfile(p):
            findings.extend(scan_file(p, root))
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

    regression_failures = self_test_since_regression()
    return 1 if (failures or regression_failures) else 0


def _git(args: list[str], cwd: str) -> None:
    subprocess.run(
        ["git"] + args, cwd=cwd, capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


def self_test_since_regression() -> int:
    """Prove `--since` fires on a NEW decay and stays silent on backlog.

    This is the falsifier for the defect that motivated `--since` in the
    first place: `--added-lines-only` filters by which lines a PR ADDS, but
    a citation decays when its TARGET file changes on lines the PR never
    touches (`EPIPHANIES.md` prepend) -- so the old mode reported success on
    precisely the change it exists to catch. Both halves below are real git
    history across two commits, not a synthetic verdict comparison.

    CAN-FIRE: file A cites `B.md:8` correctly at base. Head PREPENDS lines to
    B.md (A is untouched). A NEW decay must be reported and exit 1.

    MUST-STAY-SILENT: in the same repo, a citation already DECAYED at base
    and unchanged at head must NOT be reported as new, and must not by
    itself cause a nonzero exit.
    """
    d = tempfile.mkdtemp(prefix="citation-decay-selftest-since-")
    sub = os.path.join(d, ".claude", "board")
    os.makedirs(sub)
    b_path = os.path.join(sub, "B.md")
    a_path = os.path.join(sub, "A.md")

    def write_b(lines: list[str]) -> None:
        with open(b_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    # base B.md: the CAN-FIRE anchor's real home sits exactly at its cited
    # line 8 (OK at base). The MUST-STAY-SILENT anchor is cited at line 3
    # but its real home is line 20 -- already outside the +-WINDOW_LINES(3)
    # window at base, i.e. ALREADY DECAYED at base -- pre-existing backlog.
    base_lines = [f"line {i} filler" for i in range(1, 22)]
    base_lines[7] = "## E-CANFIRE-ANCHOR-1 heading"     # line 8 (0-indexed 7)
    base_lines[2] = "line 3 filler (backlog cite target)"  # line 3
    base_lines[19] = "## E-BACKLOG-ANCHOR-1 heading"    # line 20
    write_b(base_lines)
    with open(a_path, "w") as fh:
        fh.write(
            "The ruling E-CANFIRE-ANCHOR-1 is recorded at .claude/board/B.md:8.\n"
            + "prose padding, no anchor here at all whatsoever indeed. " * 8 + "\n"
            "The ruling E-BACKLOG-ANCHOR-1 is recorded at .claude/board/B.md:3.\n"
            + "more prose padding, still nothing citable in it at all. " * 8 + "\n"
        )
    _git(["init", "-q"], d)
    _git(["add", "-A"], d)
    _git(["commit", "-q", "-m", "base"], d)

    # head: PREPEND 5 lines to B.md (> WINDOW_LINES so the shift cannot hide
    # inside the anchor-search window). A.md is untouched -- the citing
    # lines themselves never move. This pushes the CAN-FIRE anchor's real
    # home from line 8 to line 13, outside cited-line-8's +-3 window: a real
    # NEW decay on an unchanged citing line. The backlog anchor shifts from
    # 20 to 25, still outside cited-line-3's +-3 window either side of the
    # prepend, so it stays exactly as decayed as it was at base.
    head_lines = ["PREPENDED filler"] * 5 + base_lines
    write_b(head_lines)
    _git(["add", "-A"], d)
    _git(["commit", "-q", "-m", "prepend to B.md, A.md untouched"], d)

    new_decays, preexisting, fixed, err = since_regression("HEAD~1", d)
    print("\n--- self-test --since findings ---")
    if err is not None:
        print(f"  ERROR: {err}")
        print("--- self-test --since FAILED ---")
        return 1
    for label, group in (("new", new_decays), ("preexisting", preexisting), ("fixed", fixed)):
        for f in group:
            print(f"  [{label}] " + str(f))

    fail_msgs = []
    canfire_new = any(f.cited == ".claude/board/B.md" and f.start == 8 for f in new_decays)
    if not canfire_new:
        fail_msgs.append("CAN-FIRE: .claude/board/B.md:8 must be reported as a NEW decay")
    backlog_new = any(f.cited == ".claude/board/B.md" and f.start == 3 for f in new_decays)
    if backlog_new:
        fail_msgs.append("MUST-STAY-SILENT: .claude/board/B.md:3 (pre-existing decay) must NOT be reported as new")
    backlog_preexisting = any(f.cited == ".claude/board/B.md" and f.start == 3 for f in preexisting)
    if not backlog_preexisting:
        fail_msgs.append("MUST-STAY-SILENT: .claude/board/B.md:3 must be counted as pre-existing backlog")

    # The verdict the GATE would return on this corpus -- NOT the self-test's
    # own pass/fail. The first version printed `0 if ok else 1` under a label
    # reading "expect 1", so a PASSING run printed `0` next to the word
    # "expect 1" and asserted nothing whatsoever about the real exit path.
    gate_rc = 1 if new_decays else 0
    if gate_rc != 1:
        fail_msgs.append("CAN-FIRE: the gate must exit 1 while a new decay is present")

    for m in fail_msgs:
        print(f"  [FAIL] {m}")
    ok = not fail_msgs
    print(f"  gate exit code on this corpus: {gate_rc} (expect 1 -- new decay present)")
    print("--- self-test --since " + ("PASSED ---" if ok else "FAILED ---"))
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    root = os.getcwd()
    if "--self-test" in argv:
        return self_test()
    if "--since" in argv:
        i = argv.index("--since")
        base = argv[i + 1]
        return run_since(base, root)
    pats = [a for a in argv if not a.startswith("-")] or DEFAULT_GLOBS
    paths: list[str] = []
    for p in pats:
        hits = glob.glob(p, recursive=True)
        paths.extend(hits if hits else ([p] if os.path.isfile(p) else []))
    if not paths:
        print("citation-decay: no input files matched", file=sys.stderr)
        return 0
    return run(paths, root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
