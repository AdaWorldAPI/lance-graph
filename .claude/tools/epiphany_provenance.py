#!/usr/bin/env python3
"""Every EPIPHANY added after the baseline must cite its originating entry.

    python3 .claude/tools/epiphany_provenance.py            # gate
    python3 .claude/tools/epiphany_provenance.py --self-test # falsifier

WHAT IT PROVES, AND ONLY THIS
-----------------------------
PROVENANCE. A level-2 `EPIPHANIES.md` heading added since
`PROCESSED_THROUGH_SHA` must carry a reference to a
`.claude/board/entries/YYYY-MM-DD-<slug>.md` that EXISTS. That enforces the
mechanical invariant `work -> entries/`, never `work -> EPIPHANIES.md`.

It does NOT decide whether the entry is a Eureka, whether NEW / LOAD-BEARING /
DURABLE is satisfied, or whether promoting it was wise. Those are the human
closeout admission rule and are deliberately not mechanised: a regex that
tried to judge Eureka-ness would be a guard that fires on everything, which
carries exactly as much information as one that never fires.

WHY THE WATERMARK IS A SHA AND NOT A DATE
-----------------------------------------
Imported entries, backdated headings, rebases and concurrent work all make a
calendar watermark lie — `supersession_index.py` already refuses git mtime as
a signal for the same reason ("2026-07-24 is a bulk import ... git dates the
import, not the work"). The SHA names a revision, so the delta is exact.

WHY THE MARKER IS READ AT THE MERGE-BASE, NOT FROM THE CHECKOUT
---------------------------------------------------------------
A branch that ADVANCES the marker would otherwise erase its own delta.
MEASURED (two ordinary commits, reproduced before this was fixed): C1 adds an
uncited heading — the gate fires; C2 advances `PROCESSED_THROUGH_SHA` to C1 —
`git diff C1..HEAD` no longer contains C1's own change, so the gate reports
`0 added, 0 violations` and the uncited heading ships. Reading the marker as
the branch INHERITED it closes that: the delta is measured from the baseline
main had, which no commit on the branch can move. This is the mechanism
`append_only_gate.py` already uses and documents for the same class of
problem ("a straight `git show <base>:<path>` would compare against work the
branch never saw").

The marker not existing at the merge-base is the introducing PR's own case and
falls back to the checkout, printed so the weaker reference is never silent.

FAIL CLOSED ON A SHALLOW CLONE
------------------------------
This repo is routinely a shallow clone (`.git/shallow`, 6 grafts). If the
baseline revision is above the graft boundary, `git diff` cannot see it. The
gate then REFUSES with a diagnostic naming the revision. Treating unreachable
history as an empty delta is the failure this rule exists to prevent: it would
report a clean pass precisely when it can see nothing.
"""

import os
import pathlib
import re
import subprocess
import sys

MARKER = ".claude/board/PROCESSED_THROUGH"
DEFAULT_BASE = "origin/main"
EPI = ".claude/board/EPIPHANIES.md"
ENTRY_REF = re.compile(r"entries/(\d{4}-\d{2}-\d{2}-[A-Za-z0-9._-]+\.md)")
HEAD2 = re.compile(r"^##\s+(?!#)(.*)$")
EID = re.compile(r"\b(E-[A-Z0-9][A-Z0-9-]{3,})\b")


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def parse_marker(text: str, where: str) -> str:
    for line in text.splitlines():
        if line.startswith("PROCESSED_THROUGH_SHA="):
            sha = line.split("=", 1)[1].strip()
            if sha:
                return sha
            raise SystemExit(
                f"epiphany-provenance: empty PROCESSED_THROUGH_SHA= in {where}"
            )
    raise SystemExit(f"epiphany-provenance: no PROCESSED_THROUGH_SHA= line in {where}")


def baseline_sha(root: str, base_ref: str = DEFAULT_BASE) -> tuple[str, str]:
    """-> (consumed-input revision, where it was read).

    Read as the branch INHERITED it, not from the checkout — see the module
    docs: a commit that advances the marker would otherwise erase its own
    delta. Falls back to the checkout only when the marker does not exist at
    the merge-base, and the caller prints which reference was used.
    """
    mb = run(["git", "merge-base", "HEAD", base_ref], root)
    if mb.returncode == 0 and mb.stdout.strip():
        base = mb.stdout.strip()
        show = run(["git", "show", f"{base}:{MARKER}"], root)
        if show.returncode == 0:
            return parse_marker(show.stdout, f"{MARKER} at {base[:12]}"), \
                f"inherited at merge-base {base[:12]} with {base_ref}"

    p = pathlib.Path(root, MARKER)
    if not p.is_file():
        raise SystemExit(
            f"epiphany-provenance: {MARKER} is missing. The gate cannot define a "
            "delta without a baseline; add the marker rather than disabling this."
        )
    return parse_marker(p.read_text(errors="ignore"), MARKER), \
        f"the CHECKOUT ({MARKER} absent at the merge-base with {base_ref})"


def added_headings(root: str, sha: str) -> list[str]:
    """Level-2 headings ADDED to EPIPHANIES.md since `sha`.

    Fails closed when `sha` is not reachable — see the module docs.
    """
    if run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], root).returncode != 0:
        shallow = pathlib.Path(root, ".git", "shallow")
        hint = (
            " This clone is SHALLOW (.git/shallow exists), so the baseline is most "
            "likely above the graft boundary. Deepen it "
            "(`git fetch --shallow-exclude= --unshallow`) and re-run."
            if shallow.exists() else ""
        )
        raise SystemExit(
            f"epiphany-provenance: baseline revision {sha} is NOT REACHABLE in this "
            f"repository, so the delta cannot be computed.{hint} REFUSING — "
            "unreachable history is not an empty delta."
        )
    d = run(["git", "diff", "--unified=0", f"{sha}..HEAD", "--", EPI], root)
    if d.returncode != 0:
        raise SystemExit(f"epiphany-provenance: git diff failed: {d.stderr.strip()}")
    out = []
    for line in d.stdout.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        m = HEAD2.match(line[1:])
        if m:
            out.append(m.group(1))
    return out


def body_of(text: str, heading: str) -> str:
    """The entry under `heading`, up to the next heading of level <= 2."""
    lines = text.split("\n")
    try:
        start = next(i for i, l in enumerate(lines)
                     if HEAD2.match(l) and HEAD2.match(l).group(1) == heading)
    except StopIteration:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = re.match(r"^(#{1,2})\s+(?!#)", lines[j])
        if m:
            end = j
            break
    return "\n".join(lines[start:end])


def check(root: str, base_ref: str = DEFAULT_BASE) -> tuple[list[tuple[str, str]], int, str, str]:
    """-> (violations, added headings examined, baseline sha, its provenance)."""
    sha, whence = baseline_sha(root, base_ref)
    heads = added_headings(root, sha)
    text = pathlib.Path(root, EPI).read_text(errors="ignore")
    bad = []
    for h in heads:
        body = body_of(text, h)
        refs = ENTRY_REF.findall(body)
        live = [r for r in refs if pathlib.Path(root, ".claude/board/entries", r).is_file()]
        if not refs:
            bad.append((h, "no entries/ reference"))
        elif not live:
            bad.append((h, f"references a file that does not exist: {', '.join(refs[:3])}"))
    return bad, len(heads), sha, whence


def main(argv: list[str]) -> int:
    root = run(["git", "rev-parse", "--show-toplevel"], ".").stdout.strip() or "."
    if "--self-test" in argv:
        return self_test()
    bad, n, sha, whence = check(root)
    print(f"epiphany-provenance: baseline {sha[:12]} ({whence}), "
          f"{n} level-2 heading(s) added since it, {len(bad)} without provenance")
    if not bad:
        return 0
    print()
    print("::error::An EPIPHANY was added without citing its originating entry.")
    print("Ordinary work lands in .claude/board/entries/ and is reconciled at")
    print("closeout; only a surviving Eureka is promoted here, and a promotion")
    print("must name the entry it came from so the route stays recoverable.")
    for h, why in bad:
        eid = EID.search(h)
        print(f"  - {eid.group(1) if eid else h[:60]}: {why}")
    return 1


def self_test() -> int:
    """Prove the gate FIRES on a missing reference and STAYS SILENT on a real
    one — in a throwaway repo, so neither half can pass vacuously."""
    import tempfile

    d = tempfile.mkdtemp(prefix="epiphany-provenance-selftest-")
    ent = pathlib.Path(d, ".claude/board/entries")
    ent.mkdir(parents=True)
    tools = pathlib.Path(d, ".claude/tools")
    tools.mkdir(parents=True)
    epi = pathlib.Path(d, EPI)
    epi.write_text("# Epiphanies\n\n## 2026-01-01 E-BASE-1 — pre-baseline\n\nbody\n")
    (ent / "2026-09-20-e-real-1.md").write_text("# entry\n")

    def commit(label: str):
        run(["git", "add", "-A"], d)
        run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-qm", label], d)
        return run(["git", "rev-parse", "HEAD"], d).stdout.strip()

    run(["git", "init", "-q"], d)
    sha = commit("base")
    # The marker is COMMITTED, and a `main` branch is left pointing at it, so
    # the merge-base reference the gate actually reads is exercised here rather
    # than only the checkout fallback.
    pathlib.Path(d, MARKER).write_text(f"PROCESSED_THROUGH_SHA={sha}\n")
    commit("marker")
    run(["git", "branch", "-f", "main"], d)

    def commit_and_check(extra: str, label: str):
        epi.write_text(epi.read_text() + extra)
        commit(label)
        bad, n, _sha, _whence = check(d, "main")
        return bad, n

    ok = True
    # startswith, not `in`: the FALLBACK string also names the merge-base (it
    # says the marker was absent there), so a substring test passed under the
    # very disable it exists to catch -- vacuous, found by running that disable.
    _bad, _n, _sha, whence = check(d, "main")
    if not whence.startswith("inherited at merge-base"):
        print(f"  FAILED: the merge-base reference was not used ({whence})")
        ok = False

    # (a) an addition WITH a resolvable reference -> silent
    bad, n = commit_and_check(
        "\n## 2026-09-20 E-GOOD-1 — cites its entry\n\n"
        "From `.claude/board/entries/2026-09-20-e-real-1.md`.\n", "good")
    print(f"  with a live entries/ reference : {n} added, {len(bad)} violation(s)")
    if bad or n != 1:
        print("  FAILED: the gate must stay SILENT on a well-formed promotion")
        ok = False

    # (b) an addition with NO reference -> fires
    bad, n = commit_and_check("\n## 2026-09-20 E-BAD-1 — cites nothing\n\nbody\n", "bad")
    if not any(w == "no entries/ reference" for _h, w in bad):
        print(f"  FAILED: no violation raised for a reference-less addition ({bad})")
        ok = False
    else:
        print(f"  with no reference             : {len(bad)} violation(s) (fires)")

    # (c) an addition referencing a MISSING file -> fires (a name is not a file)
    bad, n = commit_and_check(
        "\n## 2026-09-20 E-BAD-2 — cites a ghost\n\n"
        "See `.claude/board/entries/2026-09-20-e-does-not-exist.md`.\n", "ghost")
    if not any("does not exist" in w for _h, w in bad):
        print(f"  FAILED: a dangling reference was accepted ({bad})")
        ok = False
    else:
        print(f"  with a dangling reference     : fires")

    # (d) ADVANCING the marker past an uncited heading must not erase the
    # delta. Without the merge-base read this is the measured bypass: two
    # ordinary commits and the gate goes silent (CodeRabbit on #1254).
    pathlib.Path(d, MARKER).write_text(
        f"PROCESSED_THROUGH_SHA={run(['git', 'rev-parse', 'HEAD'], d).stdout.strip()}\n")
    commit("advance the marker past the uncited headings")
    bad, n = check(d, "main")[:2]
    if not bad:
        print("  FAILED: advancing the marker erased the delta (the bypass)")
        ok = False
    else:
        print(f"  with the marker advanced      : {len(bad)} violation(s) (fires)")

    # (e) an unreachable baseline must REFUSE, never report a clean delta.
    # Written to BOTH the checkout and the merge-base reference, so neither
    # path can quietly supply a good baseline and make this arm vacuous.
    pathlib.Path(d, MARKER).write_text(
        "PROCESSED_THROUGH_SHA=" + "0" * 40 + "\n")
    commit("unreachable baseline")
    run(["git", "branch", "-f", "main", "HEAD"], d)
    try:
        check(d, "main")
        print("  FAILED: an unreachable baseline did not refuse")
        ok = False
    except SystemExit as exc:
        if "NOT REACHABLE" not in str(exc):
            print(f"  FAILED: wrong refusal: {exc}")
            ok = False
        else:
            print("  with an unreachable baseline  : refuses (fail-closed)")

    print("epiphany-provenance --self-test " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
