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
EPI = ".claude/board/EPIPHANIES.md"
ENTRY_REF = re.compile(r"entries/(\d{4}-\d{2}-\d{2}-[A-Za-z0-9._-]+\.md)")
HEAD2 = re.compile(r"^##\s+(?!#)(.*)$")
EID = re.compile(r"\b(E-[A-Z0-9][A-Z0-9-]{3,})\b")


def run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def baseline_sha(root: str) -> str:
    """The consumed-input revision. Never this commit's own hash."""
    p = pathlib.Path(root, MARKER)
    if not p.is_file():
        raise SystemExit(
            f"epiphany-provenance: {MARKER} is missing. The gate cannot define a "
            "delta without a baseline; add the marker rather than disabling this."
        )
    for line in p.read_text(errors="ignore").splitlines():
        if line.startswith("PROCESSED_THROUGH_SHA="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"epiphany-provenance: no PROCESSED_THROUGH_SHA= line in {MARKER}")


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


def check(root: str) -> tuple[list[tuple[str, str]], int]:
    """-> (violations, number of added headings examined)."""
    sha = baseline_sha(root)
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
    return bad, len(heads)


def main(argv: list[str]) -> int:
    root = run(["git", "rev-parse", "--show-toplevel"], ".").stdout.strip() or "."
    if "--self-test" in argv:
        return self_test()
    bad, n = check(root)
    print(f"epiphany-provenance: baseline {baseline_sha(root)[:12]}, "
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
    for a in (["init", "-q"], ["add", "-A"],
              ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"]):
        run(["git", *a], d)
    sha = run(["git", "rev-parse", "HEAD"], d).stdout.strip()
    pathlib.Path(d, MARKER).write_text(f"PROCESSED_THROUGH_SHA={sha}\n")

    def commit_and_check(extra: str, label: str):
        epi.write_text(epi.read_text() + extra)
        run(["git", "add", "-A"], d)
        run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", label], d)
        return check(d)

    ok = True

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

    # (d) an unreachable baseline must REFUSE, never report a clean delta
    pathlib.Path(d, MARKER).write_text(
        "PROCESSED_THROUGH_SHA=" + "0" * 40 + "\n")
    try:
        check(d)
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
