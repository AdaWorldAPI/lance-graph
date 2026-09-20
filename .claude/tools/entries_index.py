#!/usr/bin/env python3
"""Generate `.claude/board/entries/README.md` — the transient-tier index.

WHY GENERATED
-------------
The index was hand-maintained and went stale exactly the way a hand-maintained
index does. Measured 2026-09-20, before this generator existed:

  * its header claimed "135 entries, 2026-08-06 .. 2026-08-26" against **144**
    files / 142 rows / range 2026-08-06 .. **2026-08-31**;
  * its OWN falsifier #2 was RED — two entry files had landed with no index row
    (`2026-08-27-e-the-fused-payload-…-1`, `2026-08-31-e-q8-…-1`), i.e. the
    exact stranding the README says the check exists to catch;
  * one row sat out of date order.

Nothing ran those falsifiers: they were shell snippets in a README, and the
directory is in no structural gate. So the stranding was invisible by
construction, which is the same shape as every other drift this workspace
gates mechanically (`supersession_index.py`, `citation_decay.py`,
`append_only_gate.py`).

WHAT IS DERIVED AND WHAT IS CARRIED FORWARD (this asymmetry is measured)
-----------------------------------------------------------------------
Entry files do NOT share a heading shape. All four of these are live on disk:

    ### E-ID-1
    ## 2026-08-19 — E-ID-1
    # E-ID-1            (with a separate `**Date:** 2026-08-27` line)
    ## 2026-08-31 — E-ID-1 — <the finding, inline after a second em-dash>

So:

  * `date` comes from the FILENAME, which IS uniform (`YYYY-MM-DD-<slug>.md`).
    Never from the heading — three of the four shapes do not carry it.
  * `file` is the filename.
  * `id` is carried forward when the file is already indexed (so the existing
    mixed case is preserved rather than mass-rewritten), else recovered from
    the heading, else derived from the slug.
  * `finding` is **carried forward verbatim**. It is NOT derivable: over 100
    rows carry a hand-written one-line summary that no heading shape contains.
    A generator that "derived" this column would silently delete curation.
    For a NEW entry it is taken from the heading's second em-dash segment when
    that shape is used, else left empty — exactly today's behaviour.

Curation therefore stays possible in the `finding` cell; structure (which rows
exist, their dates, their order, the counts) is enforced.

THE TRUNCATION TRAP — WHY THERE IS NO `>` USAGE
-----------------------------------------------
Because the committed index is an INPUT (the carried-forward prose), the house
convention `python3 tool.py > target.md` would have the shell TRUNCATE the file
before this script reads it, destroying every curated summary in one keystroke.
That is the destructive-prepend law in root `CLAUDE.md`
(`.claude/knowledge/never-truncate-a-file-you-still-need-to-read.md`), and it
is a live risk here precisely because the sibling generator IS used that way.

Hence: default prints to stdout for DIFFING only; `--write` is the sole
sanctioned mutation and does read-then-write; and `--write` REFUSES to emit
fewer non-empty `finding` cells than the committed file already has unless
`--allow-finding-loss` is passed. That guard is the semantic form of the
no-shrink gate: for a generated table the meaningful quantity is curated cells,
not lines.

USAGE
    python3 .claude/tools/entries_index.py            # print (for diffing)
    python3 .claude/tools/entries_index.py --write     # regenerate in place
    python3 .claude/tools/entries_index.py --check     # CI: falsifiers + staleness
    python3 .claude/tools/entries_index.py --self-test

Pure stdlib.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ENTRIES_DIR = ".claude/board/entries"
INDEX = os.path.join(ENTRIES_DIR, "README.md")

FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)\.md$")
# The link target inside the `file` cell: `[text](2026-08-06-foo-1.md)`.
LINK_TARGET_RE = re.compile(r"\(([^()]*\.md)\)")
# An id in a heading, with or without a leading date and surrounding markup.
HEADING_RE = re.compile(r"^#{1,4}\s+(.*)$")
ID_IN_HEADING_RE = re.compile(r"\b((?:E|D|I|ISS|PROBE|ADR|EXP)-[A-Z0-9][A-Za-z0-9-]{2,})\b")


def repo_root() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    return out.stdout.strip() or "."


def parse_index(text: str) -> dict[str, dict[str, str]]:
    """Existing rows, keyed by the `file` cell's link target.

    Keyed on the link TARGET rather than the id, because the target is what
    both falsifiers resolve against and the only cell that must match a real
    path. A row whose target is unparseable is dropped from carry-forward --
    it cannot be matched to a file, which is itself the finding.

    Cells are split on `|` and rejoined for the middle column, so a `finding`
    containing a literal pipe survives a round trip instead of shifting every
    column right of it.
    """
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("| 20"):
            continue
        parts = line.split("|")
        if len(parts) < 6:
            continue
        date = parts[1].strip()
        entry_id = parts[2].strip()
        finding = "|".join(parts[3:-2]).strip()
        file_cell = parts[-2].strip()
        m = LINK_TARGET_RE.search(file_cell)
        if not m:
            continue
        rows[m.group(1)] = {"date": date, "id": entry_id, "finding": finding}
    return rows


def recover_from_file(path: str) -> tuple[str, str]:
    """(id, inline_finding) recovered from the entry's own first heading.

    Handles all four shapes measured on disk. The inline finding is the SECOND
    em-dash segment of a `## <date> — <ID> — <finding>` heading; the other
    shapes have none, and the caller leaves the cell empty rather than
    inventing one.
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                m = HEADING_RE.match(line.strip())
                if not m:
                    continue
                head = m.group(1).strip()
                segments = [s.strip() for s in head.split("—")]
                ident = ""
                for seg in segments:
                    hit = ID_IN_HEADING_RE.search(seg)
                    if hit:
                        ident = hit.group(1)
                        break
                inline = ""
                if ident and len(segments) >= 3:
                    tail = segments[2:]
                    inline = " — ".join(t for t in tail if t).strip()
                return ident, inline
    except OSError:
        pass
    return "", ""


def falsifiers(root: str) -> list[str]:
    """The README's own three structural checks, as code rather than prose.

    Same three, same directions, unchanged in meaning:
      1. every index row's file resolves  (catches a row whose file never landed)
      2. every file has an index row      (catches a file that landed with no row)
      3. no duplicate entry id
    1 and 2 are deliberately opposite; a stranding shows up in exactly one.
    """
    d = os.path.join(root, ENTRIES_DIR)
    index_path = os.path.join(root, INDEX)
    try:
        with open(index_path, encoding="utf-8") as fh:
            rows = parse_index(fh.read())
    except OSError:
        return [f"FATAL: cannot read {INDEX}"]

    on_disk = sorted(f for f in os.listdir(d) if FILE_RE.match(f))
    problems: list[str] = []

    for target in sorted(rows):
        if not os.path.isfile(os.path.join(d, target)):
            problems.append(f"DANGLING: {target} (index row with no file)")

    for f in on_disk:
        if f not in rows:
            problems.append(f"UNREFERENCED: {f} (file with no index row)")

    seen: dict[str, int] = {}
    for meta in rows.values():
        key = meta["id"].strip("`").upper()
        seen[key] = seen.get(key, 0) + 1
    for key, n in sorted(seen.items()):
        if n > 1 and key:
            problems.append(f"DUPLICATE ID: {key} ({n} rows)")

    return problems


def render(root: str) -> str:
    d = os.path.join(root, ENTRIES_DIR)
    try:
        with open(os.path.join(root, INDEX), encoding="utf-8") as fh:
            prior = parse_index(fh.read())
    except OSError:
        prior = {}

    files = sorted(f for f in os.listdir(d) if FILE_RE.match(f))
    rows = []
    for f in files:
        m = FILE_RE.match(f)
        assert m  # guarded by the filter above
        date, slug = m.group(1), m.group(2)
        carried = prior.get(f, {})
        ident = carried.get("id", "")
        finding = carried.get("finding", "")
        if not ident or not finding:
            rec_id, rec_finding = recover_from_file(os.path.join(d, f))
            if not ident:
                ident = f"`{rec_id}`" if rec_id else f"`{slug}`"
            if not finding:
                finding = rec_finding
        rows.append((date, ident, finding, f))

    # Newest first; filename as the tie-break so the order is total and stable.
    rows.sort(key=lambda r: (r[0], r[3]), reverse=True)

    dates = [r[0] for r in rows]
    out: list[str] = []
    out.append("# Board entries — one file per finding\n")
    out.append("")
    out.append("> **GENERATED — do not hand-edit the table's structure.**")
    out.append("> `python3 .claude/tools/entries_index.py --write`")
    out.append(">")
    out.append("> Each entry is `YYYY-MM-DD-<entry-id>.md`, carrying the entry **verbatim**.")
    out.append("> This table is the index; the files are the content. A row whose file does")
    out.append("> not resolve is a broken reference — that is the falsifier, and it is why")
    out.append("> the index and the content are separate objects.")
    out.append(">")
    out.append("> The `finding` cell is the ONE hand-curated column: it is carried forward")
    out.append("> verbatim on every regeneration, because the entry files do not share a")
    out.append("> heading shape and it cannot be derived. Edit it freely. `date`, `id`,")
    out.append("> `file`, the ordering and the counts are derived and will be overwritten.")
    out.append(">")
    out.append("> **Never `… > README.md`.** This file is an INPUT to its own generator, so")
    out.append("> a shell redirect truncates it before the script reads it and every curated")
    out.append("> `finding` is lost. Use `--write`, which reads first and refuses to drop")
    out.append("> curated cells.")
    out.append("")
    out.append("**Falsifiers** — now executed by CI (`entries_index.py --check`), not just")
    out.append("described here: (1) every index row's file resolves, (2) every file has an")
    out.append("index row, (3) no duplicate entry id. Checks 1 and 2 are deliberately")
    out.append("opposite directions; the stranding this convention prevents shows up in")
    out.append("exactly one of them, never both.")
    out.append("")
    if rows:
        out.append(f"{len(rows)} entries, {min(dates)} .. {max(dates)}.")
    else:
        out.append("0 entries.")
    out.append("")
    out.append("| date | entry id | finding | file |")
    out.append("|---|---|---|---|")
    for date, ident, finding, f in rows:
        # Link TEXT keeps the trailing `.md`: that is the existing convention in
        # 138 of 142 committed rows, and normalising it away would churn every
        # row of the diff to change nothing a reader sees differently.
        out.append(f"| {date} | {ident} | {finding} | [{f}]({f}) |")
    out.append("")
    return "\n".join(out)


def nonempty_findings(text: str) -> int:
    return sum(1 for meta in parse_index(text).values() if meta["finding"])


def committed_findings_at_head(root: str) -> int:
    """Curated `finding` cells in the index as COMMITTED at git HEAD.

    The write guard's reference point. It must not be the working file: the
    generated table carries its prose forward FROM that file, so a truncated
    working copy drags the "after" count down with the "before" count and the
    guard silently passes. HEAD is outside the shell's reach.

    A path absent at HEAD (the first-ever add) returns 0, which correctly makes
    the guard inert rather than blocking the initial commit.
    """
    out = subprocess.run(
        ["git", "-C", root, "show", f"HEAD:{INDEX}"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return 0
    return nonempty_findings(out.stdout)


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()

    root = repo_root()
    index_path = os.path.join(root, INDEX)
    generated = render(root)

    if "--check" in argv:
        problems = falsifiers(root)
        for p in problems:
            print(f"  {p}")
        try:
            with open(index_path, encoding="utf-8") as fh:
                committed = fh.read()
        except OSError:
            print(f"::error::{INDEX} is missing")
            return 1
        stale = committed != generated
        if stale:
            print(f"::error::{INDEX} is stale.")
            print("It is GENERATED from the entry files in .claude/board/entries/.")
            print("Regenerate and commit:")
            print("  python3 .claude/tools/entries_index.py --write")
            print("NEVER `… > README.md` — the file is its own input and a redirect")
            print("truncates it before the generator reads it.")
        if problems:
            print("::error::the entries tier failed a structural falsifier (see above).")
            print("A file with no row is invisible to every index consumer; a row with")
            print("no file is a broken reference. `--write` fixes both by regenerating.")
        if stale or problems:
            return 1
        print(f"entries index is current and structurally sound ({len(parse_index(generated))} rows)")
        return 0

    if "--write" in argv:
        # The reference is git HEAD, NOT the working file. Comparing against the
        # working file makes this guard VACUOUS for the one trap it exists to
        # catch: the generated output is DERIVED from that file, so truncating
        # it lowers both sides equally and an emptied index writes cleanly
        # (measured -- the first version of this guard returned 0 on a blanked
        # index, and a `>` redirect would have yielded before=after=0). HEAD is
        # the last state a shell redirect cannot have destroyed.
        before = committed_findings_at_head(root)
        after = nonempty_findings(generated)
        if after < before and "--allow-finding-loss" not in argv:
            print(
                f"::error::refusing to write: curated `finding` cells would drop "
                f"{before} -> {after}.",
                file=sys.stderr,
            )
            print(
                "The `finding` column is hand-written and carried forward; losing cells "
                "means the index was truncated before being read (a `>` redirect) or an "
                "entry file was renamed out from under its row. Investigate rather than "
                "overwrite. `--allow-finding-loss` forces it.",
                file=sys.stderr,
            )
            return 1
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.write(generated)
        print(f"wrote {INDEX}: {len(parse_index(generated))} rows, {after} curated findings")
        return 0

    sys.stdout.write(generated)
    return 0


def self_test() -> int:
    """Unit checks on the two things that can silently lose data."""
    fails = []

    # 1. a `finding` containing a literal pipe survives a round trip.
    piped = (
        "| date | entry id | finding | file |\n"
        "|---|---|---|---|\n"
        "| 2026-08-06 | `E-X-1` | a \\| b and more | [2026-08-06-e-x-1](2026-08-06-e-x-1.md) |\n"
    )
    got = parse_index(piped)
    if got.get("2026-08-06-e-x-1.md", {}).get("finding") != "a \\| b and more":
        fails.append(f"pipe-in-finding round trip: got {got}")

    # 2. the carried-forward count is what the write guard compares.
    if nonempty_findings(piped) != 1:
        fails.append("nonempty_findings miscounted a single curated cell")
    empty = (
        "| date | entry id | finding | file |\n"
        "|---|---|---|---|\n"
        "| 2026-08-06 | `E-X-1` |  | [2026-08-06-e-x-1](2026-08-06-e-x-1.md) |\n"
    )
    if nonempty_findings(empty) != 0:
        fails.append("nonempty_findings counted an empty cell as curated")

    # 3. all four measured heading shapes yield an id; only the 3-segment one
    #    yields an inline finding.
    import tempfile

    shapes = [
        ("### E-THE-A-1\n", "E-THE-A-1", ""),
        ("## 2026-08-19 — E-THE-B-1\n", "E-THE-B-1", ""),
        ("# E-THE-C-1\n\n**Date:** 2026-08-27\n", "E-THE-C-1", ""),
        ("## 2026-08-31 — E-THE-D-1 — the six does no work\n", "E-THE-D-1",
         "the six does no work"),
    ]
    with tempfile.TemporaryDirectory() as td:
        for i, (body, want_id, want_find) in enumerate(shapes):
            p = os.path.join(td, f"2026-08-0{i+1}-e-the-x-1.md")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(body)
            gid, gfind = recover_from_file(p)
            if gid != want_id:
                fails.append(f"shape {i}: id {gid!r} != {want_id!r}")
            if gfind != want_find:
                fails.append(f"shape {i}: inline finding {gfind!r} != {want_find!r}")

    for f in fails:
        print(f"  FAIL  {f}")
    print("self-test: " + ("ALL PASSED" if not fails else f"{len(fails)} FAILURE(S)"))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
