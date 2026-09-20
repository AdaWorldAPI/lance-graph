#!/usr/bin/env python3
"""Mechanically project the EPIPHANIES archive into a compact canonical table.

    compact_epiphanies.py --measure           # archive census only, writes nothing
    compact_epiphanies.py --write             # render EPIPHANIES.md from the rows file
    compact_epiphanies.py --check             # the projection is current
    compact_epiphanies.py --self-test         # falsifiers

WHAT THIS IS, AND ONLY THIS
---------------------------
TWO INPUTS, and the split between them is the whole design:

  1. the ARCHIVE -- frozen, lossless, byte-identical to the pre-compaction file,
     never appended to and never rewritten. `--measure` reads only this and
     prints the mechanical census (structured status tokens, terminal drops,
     duplicate E-ids) with an accounting assertion.
  2. the ROWS file -- `.claude/board/epiphanies-rows.json`, the committed
     CONSOLIDATION data: one record per archived heading carrying its topic,
     priority, one-line finding and closed-out verdict. It was produced ONCE, by
     a Sonnet worker fleet over the archive, and is data from here on.

`--write` renders the kanban projection from the ROWS file; `--check` proves the
projection is current. Rendering is pure: same rows in, byte-identical table
out, no model in the loop. The one provenance rule is enforced in code -- every
row's line must resolve to a real level-2 heading in the archive, so a row can
never describe an entry that does not exist.

The MECHANICAL status filter was measured and REJECTED as the compaction
mechanism: the corpus grades epistemically (FINDING / RULING / CORRECTION), so
dropping explicit terminal tokens took 891 rows to 877 -- no compaction at all.
It is retained as the census (`--measure`), which is what it is honestly good
for. Compaction comes from the consolidation instead: a 3.5 MB prose corpus
becomes a ~150 KB one-line-per-entry table.

WHY THE STATUS PARSER IS STRUCTURED-FIELD-ONLY
----------------------------------------------
Searching prose for a status word drops entries that merely CITE one. The
measured shape in this corpus is a sibling's caveat quoted as `(⊘ in E-FOO-1)`,
which a substring test reads as this entry being retired -- the exact error that
over-counted SUPERSEDED 9 -> 5 in the findings baseline. So the token must come
from the entry's own `**Status:** ` / `**Verdict:** ` / `**State:** ` field, and
only its LEADING word.

UNKNOWN AND ABSENT STATUS KEEP
------------------------------
An epistemic grade (FINDING / RULING / MEASURED / CORRECTION / OPEN / ...)
answers *how well established*, not *is it done*, so it is not terminal. SHIPPED
is deliberately NOT terminal either: a shipped implementation can still encode a
durable architectural insight. Absent or unrecognised status keeps, because a
compactor that guessed would be adjudicating.
"""

import hashlib
import json
import pathlib
import re
import sys
import tempfile

ARCHIVE = ".claude/board/EPIPHANIES-ARCHIVE-2026-09-20.md"
COMPACT = ".claude/board/EPIPHANIES.md"
ROWS = ".claude/board/epiphanies-rows.json"

# The consolidation's topic vocabulary. Closed: a row outside it is a defect,
# not a new topic -- a table whose topic column is open-ended cannot be scanned.
TOPICS = {
    "nars-thinking", "classid-facet", "evidence-method", "ogar-transcode",
    "substrate-soa", "vsa-codec", "planner-r2il-loco", "mask-risc-quack",
    "deps-cargo", "board-process", "ci-tooling", "lance-storage",
    "ocr-tesseract", "consumer-app", "java-lgj", "other",
}
PRIORITIES = {"P0", "P1", "P2", "P3"}

H2 = re.compile(r"^##\s+(?!#)(.*)$")
# Lowercase is admitted after the first char and a trailing `-` is stripped:
# `E-0xFFF-IS-ONE-ALIGNED-ADDRESS` was unmatchable (the `x`), so the heading's
# CITED id got taken as its own; `E-X265-MORTON-SHIFT-1a`/`-1b` both truncated
# to `E-X265-MORTON-SHIFT-` and one was discarded as a false duplicate. Measured
# on the archive before the fix; both keyed correctly after it.
EID = re.compile(r"\b(E-[A-Z0-9][A-Za-z0-9-]{3,}?)(?=[^A-Za-z0-9-]|$)")
DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
BARE_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")

# The entry's OWN structured field, leading token only. Never arbitrary prose.
STATUS = re.compile(
    # NOTE the `\s*` after the trailing `\*{0,2}`: the corpus writes
    # `**Status:** RESOLVED`, i.e. the bold CLOSES AFTER the colon, so a space
    # sits between `**` and the token. Without it this matches nothing at all in
    # this corpus -- measured, the self-test caught it.
    r"^\s*[>*\-\s]*\*{0,2}(?:Status|Verdict|State)\*{0,2}\s*[:—-]\s*\*{0,2}\s*([A-Za-z⊘-]+)"
)

# Explicitly terminal. Everything else -- including every epistemic grade and
# SHIPPED -- survives. Widening this set is a measured decision, not a tidy-up.
DROP = {
    "CLOSED", "DONE", "FIXED", "DEPRECATED", "SUPERSEDED",
    "RETIRED", "WITHDRAWN", "REJECTED-BY-FALSIFIER", "⊘",
}


class Entry:
    __slots__ = ("line", "heading", "date", "eid", "status")

    def __init__(self, line, heading, date, eid, status):
        """One level-2 entry: 1-based line, raw heading, and its parsed fields."""
        self.line, self.heading, self.date = line, heading, date
        self.eid, self.status = eid, status


def parse(text: str):
    """-> (entries, structural, no_id) -- every level-2 heading lands in exactly one."""
    lines = text.split("\n")
    heads = [(i, m.group(1)) for i, l in enumerate(lines) if (m := H2.match(l))]
    entries, structural, no_id = [], [], []

    for n, (i, heading) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        body = lines[i:end]
        status = None
        for bl in body:
            m = STATUS.match(bl)
            if m:
                status = m.group(1).upper()
                break
        d = DATE.search(heading)
        e = EID.search(heading)
        eid = e.group(1).rstrip("-") if e else None
        if BARE_DATE.match(heading) or not d:
            structural.append(Entry(i + 1, heading, d.group(1) if d else None, None, status))
        elif not e:
            no_id.append(Entry(i + 1, heading, d.group(1), None, status))
        else:
            entries.append(Entry(i + 1, heading, d.group(1), eid, status))
    return entries, structural, no_id


def compact(entries):
    """-> (kept, dropped, dup_discarded, duplicates) by the two mechanical rules."""
    dropped = [e for e in entries if e.status in DROP]
    survivors = [e for e in entries if e.status not in DROP]

    seen, dup_discarded, duplicates = {}, [], {}
    for e in survivors:
        seen.setdefault(e.eid, []).append(e)
    kept = []
    for eid, occ in seen.items():
        if len(occ) > 1:
            # Newest surviving occurrence wins; the file is reverse-chronological,
            # so that is the EARLIEST line. Ordered by date, line as the tiebreak.
            occ_sorted = sorted(occ, key=lambda x: (x.date, -x.line), reverse=True)
            duplicates[eid] = occ_sorted
            kept.append(occ_sorted[0])
            dup_discarded.extend(occ_sorted[1:])
        else:
            kept.append(occ[0])
    kept.sort(key=lambda e: (e.date, -e.line), reverse=True)
    return kept, dropped, dup_discarded, duplicates


def finding_of(e: Entry) -> str:
    """Heading minus date/E-id boilerplate -- the claim, for `--measure` output.

    Ambiguous -> verbatim heading. This was also the fallback the one-off
    consolidation used when a worker echoed a status line instead of summarising.
    """
    s = e.heading
    for pat in (rf"^\s*{re.escape(e.date)}\s*[—–:-]*\s*", rf"^\s*{re.escape(e.eid)}\s*[—–:-]*\s*"):
        s2 = re.sub(pat, "", s, count=1)
        if s2 != s:
            s = s2
    s = s.strip()
    return (s or e.heading).replace("|", "\\|")


def cell(text, limit=110) -> str:
    """One-line, markdown-free, pipe-safe table cell text."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = t.replace("`", "").replace("|", "/").replace("*", "")
    if len(t) > limit:
        t = t[: limit - 1].rstrip() + "\u2026"
    return t


def load_rows(root, rows_rel: str):
    """Read the committed consolidation rows and validate their closed shape.

    Validation is not decoration: an out-of-vocabulary topic or a row whose line
    does not resolve to a real archive heading would silently put a claim in the
    table that nothing in the record backs.
    """
    p = pathlib.Path(root, rows_rel)
    if not p.is_file():
        raise SystemExit(
            "compact-epiphanies: %s is missing. It is the consolidation DATA "
            "and cannot be regenerated mechanically; restore it from git."
            % rows_rel
        )
    rows = json.loads(p.read_text(encoding="utf-8"))
    problems = []
    seen = set()
    for r in rows:
        ln = r.get("line")
        if not isinstance(ln, int):
            problems.append("non-integer line %r" % (ln,))
            continue
        if ln in seen:
            problems.append("line %d: duplicate row" % ln)
        seen.add(ln)
        if r.get("topic") not in TOPICS:
            problems.append("line %d: topic %r outside the closed vocabulary" % (ln, r.get("topic")))
        if r.get("priority") not in PRIORITIES:
            problems.append("line %d: priority %r" % (ln, r.get("priority")))
        if not str(r.get("finding") or "").strip():
            problems.append("line %d: empty finding" % ln)
    if problems:
        raise SystemExit("compact-epiphanies: %d invalid rows\n  %s" % (
            len(problems), "\n  ".join(problems[:20])))
    return rows


def assert_rows_resolve(rows, archive_text: str):
    """Every row must name a real level-2 heading line in the archive.

    The falsifier for the projection's provenance. Without it a row could
    describe an entry the archive does not contain, and the table would be
    unfalsifiable against its own source.
    """
    lines = archive_text.split("\n")
    bad = []
    for r in rows:
        i = r["line"] - 1
        if not (0 <= i < len(lines)) or not H2.match(lines[i]):
            bad.append(r["line"])
    if bad:
        raise SystemExit(
            "compact-epiphanies: %d rows do not resolve to an archive heading "
            "(first: %s). The rows file and the archive have diverged."
            % (len(bad), bad[:10])
        )


def render(rows, archive_rel: str, archive_raw: bytes) -> str:
    """The kanban projection. Deterministic: same rows in, byte-identical out."""
    live = [r for r in rows if not r.get("closed_out")]
    closed = len(rows) - len(live)
    # Priority first, then topic, then date: the table is read by "what is most
    # load-bearing in this area", not chronologically -- the archive is where
    # chronology lives.
    live.sort(key=lambda r: (r.get("priority", "P3"), r.get("topic", ""),
                             r.get("date") or "", r["line"]))

    out = [
        "# Epiphanies \u2014 compact kanban projection",
        "",
        "GENERATED by `.claude/tools/compact_epiphanies.py --write` from two inputs:",
        "the frozen archive `%s`" % pathlib.Path(archive_rel).name,
        "(%s bytes, %d lines, sha256 `%s`) and the committed consolidation rows"
        % (f"{len(archive_raw):,}", archive_raw.count(b"\n"),
           hashlib.sha256(archive_raw).hexdigest()[:16] + "\u2026"),
        "`%s`. Do not hand-edit this file \u2014 edit the rows and regenerate." % ROWS,
        "",
        "The ARCHIVE is the record: lossless, never appended to, never rewritten, and",
        "never read by routine work. This table is the PROJECTION: one line per entry,",
        "so the surface stays scannable while nothing is lost. Everything omitted here",
        "is in the archive, at the line each row names.",
        "",
        "Closed-out rows (%d of %d) are corrections whose fix landed, superseded or"
        % (closed, len(rows)),
        "retired rulings, and one-off process lessons already encoded in a test, a guard",
        "or CLAUDE.md. They are history, not live contract.",
        "",
        "New work goes to `.claude/board/entries/`. Only a rare surviving Eureka is",
        "promoted here, and a promotion must cite the entry it came from",
        "(`epiphany_provenance.py` enforces that the reference resolves).",
        "",
        "`P` is load-bearing-ness: P0 substrate law or canon pin, P1 architecture",
        "decision, P2 finding or measurement, P3 process note. `refs` is the entry's own",
        "provenance \u2014 PR, plan, D-id/ISS, else the commit that authored the heading.",
        "`line` is its line in the archive.",
        "",
        "Live rows: %d" % len(live),
        "",
        "| P | topic | date | id | status | finding | refs | line |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in live:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %d |" % (
            r["priority"], r["topic"], r.get("date") or "\u2014",
            cell(r.get("eid") or "\u2014", 52), cell(r.get("status") or "\u2014", 16),
            cell(r.get("finding"), 110), cell(r.get("refs") or "\u2014", 46),
            r["line"],
        ))
    out.append("")
    return "\n".join(out)


def load(path):
    """Read the archive, or refuse -- it is the only input this tool has."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise SystemExit(
            f"compact-epiphanies: {path} is missing. The archive is the ONLY input; "
            "create it byte-identically from EPIPHANIES.md before compacting."
        )
    return p.read_bytes()


def measure(root, archive):
    """Print the census and assert the accounting balances. Writes nothing."""
    raw = load(pathlib.Path(root, archive))
    text = raw.decode("utf-8", "replace")
    entries, structural, no_id = parse(text)
    kept, dropped, dup_discarded, duplicates = compact(entries)

    print("ARCHIVE  %s" % archive)
    print("  bytes %d | lines %d | sha256 %s" % (
        len(raw), text.count("\n"), hashlib.sha256(raw).hexdigest()))
    print("  level-2 headings %d = entries %d + no-id %d + structural %d" % (
        len(entries) + len(no_id) + len(structural), len(entries), len(no_id), len(structural)))
    print()
    print("COMPACTION")
    by_tok = {}
    for e in dropped:
        by_tok[e.status] = by_tok.get(e.status, 0) + 1
    print("  terminal dropped            %d" % len(dropped))
    for tok in sorted(by_tok, key=lambda k: -by_tok[k]):
        print("      %-24s %4d" % (tok, by_tok[tok]))
    print("  duplicate E-ids             %d (occurrences discarded %d)" % (
        len(duplicates), len(dup_discarded)))
    for eid, occ in sorted(duplicates.items()):
        print("      %s" % eid)
        print("          KEPT    line %-6d %s  %.60s" % (occ[0].line, occ[0].date, occ[0].heading))
        for o in occ[1:]:
            print("          OMITTED line %-6d %s  %.60s" % (o.line, o.date, o.heading))
    surv_status = {}
    for e in kept:
        surv_status[e.status or "(none)"] = surv_status.get(e.status or "(none)", 0) + 1
    print("  SHIPPED among survivors     %d" % surv_status.get("SHIPPED", 0))
    print("  no structured status        %d" % surv_status.get("(none)", 0))
    print("  PROJECTED SURVIVORS         %d" % len(kept))
    print()
    print("  accounting: %d entries = %d kept + %d terminal + %d dup-discarded -> %s" % (
        len(entries), len(kept), len(dropped), len(dup_discarded),
        "BALANCED" if len(entries) == len(kept) + len(dropped) + len(dup_discarded) else "REMAINDER!"))
    assert len(entries) == len(kept) + len(dropped) + len(dup_discarded), "silent remainder"
    print()
    print("SURVIVOR STATUS DISTRIBUTION (top 25)")
    for tok in sorted(surv_status, key=lambda k: -surv_status[k])[:25]:
        print("      %-28s %4d" % (tok, surv_status[tok]))
    return entries, structural, no_id, kept, dropped, dup_discarded, duplicates


def main(argv):
    """Dispatch --measure (default) / --write / --check / --self-test."""
    root = pathlib.Path(__file__).resolve().parents[2]
    archive = ARCHIVE
    if "--from" in argv:
        i = argv.index("--from")
        if i + 1 >= len(argv):
            print(__doc__)
            return 2
        archive = argv[i + 1]
    if "--self-test" in argv:
        return self_test()
    if "--measure" in argv or not argv:
        measure(root, archive)
        return 0
    raw = load(pathlib.Path(root, archive))
    rows = load_rows(root, ROWS)
    assert_rows_resolve(rows, raw.decode("utf-8", "replace"))
    body = render(rows, archive, raw)
    live = sum(1 for r in rows if not r.get("closed_out"))
    target = pathlib.Path(root, COMPACT)
    if "--check" in argv:
        cur = target.read_text(encoding="utf-8") if target.is_file() else ""
        if cur == body:
            print("compact-epiphanies: %s is current (%d live rows)" % (COMPACT, live))
            return 0
        print("::error::%s is stale. Regenerate: "
              "python3 .claude/tools/compact_epiphanies.py --write" % COMPACT)
        return 1
    if "--write" in argv:
        target.write_text(body, encoding="utf-8")
        print("compact-epiphanies: wrote %s (%d live rows of %d)"
              % (COMPACT, live, len(rows)))
        return 0
    print(__doc__)
    return 2


def self_test() -> int:
    """Falsifiers for both mechanical rules: the status token and the dedup."""
    ok = True

    def parse1(md):
        """Parse a markdown fixture; returns (entries, structural, no_id)."""
        e, s, n = parse(md)
        return e, s, n

    # (a) a CITED terminal marker must never drop the citing entry
    e, _, _ = parse1("## 2026-01-01 E-ALPHA-1 — keeps\n\n**Status:** FINDING\n\n"
                     "Sibling caveat (⊘ in E-FOOBAR-1) and SUPERSEDED prose here.\n")
    if len(e) != 1 or e[0].status != "FINDING" or e[0].status in DROP:
        print("  FAILED: a cited terminal marker leaked into the status"); ok = False
    else:
        print("  a cited ⊘/SUPERSEDED in prose   : not terminal (keeps)")

    # (b) the entry's OWN structured field IS read, and drops
    e, _, _ = parse1("## 2026-01-01 E-BRAVO-1 — goes\n\n**Status:** SUPERSEDED by E-CHARLIE-1\n")
    if not e or e[0].status != "SUPERSEDED" or e[0].status not in DROP:
        print(f"  FAILED: own structured SUPERSEDED not terminal ({e[0].status if e else None})"); ok = False
    else:
        print("  own **Status:** SUPERSEDED     : terminal (drops)")

    # (c) unknown and absent status KEEP -- never silently dropped
    e, _, _ = parse1("## 2026-01-01 E-DELTA-1 — unknown\n\n**Status:** FLUMMOXED\n"
                     "## 2026-01-01 E-ECHO-1 — absent\n\nno field at all\n")
    kept, dropped, _, _ = compact(e)
    if len(kept) != 2 or dropped:
        print(f"  FAILED: unknown/absent status did not keep ({len(kept)} kept, {len(dropped)} dropped)"); ok = False
    else:
        print("  unknown + absent status        : both keep")

    # (d) SHIPPED is deliberately NOT terminal
    e, _, _ = parse1("## 2026-01-01 E-FOXTROT-1 — shipped\n\n**Status:** SHIPPED in #1\n")
    kept, dropped, _, _ = compact(e)
    if len(kept) != 1 or dropped:
        print("  FAILED: SHIPPED was dropped"); ok = False
    else:
        print("  SHIPPED                        : keeps (may still be a Eureka)")

    # (e) exact-E-id dedup keeps the NEWEST surviving occurrence
    e, _, _ = parse1("## 2026-05-05 E-GOLF-1 — newer\n\n**Status:** FINDING\n"
                     "## 2026-01-01 E-GOLF-1 — older\n\n**Status:** FINDING\n")
    kept, _, dup, dups = compact(e)
    if len(kept) != 1 or kept[0].date != "2026-05-05" or len(dup) != 1 or "E-GOLF-1" not in dups:
        print(f"  FAILED: dedup kept the wrong occurrence ({[(k.date) for k in kept]})"); ok = False
    else:
        print("  duplicate E-id                 : newest kept, older reported")

    # (f) bare date groups and id-less headings are NOT entries
    e, s, n = parse1("## 2026-08-09\n\n## 2026-05-31 — FINDING (no id): x\n\n## How to use\n")
    if len(e) != 0 or len(n) != 1 or len(s) != 2:
        print(f"  FAILED: heading triage wrong (entries {len(e)}, no-id {len(n)}, structural {len(s)})"); ok = False
    else:
        print("  bare date / id-less / section  : 0 entries, 1 no-id, 2 structural")

    # (g) idempotence: rendering twice is byte-identical
    rows = [{"line": 1, "date": "2026-01-01", "eid": "E-HOTEL-1", "status": "FINDING",
             "topic": "board-process", "priority": "P2", "finding": "x",
             "refs": "\u2014", "closed_out": False}]
    if render(rows, ARCHIVE, b"##\n") != render(rows, ARCHIVE, b"##\n"):
        print("  FAILED: render is not deterministic"); ok = False
    else:
        print("  render determinism             : byte-identical")

    # (h) a closed-out row leaves the table but stays COUNTED in the header --
    # the accounting the archive-is-lossless claim rests on.
    rows2 = rows + [dict(rows[0], line=2, eid="E-INDIA-1", closed_out=True)]
    body = render(rows2, ARCHIVE, b"##\n")
    if "E-INDIA-1" in body or "Live rows: 1" not in body or "(1 of 2)" not in body:
        print("  FAILED: closed-out row not omitted-and-counted"); ok = False
    else:
        print("  closed-out row                 : omitted from table, counted in header")

    # (i) a row that does not resolve to an archive heading is REFUSED. The
    # projection's provenance falsifier: without it the table could carry a
    # claim the archive does not contain.
    try:
        assert_rows_resolve([{"line": 2}], "## real heading\nprose\n")
    except SystemExit:
        print("  row off a non-heading line     : refused")
    else:
        print("  FAILED: a non-heading row line was accepted"); ok = False
    try:
        assert_rows_resolve([{"line": 1}], "## real heading\nprose\n")
    except SystemExit:
        print("  FAILED: a valid heading row was refused"); ok = False
    else:
        print("  row on a real heading line     : accepted")

    # (j) the topic vocabulary is CLOSED -- an unknown tag is a defect, because a
    # table whose topic column is open-ended cannot be scanned by topic.
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td, ".claude", "board")
        d.mkdir(parents=True)
        bad = dict(rows[0], topic="freshly-invented")
        (d / "epiphanies-rows.json").write_text(json.dumps([bad]), encoding="utf-8")
        try:
            load_rows(td, ROWS)
        except SystemExit:
            print("  topic outside the vocabulary   : refused")
        else:
            print("  FAILED: an unknown topic was accepted"); ok = False
        (d / "epiphanies-rows.json").write_text(json.dumps(rows), encoding="utf-8")
        if len(load_rows(td, ROWS)) != 1:
            print("  FAILED: a valid rows file was refused"); ok = False
        else:
            print("  valid rows file                : accepted")

    print("compact-epiphanies --self-test " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
