#!/usr/bin/env python3
"""Mechanically project the EPIPHANIES archive into a compact canonical table.

    compact_epiphanies.py --measure           # census only, writes nothing
    compact_epiphanies.py --write             # emit the compact EPIPHANIES.md
    compact_epiphanies.py --check             # compact file is current
    compact_epiphanies.py --self-test         # falsifiers

WHAT THIS IS, AND ONLY THIS
---------------------------
COMPACTION, never adjudication. It reads ONE input (the archive), inspects each
entry's own STRUCTURED leading status token, drops the explicitly terminal ones,
deduplicates exact E-ids, and emits a table. It consults no code, no
TECH_DEBT/STATUS_BOARD, no PR state, no plan, no supersession prose, no citation
decay, and no model. A finding's meaning is never reinterpreted here.

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
import pathlib
import re
import sys

ARCHIVE = ".claude/board/EPIPHANIES-ARCHIVE-2026-09-20.md"
COMPACT = ".claude/board/EPIPHANIES.md"

H2 = re.compile(r"^##\s+(?!#)(.*)$")
EID = re.compile(r"\b(E-[A-Z0-9][A-Z0-9-]{3,})\b")
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
        if BARE_DATE.match(heading) or not d:
            structural.append(Entry(i + 1, heading, d.group(1) if d else None, None, status))
        elif not e:
            no_id.append(Entry(i + 1, heading, d.group(1), None, status))
        else:
            entries.append(Entry(i + 1, heading, d.group(1), e.group(1), status))
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
    """Heading minus date/E-id boilerplate. Ambiguous -> verbatim heading."""
    s = e.heading
    for pat in (rf"^\s*{re.escape(e.date)}\s*[—–:-]*\s*", rf"^\s*{re.escape(e.eid)}\s*[—–:-]*\s*"):
        s2 = re.sub(pat, "", s, count=1)
        if s2 != s:
            s = s2
    s = s.strip()
    return (s or e.heading).replace("|", "\\|")


def render(kept, archive_rel: str) -> str:
    """The compact table. Deterministic: same input, byte-identical output."""
    out = [
        "# Epiphanies",
        "",
        f"Historical source: `{archive_rel}`",
        "",
        "This file is the COMPACT CANONICAL PROJECTION of that archive, generated by",
        "`.claude/tools/compact_epiphanies.py`. Terminal findings (explicitly CLOSED /",
        "DONE / FIXED / DEPRECATED / SUPERSEDED / RETIRED / WITHDRAWN /",
        "REJECTED-BY-FALSIFIER / ⊘) and older duplicate occurrences of the same E-id are",
        "PRESERVED IN THE ARCHIVE and omitted here. Nothing was summarised, reworded or",
        "reinterpreted: each row's text is its own heading.",
        "",
        "The archive is lossless history and is never appended to, never rewritten, and",
        "never read by routine work. New work goes to `.claude/board/entries/`; only a",
        "rare surviving Eureka is promoted here, and a promotion must cite the entry it",
        "came from (`epiphany_provenance.py` enforces that the reference resolves).",
        "",
        f"Rows: {len(kept)}",
        "",
        "| date | id | status | finding |",
        "|---|---|---|---|",
    ]
    for e in kept:
        out.append(f"| {e.date} | {e.eid} | {e.status or '—'} | {finding_of(e)} |")
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
        archive = argv[argv.index("--from") + 1]
    if "--self-test" in argv:
        return self_test()
    if "--measure" in argv or not argv:
        measure(root, archive)
        return 0
    raw = load(pathlib.Path(root, archive))
    entries, _s, _n = parse(raw.decode("utf-8", "replace"))
    kept, *_ = compact(entries)
    body = render(kept, ARCHIVE)
    target = pathlib.Path(root, COMPACT)
    if "--check" in argv:
        cur = target.read_text(errors="replace") if target.is_file() else ""
        if cur == body:
            print("compact-epiphanies: %s is current (%d rows)" % (COMPACT, len(kept)))
            return 0
        print("::error::%s is stale. Regenerate: "
              "python3 .claude/tools/compact_epiphanies.py --write" % COMPACT)
        return 1
    if "--write" in argv:
        target.write_text(body)
        print("compact-epiphanies: wrote %s (%d rows)" % (COMPACT, len(kept)))
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
    e, _, _ = parse1("## 2026-01-01 E-HOTEL-1 — x\n\n**Status:** FINDING\n")
    k, *_ = compact(e)
    if render(k, ARCHIVE) != render(k, ARCHIVE):
        print("  FAILED: render is not deterministic"); ok = False
    else:
        print("  render determinism             : byte-identical")

    print("compact-epiphanies --self-test " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
