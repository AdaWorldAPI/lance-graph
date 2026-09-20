#!/usr/bin/env python3
"""Project the immutable EPIPHANIES archive into a compact canonical table.

    compact_epiphanies.py --measure     # archive census, writes nothing
    compact_epiphanies.py --write       # render EPIPHANIES.md
    compact_epiphanies.py --check       # the projection matches its inputs
    compact_epiphanies.py --self-test   # falsifiers

TWO INPUTS, AND THE SPLIT IS THE DESIGN
---------------------------------------
  1. the ARCHIVE -- byte-identical to the pre-compaction file, frozen. Its
     sha256 is recorded in the decisions file and checked on every run, so a
     mutation is caught rather than silently projected.
  2. the DECISIONS file -- `.claude/board/epiphanies-canon.json`, the compact
     generated result of a ONE-TIME semantic closeout: a state per historical
     entry, plus one canonical row per surviving MECHANISM.

     It is keyed by ARCHIVE LINE, not by id. Three ids legitimately recur in
     this corpus, so an id-keyed map is ambiguous; and a line key cannot drift
     from the archive the way a stored display string could. Every id shown in
     the table is recomputed from the archive at render time.

This tool parses, accounts and renders. It holds no semantic heuristics: the
judgement lives in the data block, never in clever regex here. A general
reconciliation service is explicitly not what this is.

WHY THE MECHANICAL STATUS FILTER IS ONLY A CENSUS
------------------------------------------------
It was measured and rejected as the compaction mechanism. This corpus grades
EPISTEMICALLY (FINDING / RULING / MEASURED / CORRECTION) -- how well established
a claim is, never whether it is done -- so dropping explicit terminal tokens took
891 rows to 877. No compaction at all. `--measure` keeps that census, which is
what the parser is honestly good for; compaction comes from the closeout.

IDENTITY IS POSITIONAL, NEVER A SEARCH
--------------------------------------
An entry's id is the leading token of the identifier POSITION (after the date),
or the entry owns none. Searching the whole heading takes a CITED sibling's id,
which measurably happened. An entry with no id of its own is keyed by its
archive line (`L12345`) -- a stable coordinate, because the archive is
immutable. Synthetic ids are never minted.
"""

import hashlib
import json
import pathlib
import re
import sys
import tempfile

ARCHIVE = ".claude/board/EPIPHANIES-ARCHIVE-2026-09-20.md"
COMPACT = ".claude/board/EPIPHANIES.md"
CANON = ".claude/board/epiphanies-canon.json"

H2 = re.compile(r"^##\s+(?!#)(.*)$")
# The date position may carry a same-day counter, `2026-09-14 (3)`. It belongs
# to the DATE, not to the identifier that follows it: 17 entries that own a
# real id were being keyed by archive line because `(` stopped the separator
# scan. Found by the merge verification, not by reading -- the worker returned
# an id where the parser had found none.
DATE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})(?:\s*\(\d+\))?\s*")
BARE_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")
LEAD_SEP = re.compile(r"^\s*(?:[—–-]+|:)\s*")
# Anchored at the identifier position. `[A-Za-z0-9]` after the prefix admits
# `E-0xFFF-...` (the lowercase x) and a trailing `-1a`/`-1b` disambiguator is
# part of the id, never truncated -- both were measured losses.
OWN_ID = re.compile(r"^(?:E|I)-[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9]")

# The entry's OWN structured grade field, leading token only, never prose. The
# measured trap is a sibling's caveat quoted as `(⊘ in E-FOO-1)`, which a
# substring test reads as this entry being retired.
GRADE = re.compile(
    # NOTE the `\s*` after the trailing `\*{0,2}`: the corpus writes
    # `**Status:** RESOLVED`, so the bold CLOSES AFTER the colon and a space
    # sits before the token. Without it this matches nothing in this corpus.
    r"^\s*[>*\-\s]*\*{0,2}(?:Status|Verdict|State)\*{0,2}\s*[:—-]\s*\*{0,2}\s*([A-Za-z⊘-]+)"
)

# Explicitly terminal grade tokens, for the census only.
TERMINAL_GRADES = {
    "CLOSED", "DONE", "FIXED", "DEPRECATED", "SUPERSEDED",
    "RETIRED", "WITHDRAWN", "REJECTED-BY-FALSIFIER", "⊘",
}

# Closeout states. Only the first three reach the table; the rest stay in the
# archive, which is why nothing is lost by omitting them here.
LIVE_STATES = ("CURRENT", "OPEN", "UNKNOWN")
TERMINAL_STATES = ("CLOSED", "SUPERSEDED", "DEPRECATED", "REJECTED", "DUPLICATE")
STATES = set(LIVE_STATES) | set(TERMINAL_STATES)

# Closed on purpose: a table whose topic column is open-ended cannot be scanned.
METAS = {
    "masking/folding", "CE64", "counterfactual/revision", "temporal/versioning",
    "NodeGuid/addressing", "OGAR/loco/R2IL", "DeepNSM", "BatchWriter/Kanban",
    "storage/SoA", "planner", "membrane/BBB", "simd/ndarray", "codec/palette",
    "deps/build", "evidence/method", "process/tooling",
}
PRIORITIES = {"P0", "P1", "P2", "P3", "-"}


def own_identity(heading):
    """-> (date, id_or_None), the id taken from the identifier POSITION only."""
    d = DATE.match(heading)
    if not d:
        return None, None
    rest = LEAD_SEP.sub("", heading[d.end():], count=1)
    m = OWN_ID.match(rest)
    if not m:
        return d.group(1), None
    tok = m.group(0)
    # A trailing `-` means the match stopped mid-token: refuse and let the row
    # be keyed by line rather than guess at an id.
    return d.group(1), (None if tok.endswith("-") else tok)


def source_key(line, ident):
    """The row's stable coordinate: its own id, else an immutable archive line."""
    return ident or ("L%d" % line)


def parse(text):
    """-> (entries, structural); every level-2 heading lands in exactly one.

    An entry is a dated heading. `structural` is bare date groups and prose
    sections -- they carry no finding and are not part of the population.
    """
    lines = text.split("\n")
    heads = [(i + 1, m.group(1)) for i, l in enumerate(lines) if (m := H2.match(l))]
    entries, structural = [], []
    for n, (ln, heading) in enumerate(heads):
        end = heads[n + 1][0] - 1 if n + 1 < len(heads) else len(lines)
        date, ident = own_identity(heading)
        if BARE_DATE.match(heading) or not date:
            structural.append((ln, heading))
            continue
        grade = None
        for bline in lines[ln - 1:end]:
            if (m := GRADE.match(bline)):
                grade = m.group(1).upper()
                break
        entries.append({
            "line": ln, "heading": heading, "date": date, "id": ident,
            "source": source_key(ln, ident), "grade": grade,
        })
    return entries, structural


def load_archive(root, rel):
    """Read the archive, or refuse: it is the record and cannot be regenerated."""
    p = pathlib.Path(root, rel)
    if not p.is_file():
        raise SystemExit(
            "compact-epiphanies: %s is missing. It is the immutable record; "
            "restore it from git rather than recreating it." % rel)
    return p.read_bytes()


def load_canon(root, rel=CANON):
    """Read the closeout decisions and validate their closed shape."""
    p = pathlib.Path(root, rel)
    if not p.is_file():
        raise SystemExit(
            "compact-epiphanies: %s is missing. It is the one-time closeout "
            "DATA and cannot be regenerated mechanically; restore it from git."
            % rel)
    doc = json.loads(p.read_text(encoding="utf-8"))
    problems = []
    for key, st in doc.get("decisions", {}).items():
        if not str(key).isdigit():
            problems.append("decision key %r is not an archive line" % (key,))
        if st not in STATES:
            problems.append("decision %s: state %r" % (key, st))
    seen = set()
    for i, r in enumerate(doc.get("rows", [])):
        if r.get("state") not in LIVE_STATES:
            problems.append("row %d: state %r is not live" % (i, r.get("state")))
        if r.get("meta") not in METAS:
            problems.append("row %d: meta %r outside the closed vocabulary" % (i, r.get("meta")))
        if r.get("P") not in PRIORITIES:
            problems.append("row %d: P %r" % (i, r.get("P")))
        if not str(r.get("insight") or "").strip():
            problems.append("row %d: empty insight" % i)
        if not r.get("lines"):
            problems.append("row %d: no archive lines" % i)
        for ln in r.get("lines", []):
            if not isinstance(ln, int):
                problems.append("row %d: line %r is not an integer" % (i, ln))
            if ln in seen:
                problems.append("row %d: archive line %s appears in two rows" % (i, ln))
            seen.add(ln)
    if problems:
        raise SystemExit("compact-epiphanies: %d invalid decisions\n  %s"
                         % (len(problems), "\n  ".join(problems[:20])))
    return doc


def assert_archive_unchanged(doc, raw):
    """Every recorded archive measurement must match the file being projected.

    Without this the table could describe a file that has since moved, which is
    the one way an immutable-record claim can quietly become false.

    A MISSING hash is REFUSED, never skipped. The first version read
    `if want and want != got`, so an absent or empty sha256 disabled the
    immutability check silently -- and a guard that cannot fire carries exactly
    as much information as one that never fires. This one exists for the single
    claim the whole design rests on.

    Bytes and lines are checked against the actual file too. They are printed in
    the projection's header, so leaving them unchecked would let one file's
    measurements be reported over another file's content.
    """
    arc = doc.get("archive") or {}
    want = arc.get("sha256")
    got = hashlib.sha256(raw).hexdigest()
    if not want:
        raise SystemExit(
            "compact-epiphanies: the decisions file records no archive sha256, "
            "so the immutability check cannot run. Restore the recorded hash; "
            "do not project an unverified archive.")
    if want != got:
        raise SystemExit(
            "compact-epiphanies: ARCHIVE MUTATED\n  recorded %s\n  actual   %s\n"
            "  The archive is immutable. Restore it; do not re-record the hash."
            % (want, got))
    nl = raw.count(b"\n")
    if (arc.get("bytes"), arc.get("lines")) != (len(raw), nl):
        raise SystemExit(
            "compact-epiphanies: archive measurements do not match the file\n"
            "  recorded %s bytes, %s lines\n  actual   %s bytes, %s lines"
            % (arc.get("bytes"), arc.get("lines"), len(raw), nl))
    return got


def assert_sources_resolve(doc, entries):
    """Every decision and row source must be a real archive entry.

    The projection's provenance falsifier: without it a row could state a claim
    the archive does not contain, and the table would be unfalsifiable against
    its own source.
    """
    known = {e["line"] for e in entries}
    bad = sorted({int(k) for k in doc.get("decisions", {})} - known)
    orphan = sorted({ln for r in doc.get("rows", []) for ln in r["lines"]}
                    - {int(k) for k in doc.get("decisions", {})})
    if bad or orphan:
        raise SystemExit(
            "compact-epiphanies: %d decisions name no archive entry (first %s); "
            "%d row lines have no decision (first %s)"
            % (len(bad), bad[:6], len(orphan), orphan[:6]))


def account(doc, entries):
    """-> (counts, live_lines). Asserts the population balances."""
    dec = {int(k): v for k, v in doc.get("decisions", {}).items()}
    counts = {}
    for st in dec.values():
        counts[st] = counts.get(st, 0) + 1
    undecided = [e["line"] for e in entries if e["line"] not in dec]
    if undecided:
        raise SystemExit("compact-epiphanies: %d entries have no closeout decision "
                         "(first %s)" % (len(undecided), undecided[:6]))
    live = {ln for ln, st in dec.items() if st in LIVE_STATES}
    covered = {ln for r in doc["rows"] for ln in r["lines"]}
    missing, extra = sorted(live - covered), sorted(covered - live)
    if missing or extra:
        raise SystemExit(
            "compact-epiphanies: live entries and canonical rows disagree -- "
            "%d live not in any row (first %s), %d row lines not live (first %s)"
            % (len(missing), missing[:6], len(extra), extra[:6]))
    return counts, live


def cell(text, limit=170):
    """One-line, markdown-free, pipe-safe table cell."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = t.replace("`", "").replace("|", "/")
    return t[: limit - 1].rstrip() + "…" if len(t) > limit else t


def render(doc, raw, entries):
    """The canonical table. Deterministic: same inputs, byte-identical output."""
    counts, live = account(doc, entries)
    label = {e["line"]: e["source"] for e in entries}
    rows = sorted(doc["rows"], key=lambda r: (r["meta"], r["state"] != "OPEN",
                                              r.get("date") or "", r["insight"]))
    arc = doc["archive"]
    terminal = sum(counts.get(s, 0) for s in TERMINAL_STATES)
    out = [
        "# Epiphanies — canonical architecture map",
        "",
        "One line per SURVIVING MECHANISM. Generated by",
        "`.claude/tools/compact_epiphanies.py --write`; do not hand-edit — change",
        "`%s` and regenerate." % CANON,
        "",
        "| | |",
        "|---|---|",
        "| archive | `%s` |" % arc["file"],
        "| | %s bytes, %s lines, sha256 `%s` |" % (
            # MEASURED from the file in hand, never copied from the decisions
            # block: this header is a claim about what was projected.
            f"{len(raw):,}", f"{raw.count(chr(10).encode()):,}",
            arc["sha256"][:16] + "…"),
        "| historical entries | %d |" % len(entries),
        "| live after closeout | %d (%s) |" % (
            len(live), ", ".join("%s %d" % (s, counts.get(s, 0))
                                 for s in LIVE_STATES if counts.get(s))),
        "| consolidated into | **%d canonical concepts** |" % len(rows),
        "| terminal, archive only | %d (%s) |" % (
            terminal, ", ".join("%s %d" % (s, counts.get(s, 0))
                                for s in TERMINAL_STATES if counts.get(s))),
        "",
        "The ARCHIVE is the record: lossless, immutable, never appended to, and never",
        "read by routine work. This table is the PROJECTION and is lossy by design —",
        "successive iterations of one mechanism (hypothesis, correction, measured",
        "refinement, final ruling) collapse to the one row stating what survived, and",
        "terminal findings stay in the archive. Nothing is deleted from history.",
        "",
        "`source` is each concept's surviving id(s), or an immutable archive line",
        "anchor (`L12345`) where the entry never owned one. No id was ever minted.",
        "`P` is a priority the entry itself stated; `—` means it stated none.",
        "`impl` is evidence from the entry, never an inferred commit.",
        "",
        "New work goes to `.claude/board/entries/` and dies at closeout, or becomes an",
        "OPEN row, or — rarely — a Eureka promoted here citing its entry",
        "(`epiphany_provenance.py` enforces that the reference resolves).",
        "",
        "| date | source | status | P | meta | canonical insight | implementation | plan / context |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("date") or "—",
            cell(", ".join(label[ln] for ln in r["lines"]), 64),
            r["state"],
            r["P"].replace("-", "—"),
            r["meta"],
            cell(r["insight"], 170),
            cell(r.get("impl") or "—", 34),
            cell(r.get("plan") or "—", 56),
        ))
    out.append("")
    return "\n".join(out)


def measure(root, rel):
    """Print the archive census and the closeout accounting. Writes nothing."""
    raw = load_archive(root, rel)
    text = raw.decode("utf-8", "replace")
    entries, structural = parse(text)
    sha = hashlib.sha256(raw).hexdigest()

    print("ARCHIVE  %s" % rel)
    print("  bytes %d | lines %d | sha256 %s" % (len(raw), text.count("\n"), sha))
    print("  level-2 headings %d = entries %d + structural %d"
          % (len(entries) + len(structural), len(entries), len(structural)))
    print()
    print("IDENTITY (positional, never a search; no id is ever minted)")
    e_ids = [e for e in entries if e["id"]]
    print("  E-* %d | I-* %d | archive-line-keyed %d"
          % (sum(1 for e in e_ids if e["id"].startswith("E-")),
             sum(1 for e in e_ids if e["id"].startswith("I-")),
             sum(1 for e in entries if not e["id"])))
    dups = {}
    for e in entries:
        dups.setdefault(e["source"], []).append(e)
    rep = {k: v for k, v in dups.items() if len(v) > 1}
    print("  sources appearing more than once %d %s" % (len(rep), sorted(rep)[:6]))
    print()
    print("MECHANICAL GRADE CENSUS (why this is NOT the compaction mechanism)")
    g = {}
    for e in entries:
        g[e["grade"] or "(none)"] = g.get(e["grade"] or "(none)", 0) + 1
    term = sum(v for k, v in g.items() if k in TERMINAL_GRADES)
    print("  explicitly terminal grade %d of %d entries -- dropping these alone "
          "compacts nothing" % (term, len(entries)))
    for k in sorted(g, key=lambda x: -g[x])[:8]:
        print("      %-28s %4d" % (k, g[k]))
    print()

    canon = pathlib.Path(root, CANON)
    if not canon.is_file():
        print("CLOSEOUT  %s not present yet" % CANON)
        return entries, structural, None
    doc = load_canon(root)
    assert_archive_unchanged(doc, raw)
    assert_sources_resolve(doc, entries)
    counts, live = account(doc, entries)
    print("SEMANTIC CLOSEOUT")
    for st in LIVE_STATES + TERMINAL_STATES:
        print("      %-12s %4d" % (st, counts.get(st, 0)))
    print("  accounting: %d entries = %d decided -> %s"
          % (len(entries), sum(counts.values()),
             "BALANCED" if sum(counts.values()) == len(entries) else "REMAINDER!"))
    assert sum(counts.values()) == len(entries), "silent remainder"
    print()
    print("CONSOLIDATION")
    print("  historical entries            %d" % len(entries))
    print("  live after closeout           %d" % len(live))
    print("  canonical concept rows        %d" % len(doc["rows"]))
    print("  collapse ratio                %.1fx" % (len(live) / max(1, len(doc["rows"]))))
    print()
    print("IMPLEMENTATION")
    withimpl = sum(1 for r in doc["rows"] if (r.get("impl") or "-") not in ("-", ""))
    withplan = sum(1 for r in doc["rows"] if (r.get("plan") or "").strip())
    neither = sum(1 for r in doc["rows"]
                  if (r.get("impl") or "-") in ("-", "") and not (r.get("plan") or "").strip())
    print("  rows with PR/commit %d | with plan %d | with neither %d"
          % (withimpl, withplan, neither))
    return entries, structural, doc


def main(argv):
    """Dispatch --measure (default) / --write / --check / --self-test."""
    root = pathlib.Path(__file__).resolve().parents[2]
    rel = ARCHIVE
    if "--from" in argv:
        i = argv.index("--from")
        if i + 1 >= len(argv):
            print(__doc__)
            return 2
        rel = argv[i + 1]
    if "--self-test" in argv:
        return self_test()
    if "--measure" in argv or not argv:
        measure(root, rel)
        return 0

    raw = load_archive(root, rel)
    entries, _ = parse(raw.decode("utf-8", "replace"))
    doc = load_canon(root)
    assert_archive_unchanged(doc, raw)
    assert_sources_resolve(doc, entries)
    body = render(doc, raw, entries)
    target = pathlib.Path(root, COMPACT)
    if "--check" in argv:
        cur = target.read_text(encoding="utf-8") if target.is_file() else ""
        if cur == body:
            print("compact-epiphanies: %s is current (%d canonical rows)"
                  % (COMPACT, len(doc["rows"])))
            return 0
        print("::error::%s is stale. Regenerate: "
              "python3 .claude/tools/compact_epiphanies.py --write" % COMPACT)
        return 1
    if "--write" in argv:
        target.write_text(body, encoding="utf-8")
        print("compact-epiphanies: wrote %s (%d canonical rows from %d entries)"
              % (COMPACT, len(doc["rows"]), len(entries)))
        return 0
    print(__doc__)
    return 2


def _doc(rows, decisions, sha="0" * 64):
    """A minimal valid decisions document, for the falsifiers."""
    # bytes/lines match the b"##\n" fixture every falsifier passes in.
    return {"archive": {"file": "A.md", "sha256": sha, "bytes": 3, "lines": 1},
            "decisions": decisions, "rows": rows}


def _row(**kw):
    """A minimal valid canonical row, for the falsifiers."""
    base = {"date": "2026-01-01", "lines": [1], "state": "CURRENT",
            "P": "-", "meta": "planner", "insight": "x", "impl": "-", "plan": ""}
    base.update(kw)
    return base


def self_test():
    """Falsifiers. Each proves a rule can REFUSE, not merely that it can pass."""
    ok = True

    def check(label, fn, want_refusal):
        """Run fn; report whether it refused, against what the rule requires."""
        nonlocal ok
        try:
            fn()
            refused = False
        except SystemExit:
            refused = True
        good = refused == want_refusal
        ok = ok and good
        print("  %s %-46s %s" % ("ok  " if good else "FAIL", label,
                                 "refused" if refused else "accepted"))

    # --- identity: positional, never a search -------------------------------
    cases = [
        ("2026-06-12 — E-0xFFF-IS-ONE-ALIGNED-ADDRESS — x",
         "E-0xFFF-IS-ONE-ALIGNED-ADDRESS", "lowercase x in the id"),
        ("2026-07-01 — E-X265-MORTON-SHIFT-1a — x",
         "E-X265-MORTON-SHIFT-1a", "trailing -1a kept"),
        ("2026-07-01 — E-X265-MORTON-SHIFT-1b — x",
         "E-X265-MORTON-SHIFT-1b", "its -1b sibling, no collision"),
        ("2026-05-31 — FINDING: masks fold, per E-OTHER-1", None,
         "owns none, CITES one: must not borrow"),
        ("2026-08-02 — I-STRINGS-ARE-INTERNED — x",
         "I-STRINGS-ARE-INTERNED", "I-* is a first-class identity"),
    ]
    for heading, want, why in cases:
        got = own_identity(heading)[1]
        if got != want:
            print("  FAIL identity %-40s got %r want %r" % (why, got, want))
            ok = False
        else:
            print("  ok   identity %-40s -> %s" % (why, got))
    if source_key(12345, None) != "L12345":
        print("  FAIL id-less entry did not fall back to an archive anchor"); ok = False
    else:
        print("  ok   identity %-40s -> L12345" % "id-less entry keyed by archive line")

    # --- grade parser: own field only ---------------------------------------
    e, _ = parse("## 2026-01-01 — E-ALPHA-1 — k\n\n**Status:** FINDING\n\n"
                 "Sibling caveat (⊘ in E-FOOBAR-1), SUPERSEDED prose.\n")
    if len(e) != 1 or e[0]["grade"] != "FINDING":
        print("  FAIL a cited terminal marker leaked into the grade"); ok = False
    else:
        print("  ok   grade     %-40s -> FINDING" % "cited ⊘/SUPERSEDED in prose")
    e, s = parse("## 2026-08-09\n\n## How to use\n\n## 2026-01-01 — E-B-1 — k\n")
    if len(e) != 1 or len(s) != 2:
        print("  FAIL heading triage wrong (%d entries, %d structural)" % (len(e), len(s)))
        ok = False
    else:
        print("  ok   grade     %-40s -> 1 entry, 2 structural" % "bare date + prose section")

    # --- the archive really is immutable ------------------------------------
    real = hashlib.sha256(b"##\n").hexdigest()
    check("archive hash matches the recorded one",
          lambda: assert_archive_unchanged(_doc([], {}, real), b"##\n"), False)
    check("archive mutated under a recorded hash",
          lambda: assert_archive_unchanged(_doc([], {}, real), b"##!\n"), True)
    # The guard must REFUSE a missing hash, not skip the check. `if want and
    # want != got` passed here, which is a guard that cannot fire.
    # The fixture's bytes/lines are CORRECT on purpose. With them omitted the
    # byte/line check refused instead, so the arm passed with the hash-presence
    # check disabled -- vacuous, and only the disable run showed it. Now the
    # ONLY thing wrong with this document is the missing hash.
    check("no recorded hash, everything else correct",
          lambda: assert_archive_unchanged(
              {"archive": {"bytes": 3, "lines": 1}}, b"##\n"), True)
    check("recorded byte/line counts that do not match the file",
          lambda: assert_archive_unchanged(
              {"archive": {"sha256": real, "bytes": 999, "lines": 7}}, b"##\n"), True)

    # --- provenance: a row cannot outrun the archive ------------------------
    ent = [{"line": 1, "source": "E-ALPHA-1"}]
    check("a decision naming a real entry",
          lambda: assert_sources_resolve(_doc([_row()], {"1": "CURRENT"}), ent), False)
    check("a decision naming no archive entry",
          lambda: assert_sources_resolve(_doc([], {"999": "CURRENT"}), ent), True)

    # --- accounting: no entry silently vanishes -----------------------------
    ent2 = [{"line": 1, "source": "E-ALPHA-1"}, {"line": 99, "source": "L99"}]
    check("every entry decided and every live one in a row",
          lambda: account(_doc([_row()], {"1": "CURRENT", "99": "CLOSED"}), ent2), False)
    check("an entry with no closeout decision",
          lambda: account(_doc([_row()], {"1": "CURRENT"}), ent2), True)
    check("a live entry absent from every canonical row",
          lambda: account(_doc([_row()], {"1": "CURRENT", "99": "OPEN"}), ent2), True)

    # --- the closed vocabularies -------------------------------------------
    def load_tmp(doc):
        """Write a decisions doc to a temp tree and load it through the validator."""
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td, ".claude", "board")
            d.mkdir(parents=True)
            (d / "epiphanies-canon.json").write_text(json.dumps(doc), encoding="utf-8")
            return load_canon(td)

    check("a valid decisions file",
          lambda: load_tmp(_doc([_row()], {"1": "CURRENT"})), False)
    check("meta outside the closed vocabulary",
          lambda: load_tmp(_doc([_row(meta="freshly-invented")], {"1": "CURRENT"})), True)
    check("a terminal state given a canonical row",
          lambda: load_tmp(_doc([_row(state="CLOSED")], {"1": "CLOSED"})), True)
    check("one entry claimed by two rows",
          lambda: load_tmp(_doc([_row(), _row(insight="y")], {"1": "CURRENT"})), True)
    check("an invented priority token",
          lambda: load_tmp(_doc([_row(P="P9")], {"1": "CURRENT"})), True)
    check("a decision keyed by an id instead of a line",
          lambda: load_tmp(_doc([_row()], {"E-ALPHA-1": "CURRENT"})), True)

    # --- render is deterministic, and terminal rows never surface -----------
    doc = _doc([_row(), _row(lines=[99], insight="y", meta="CE64")],
               {"1": "CURRENT", "99": "OPEN", "7": "SUPERSEDED"})
    ent3 = [{"line": 1, "source": "E-ALPHA-1"}, {"line": 99, "source": "L99"},
            {"line": 7, "source": "E-GONE-1"}]
    a = render(doc, b"##\n", ent3)
    if a != render(doc, b"##\n", ent3):
        print("  FAIL render is not deterministic"); ok = False
    else:
        print("  ok   render    %-40s byte-identical" % "same inputs twice")
    if "E-GONE-1" in a or "SUPERSEDED 1" not in a:
        print("  FAIL a terminal source was not omitted-and-counted"); ok = False
    else:
        print("  ok   render    %-40s omitted, still counted" % "SUPERSEDED source")

    print("compact-epiphanies --self-test " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
