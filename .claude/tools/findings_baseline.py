#!/usr/bin/env python3
"""Reconcile EPIPHANIES.md findings into OPEN / CLOSED / SUPERSEDED / AMBIGUOUS.

    python3 .claude/tools/findings_baseline.py --report
    python3 .claude/tools/findings_baseline.py --emit <out.md>
    python3 .claude/tools/findings_baseline.py --self-test

WHY THIS IS A TOOL AND NOT A ONE-OFF SCRIPT
-------------------------------------------
It produced the numbers cited in `FINDINGS-BASELINE-2026-09-20.md`, and the
DELTA closeouts after that baseline need the SAME rubric. A rubric that lives
only in prose gets re-invented per session, each time slightly differently,
and then two passes disagree about what OPEN meant.

WHAT IT REFUSES TO DO
---------------------
It never reads prose to manufacture a status. Each join answers only the
question it can answer:

    STATUS_BOARD D-id row  -> deliverable status         DECISIVE
    ISSUES section         -> unresolved / resolved      DECISIVE
    TECH_DEBT section      -> implementation debt        DECISIVE (OPEN)
    own status line        -> only if work-shaped        DECISIVE
    PR state               -> landing evidence           NEVER decides
    live code citation     -> implementation reality     NEVER decides
    entries/ , LATEST_STATE-> provenance                 NEVER decides

Conflicting decisive evidence -> AMBIGUOUS, never averaged into certainty.
Absent decisive evidence -> AMBIGUOUS.

THREE MEASURED TRAPS THIS ENCODES
---------------------------------
1. `STATUS_BOARD.md` carries 28 distinct header schemas with `status` at index
   1..6 and ABSENT in two, so the column is located from each table's own
   header. Reading a fixed index returns prose as a status.
2. 303/306 entries carry a `Status:` line but 295 lead with an EPISTEMIC GRADE
   (`FINDING` 204, `RULING` 31, `CORRECTION` 11, ...), which answers *how well
   established is this claim*, not *is the work done*. Only a work-shaped
   leading token is status evidence.
3. Cross-supersession needs DIRECTIONAL phrasing. A bare `⊘`-near-the-E-id
   rule read `caveat (⊘ in E-FOO-1)` -- a sibling CITING this entry's own
   caveat -- as the sibling superseding it. The relation is directional and
   proximity inverted it.
"""

import collections
import json
import os
import pathlib
import re
import subprocess
import sys

WATERMARK = "2026-08-06"
EPI = ".claude/board/EPIPHANIES.md"
MARKER = ".claude/board/PROCESSED_THROUGH"

DATE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
EID = re.compile(r"\b(E-[A-Z0-9][A-Z0-9-]{3,})\b")
PR = re.compile(r"#(\d{3,5})\b")
CITE = re.compile(r"\b((?:crates|native|java|\.claude)/[A-Za-z0-9_./-]+\.(?:rs|md|py|toml|sh|yml))")
RESOLVED = re.compile(r"\b(RESOLVED|CLOSED|FIXED|LANDED|SHIPPED|DONE)\b")
DONE = re.compile(r"^\W*(shipped|done|complete|completed|landed|closed|merged|resolved|✅|✔)", re.I)
OPEN = re.compile(r"^\W*(queued|in progress|in-progress|in pr|blocked|open|todo|pending|next|planned|proposed|deferred|wip)", re.I)
OWN = re.compile(r"^\s*[>*\-\s]*\*{0,2}(?:Status|STATUS|Verdict|State)\*{0,2}\s*[:—-]\s*\*{0,2}(.{0,70})", re.M)
SUP_A = r"(?:supersedes|supersede|superseding|retires|retiring|replaces)\s+\S{0,40}?%s"
SUP_B = r"%s\S{0,8}[^\n]{0,80}?(?:is (?:now )?SUPERSEDED|SUPERSEDED by|is RETIRED|now RETIRED)"
WORK_DONE = ("SHIPPED", "CLOSED", "FIXED", "LANDED", "DONE", "RESOLVED", "COMPLETE")
WORK_OPEN = ("OPEN", "QUEUED", "BLOCKED", "PROPOSAL", "PENDING", "DEFERRED")


def did_pattern(root: str) -> "re.Pattern[str]":
    """The D-id pattern, READ from the generator that owns it.

    A second copy would agree until one was edited, which is exactly when
    nobody is comparing them -- the drift `plan_dids.py` refuses for the same
    reason and by the same mechanism.
    """
    src = pathlib.Path(root, ".claude/tools/supersession_index.py").read_text(errors="ignore")
    m = re.search(r"^DID\s*=\s*re\.compile\(r'(.*)'\)\s*$", src, re.M)
    if not m:
        raise SystemExit(
            "findings-baseline: the `DID = re.compile(r'...')` definition moved in "
            "supersession_index.py. Fix this extractor; do not copy the pattern here."
        )
    return re.compile(m.group(1))


def population(root: str, watermark: str = WATERMARK) -> tuple[list[dict], dict]:
    """Level-2 post-watermark headings carrying an E-id, plus the exclusions.

    A level-3 heading is a sub-section INSIDE an entry, so it must not
    terminate a body and is not itself an entry; a dated level-2 heading with
    no E-id is a date-group header. Both are counted, so the exclusion is
    visible rather than asserted.
    """
    lines = pathlib.Path(root, EPI).read_text(errors="ignore").split("\n")
    heads = [(i, len(m.group(1)), m.group(2))
             for i, l in enumerate(lines)
             if (m := re.match(r"^(#{2,3})\s+(.*)$", l))]
    out, skipped = [], {"level3": 0, "no_eid": 0}
    for idx, (i, lvl, text) in enumerate(heads):
        d = DATE.search(text)
        if not d or d.group(1) < watermark:
            continue
        if lvl == 3:
            skipped["level3"] += 1
            continue
        e = EID.search(text)
        if not e:
            skipped["no_eid"] += 1
            continue
        end = len(lines)
        for j, l2, _t in heads[idx + 1:]:
            if l2 <= 2:
                end = j
                break
        out.append({"eid": e.group(1), "date": d.group(1), "heading": text,
                    "line": i + 1, "body": "\n".join(lines[i:end])})
    return out, skipped


def status_board(root: str, did: "re.Pattern[str]") -> dict:
    """D-id -> status rows, with the status column located PER TABLE."""
    rows, hdr = collections.defaultdict(list), None
    for ln in pathlib.Path(root, ".claude/board/STATUS_BOARD.md").read_text(errors="ignore").split("\n"):
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.rstrip().rstrip("|").split("|")[1:]]
        low = [c.lower().strip("*").strip() for c in cells]
        if low and low[0] in ("d-id", "id", "deliverable", "row", "item"):
            hdr = low
            continue
        if re.match(r"^[-: |]+$", ln.strip()) or hdr is None or not cells:
            continue
        if not did.search(cells[0]) or "status" not in hdr:
            # A schema with NO status column (`| D-id | correction |`) carries
            # no status evidence. MEASURED: separating "on the board but
            # statusless" as its own provenance key reaches 0 of 306 entries,
            # so the distinction would be an inert branch -- and a guard that
            # never fires carries as much information as one that always does.
            continue
        i = hdr.index("status")
        if i >= len(cells):
            continue
        cell = cells[i]
        v = "done" if DONE.match(cell) else "open" if OPEN.match(cell) else "other"
        for d in did.findall(cells[0]):
            rows[d].append({"status": cell[:72], "verdict": v})
    return rows


def sections(path: pathlib.Path) -> dict:
    out, cur, buf = {}, None, []
    if not path.is_file():
        return out
    for ln in path.read_text(errors="ignore").split("\n"):
        if (m := re.match(r"^##\s+(.*)$", ln)):
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), [ln]
        elif cur:
            buf.append(ln)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def named_in(secs: dict, eid: str) -> tuple[bool, bool]:
    """(live, resolved). Resolution is asserted in the heading or the first two
    lines -- a section that merely DISCUSSES a resolution is not resolved."""
    live = resolved = False
    for head, body in secs.items():
        if eid not in body:
            continue
        top = head + "\n" + "\n".join(body.split("\n")[1:3])
        if RESOLVED.search(top):
            resolved = True
        else:
            live = True
    return live, resolved


def pr_states(root: str) -> dict:
    """Cached PR number -> state, if a cache was left beside the marker.

    PR state is landing evidence only, so its ABSENCE never changes a verdict;
    it only thins the implementation column. The tool therefore does not
    require network access to reproduce a classification.
    """
    p = pathlib.Path(root, ".claude/board/.pr-state-cache.json")
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def classify(root: str, watermark: str = WATERMARK) -> tuple[list[dict], dict]:
    did = did_pattern(root)
    entries, skipped = population(root, watermark)
    sb = status_board(root, did)
    iss = sections(pathlib.Path(root, ".claude/board/ISSUES.md"))
    td = sections(pathlib.Path(root, ".claude/board/TECH_DEBT.md"))
    prs = pr_states(root)

    board = {}
    for name, rel in (("status_board", ".claude/board/STATUS_BOARD.md"),
                      ("tech_debt", ".claude/board/TECH_DEBT.md"),
                      ("integration_plans", ".claude/board/INTEGRATION_PLANS.md"),
                      ("issues", ".claude/board/ISSUES.md"),
                      ("latest_state", ".claude/board/LATEST_STATE.md"),
                      ("supersession", ".claude/board/SUPERSESSION-INDEX.md")):
        f = pathlib.Path(root, rel)
        if f.is_file():
            board[name] = f.read_text(errors="ignore")
    ed = pathlib.Path(root, ".claude/board/entries")
    board["entries"] = "\n".join(
        (ed / f).read_text(errors="ignore")
        for f in sorted(os.listdir(ed)) if f.endswith(".md") and f != "README.md"
    ) if ed.is_dir() else ""

    alltext = "\n".join(e["body"] for e in entries)
    rows = []
    for e in entries:
        eid, body = e["eid"], e["body"]
        dids = sorted(set(did.findall(body)))
        prnums = sorted({int(p) for p in PR.findall(body)})
        cites = sorted(set(CITE.findall(body)))

        keys = {}
        if (direct := [k for k, t in board.items() if eid in t]):
            keys["eid_board"] = direct
        if (known := [d for d in dids if d in sb]):
            keys["did_statusboard"] = known
        if dids and not known:
            keys["did_unknown"] = dids
        if prnums:
            keys["pr"] = prnums
        live_cites = [c for c in cites if pathlib.Path(root, c).exists()]
        dead_cites = [c for c in cites if not pathlib.Path(root, c).exists()]
        if live_cites:
            keys["cite_live"] = live_cites
        if dead_cites:
            keys["cite_dead"] = dead_cites

        grade, own = "", ""
        if (m := OWN.search(body)):
            val = m.group(1).strip().lstrip("*").strip()
            t = re.match(r"[A-Za-z⊘-]+", val)
            grade = t.group(0).upper() if t else ""
            if val.startswith("⊘"):
                own = "sup"
            elif grade in WORK_DONE:
                own = "done"
            elif grade in WORK_OPEN:
                own = "open"

        self_sup = bool(re.search(r"\b(SUPERSEDED|RETIRED|WITHDRAWN|REJECTED-BY-FALSIFIER)\b",
                                  e["heading"])) or own == "sup"
        others = alltext.replace(body, "", 1)
        q = re.escape(eid)
        cross_sup = bool(re.search(SUP_A % q, others, re.I) or re.search(SUP_B % q, others))

        verdicts = {r["verdict"] for d in known for r in sb[d]}
        iss_live, iss_res = named_in(iss, eid)
        td_live, td_res = named_in(td, eid)
        open_ev, done_ev = [], []
        if "open" in verdicts:
            open_ev.append("STATUS_BOARD row not done")
        if "done" in verdicts:
            done_ev.append("STATUS_BOARD row done")
        if iss_live:
            open_ev.append("live ISSUES entry")
        if iss_res:
            done_ev.append("ISSUES entry resolved")
        if td_live:
            open_ev.append("live TECH_DEBT entry")
        if td_res:
            done_ev.append("TECH_DEBT entry resolved")
        if own == "done":
            done_ev.append(f"own status line: {grade}")
        if own == "open":
            open_ev.append(f"own status line: {grade}")

        merged = [p for p in prnums if prs.get(str(p)) == "merged"]
        unmerged = [p for p in prnums if prs.get(str(p)) == "closed"]
        impl = []
        if merged:
            impl.append("PR " + ", ".join(f"#{p}" for p in merged[:3])
                        + (f" +{len(merged) - 3}" if len(merged) > 3 else "") + " merged")
        if unmerged:
            impl.append(f"{len(unmerged)} referenced PR(s) closed unmerged")
        if not merged and not unmerged and prnums:
            impl.append(f"{len(prnums)} PR ref(s), state not cached")
        if live_cites:
            impl.append(f"{len(live_cites)} cited path(s) live")
        if dead_cites:
            impl.append(f"{len(dead_cites)} cited path(s) GONE")
        if not impl:
            impl.append("no implementation evidence")

        if self_sup or cross_sup:
            status = "SUPERSEDED"
            why = ["own heading/status ⊘ note" if self_sup
                   else "superseded by a sibling entry"]
        elif open_ev and done_ev:
            status = "AMBIGUOUS"
            why = ["CONFLICT: " + " + ".join(open_ev) + " vs " + " + ".join(done_ev)]
        elif open_ev:
            status, why = "OPEN", open_ev
        elif done_ev:
            status, why = "CLOSED", done_ev
        else:
            status = "AMBIGUOUS"
            why = ["no decisive status evidence" + ("" if keys else " and no join key at all")]

        rows.append({"eid": eid, "date": e["date"], "status": status, "grade": grade,
                     "impl": "; ".join(impl), "why": "; ".join(why),
                     "keys": sorted(keys), "joined": bool(keys)})

    usable = ("eid_board", "did_statusboard", "pr", "cite_live")
    stats = {
        "population": len(rows),
        "excluded": skipped,
        "verdicts": dict(collections.Counter(r["status"] for r in rows)),
        "grades": dict(collections.Counter(r["grade"] for r in rows if r["grade"])),
        "joined": sum(1 for r in rows if any(k in r["keys"] for k in usable)),
        "nojoin": sum(1 for r in rows if not r["joined"]),
        "key_reach": {k: sum(1 for r in rows if k in r["keys"])
                      for k in (*usable, "did_unknown", "cite_dead")},
    }
    amb = [r for r in rows if r["status"] == "AMBIGUOUS"]
    stats["ambiguous"] = {
        "conflict": sum(1 for r in amb if r["why"].startswith("CONFLICT")),
        "no_decisive_but_joined": sum(1 for r in amb
                                      if not r["why"].startswith("CONFLICT") and r["joined"]),
        "no_join": sum(1 for r in amb if not r["joined"]),
    }
    return rows, stats


def baseline_sha(root: str) -> str:
    p = pathlib.Path(root, MARKER)
    if p.is_file():
        for line in p.read_text(errors="ignore").splitlines():
            if line.startswith("PROCESSED_THROUGH_SHA="):
                return line.split("=", 1)[1].strip()
    return ""


def main(argv: list[str]) -> int:
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True).stdout.strip() or "."
    if "--self-test" in argv:
        return self_test(root)
    rows, stats = classify(root)
    if "--emit" in argv:
        out = argv[argv.index("--emit") + 1]
        pathlib.Path(out).write_text(render(rows, stats, baseline_sha(root)))
        print(f"wrote {out}: {len(rows)} rows")
        return 0
    print(json.dumps(stats, indent=1))
    return 0


def render(rows: list[dict], stats: dict, sha: str) -> str:
    """The committed baseline document. Every number is interpolated from
    `stats`, so the prose and the table cannot disagree -- and a hand-edit to
    'correct' a count is never the right move."""
    v, a, kr = stats["verdicts"], stats["ambiguous"], stats["key_reach"]
    n = stats["population"]
    L = [f"# Findings baseline — {sha[:12] or 'unpinned'} (`EPIPHANIES.md`, post-{WATERMARK} entries)", ""]
    w = L.append
    w("> **What this is.** The ONE historical catch-up over the findings that went")
    w("> into the `EPIPHANIES.md` monolith after the 2026-08-06 split watermark.")
    w("> It is a consolidated current-state checkpoint — a **k-frame**. After it,")
    w("> routine closeout is DELTA ONLY and never censuses the monolith again.")
    w("> Like `PLAN-INVENTORY-2026-09-07.md` it mints **no D-ids**, so")
    w("> `supersession_index.py` and `plan_dids.py` do not see it — by design.")
    w("> Regenerate: `python3 .claude/tools/findings_baseline.py --emit <this file>`.")
    w(">")
    w("> **The historical prose is FROZEN, not reconciled away.** `EPIPHANIES.md`")
    w("> is untouched: nothing was migrated, re-split, deleted or rewritten, and")
    w("> no entry files were created for these findings. Frozen means *not reread")
    w(f"> by routine closeout*; it does NOT mean adjudicated — {v.get('AMBIGUOUS', 0)} of {n} were not.")
    w(">")
    w(f"> **PROCESSED_THROUGH_SHA = `{sha}`** — every eligible finding visible")
    w("> through that source revision is consumed into this baseline. The marker")
    w("> names the CONSUMED INPUT, never this file's own commit: a commit cannot")
    w("> contain its own hash. Machine-readable: `.claude/board/PROCESSED_THROUGH`.")
    w("")
    w("---")
    w("")
    w("## 0. The numbers")
    w("")
    w("| | count |")
    w("|---|---|")
    w(f"| population (level-2 post-watermark entries with an E-id) | **{n}** |")
    for k in ("OPEN", "CLOSED", "SUPERSEDED", "AMBIGUOUS"):
        w(f"| {k} | {v.get(k, 0)} |")
    w("")
    ex = stats["excluded"]
    w(f"Excluded and counted so the exclusion is visible, not asserted: **{ex['no_eid']}**")
    w(f"level-2 bare date-group headers (no E-id) and **{ex['level3']}** level-3")
    w(f"sub-headings (sections *inside* an entry). {n} + {ex['no_eid']} + {ex['level3']} =")
    w(f"{n + ex['no_eid'] + ex['level3']} dated headings at or after the watermark.")
    w("")
    w("### Mechanical reachability — the union of join keys")
    w("")
    w("A first estimate put the ceiling at 198 by taking `306 − 108 without a D-id")
    w("or PR`. That was wrong: the E-id → board joins are **independent keys** and")
    w("most of them land inside that 108.")
    w("")
    w("| join key | entries | answers |")
    w("|---|---|---|")
    for k, q in (("eid_board", "named on a board surface — provenance"),
                 ("did_statusboard", "a referenced D-id has a STATUS_BOARD row — deliverable status"),
                 ("pr", "a PR is referenced — landing evidence ONLY"),
                 ("cite_live", "a cited path still exists — implementation reality"),
                 ("did_unknown", "referenced D-id has NO status-bearing board row — dangling"),
                 ("cite_dead", "cited path is GONE — stale citation")):
        w(f"| `{k}` | {kr.get(k, 0)} | {q} |")
    w("")
    w(f"**{stats['joined']}** entries carry ≥ 1 usable key; **{stats['nojoin']}** carry none and are")
    w(f"therefore automatically AMBIGUOUS. The other {v.get('AMBIGUOUS', 0) - a['no_join']} ambiguous rows are")
    w("ambiguous for a different and more interesting reason — §1.")
    w("")
    w("## 1. Why AMBIGUOUS is the largest bucket")
    w("")
    gr = stats["grades"]
    tot = sum(gr.values())
    workish = sum(c for g, c in gr.items() if g in WORK_DONE + WORK_OPEN)
    w(f"**{tot} of the {n} entries carry their own `Status:` line, and {tot - workish} of those lead")
    w("with an EPISTEMIC GRADE rather than a work status:**")
    w("")
    w("| leading token | entries |")
    w("|---|---|")
    for k, c in sorted(gr.items(), key=lambda kv: -kv[1])[:10]:
        w(f"| `{k}` | {c} |")
    w("")
    w("`FINDING`, `RULING`, `CORRECTION`, `MEASURED` answer *how well established")
    w(f"is this claim*. They do not answer *is the work done*. Only **{workish}** entries")
    w("lead with a work-shaped token.")
    w("")
    w("That is the substantive result: **post-watermark `EPIPHANIES.md` was being")
    w("used as a findings log, not a deliverable tracker.** For most rows")
    w("OPEN/CLOSED is the wrong axis — the live question is *is this still true?*,")
    w("which no join answers mechanically. Reading their prose to manufacture a")
    w("status is what this pass was told not to do, so they stay AMBIGUOUS with")
    w("their grade recorded.")
    w("")
    w("Not a comparable number: `PLAN-INVENTORY-2026-09-07.md` reached 40/211")
    w("ambiguous **with a human read of every status line in context**, and records")
    w("that naive substring matching produced ≥ 6 false positives in its corpus.")
    w("This pass is mechanical-only by instruction; the larger residue is the price")
    w("of that, not a worse measurement.")
    w("")
    w("## 2. How to read a verdict")
    w("")
    w("Vocabulary reused from `PLAN-INVENTORY`; nothing new minted. Each join")
    w("answers only the question it can answer:")
    w("")
    w("| evidence | role | may decide status? |")
    w("|---|---|---|")
    w("| STATUS_BOARD D-id row | deliverable status | **yes** |")
    w("| ISSUES section | unresolved / resolved | **yes** |")
    w("| TECH_DEBT section | implementation debt | **yes** (OPEN) |")
    w("| the entry's own status line | only if its leading token is work-shaped | **yes** |")
    w("| INTEGRATION_PLANS | integration ownership | no — context |")
    w("| PR state | landing evidence | **no — MERGED ≠ CLOSED** |")
    w("| live code citation | implementation reality | no |")
    w("| `entries/`, `LATEST_STATE` mention | provenance | no |")
    w("")
    w(f"Conflicting decisive evidence ⇒ **AMBIGUOUS** ({a['conflict']} rows), never averaged")
    w(f"into certainty. Absent decisive evidence ⇒ **AMBIGUOUS** ({a['no_decisive_but_joined']} joined rows +")
    w(f"{a['no_join']} unjoinable). The *implementation* column carries landing and")
    w("code-liveness facts precisely so they cannot be mistaken for closure.")
    w("")
    w("Three traps the tool encodes, each measured: STATUS_BOARD's status column")
    w("is **per-table** (28 schemas, index 1..6, absent in two); a `Status:` line's")
    w("**leading token only** is read; and cross-supersession needs **directional**")
    w("phrasing, because a bare `⊘`-proximity rule read `caveat (⊘ in E-FOO-1)` —")
    w("a sibling citing this entry's caveat — as the sibling superseding it.")
    w("")
    w("## 3. The rows")
    w("")

    def outcome(r):
        if r["status"] != "AMBIGUOUS":
            return r["why"]
        if r["why"].startswith("CONFLICT"):
            return r["why"]
        if not r["joined"]:
            return f"no join key at all; graded {r['grade'] or '—'}"
        return (f"graded {r['grade'] or '—'} — an epistemic grade, not a work status; "
                "no deliverable attached")

    for st in ("OPEN", "CLOSED", "SUPERSEDED", "AMBIGUOUS"):
        sel = sorted((r for r in rows if r["status"] == st), key=lambda r: (r["date"], r["eid"]))
        w(f"### {st} ({len(sel)})")
        w("")
        w("| source id | status | implementation | outcome/open point |")
        w("|---|---|---|---|")
        for r in sel:
            w(f"| `{r['eid']}` | {st} | {r['impl']} | {outcome(r)} |")
        w("")
    w("---")
    w("")
    w("## 4. What this baseline does NOT claim")
    w("")
    w(f"- It does not claim the {v.get('AMBIGUOUS', 0)} AMBIGUOUS rows are resolved, wrong, or safe")
    w("  to delete. They are unadjudicated, and that is recorded, not hidden.")
    w("- It does not claim a merged PR closed the finding attached to it.")
    w("- It does not claim the frozen prose was reviewed. **FROZEN ≠ RECONCILED.**")
    w("- It is not a licence to start another archaeology pass over the ambiguous")
    w("  rows. If one matters later it resurfaces as live work and enters the")
    w("  transient tier like anything else.")
    w("")
    w("## 5. Steady state after this checkpoint")
    w("")
    w("```")
    w("new work → .claude/board/entries/ → reconcile against current state")
    w("         → OPEN | CLOSED | SUPERSEDED | AMBIGUOUS")
    w("         → rare Eureka promotion to EPIPHANIES.md (must cite its entry)")
    w("         → advance PROCESSED_THROUGH_SHA to the captured source head")
    w("```")
    w("")
    w("MIRROR dies. Corrections die. Failed probes normally die. Git keeps the")
    w("route. Only surviving state crosses the checkpoint.")
    w("")
    return "\n".join(L)


def self_test(root: str) -> int:
    """Falsifiers for the three measured traps, each proven to FIRE."""
    ok = True
    did = did_pattern(root)

    # (a) the per-table status parser must not read a status from a table that
    #     has no status column -- the schema that returns prose as a status.
    import tempfile
    d = tempfile.mkdtemp(prefix="findings-baseline-selftest-")
    b = pathlib.Path(d, ".claude/board")
    b.mkdir(parents=True)
    pathlib.Path(d, ".claude/tools").mkdir(parents=True)
    (pathlib.Path(root, ".claude/tools/supersession_index.py")).read_text()
    import shutil
    shutil.copy(pathlib.Path(root, ".claude/tools/supersession_index.py"),
                pathlib.Path(d, ".claude/tools/supersession_index.py"))
    (b / "STATUS_BOARD.md").write_text(
        "| D-id | correction |\n|---|---|\n| D-AAA | Shipped-looking prose |\n\n"
        "| D-id | scope | status |\n|---|---|---|\n| D-BBB | s | Shipped |\n")
    sb = status_board(d, did)
    if "D-AAA" in sb:
        print("  FAILED: read a status from a table with no status column")
        ok = False
    elif sb.get("D-BBB", [{}])[0].get("verdict") != "done":
        print(f"  FAILED: missed a real status ({sb})")
        ok = False
    else:
        print("  per-table status column       : prose ignored, real status read")

    # (b) an epistemic grade must NOT become a work status.
    for grade, expect_own in (("FINDING", ""), ("RULING", ""), ("CORRECTION", ""),
                              ("SHIPPED", "done"), ("PROPOSAL", "open")):
        body = f"## 2026-09-01 E-X-1\n\n**Status:** {grade} (measured, somewhere)\n"
        m = OWN.search(body)
        val = m.group(1).strip().lstrip("*").strip()
        t = re.match(r"[A-Za-z⊘-]+", val)
        g = t.group(0).upper() if t else ""
        own = ("done" if g in WORK_DONE else "open" if g in WORK_OPEN else "")
        if own != expect_own:
            print(f"  FAILED: grade {grade} -> own={own!r}, expected {expect_own!r}")
            ok = False
    print("  epistemic grade vs work status : FINDING/RULING/CORRECTION decide nothing")

    # (c) cross-supersession must be DIRECTIONAL: a citation of this entry's
    #     own caveat is not the sibling superseding it.
    eid = "E-FOO-1"
    q = re.escape(eid)
    citing = "caveat (⊘ in `E-FOO-1`)"
    real = "this ⊘ supersedes `E-FOO-1` outright"
    if re.search(SUP_A % q, citing, re.I) or re.search(SUP_B % q, citing):
        print("  FAILED: a citation was read as a supersession")
        ok = False
    elif not re.search(SUP_A % q, real, re.I):
        print("  FAILED: a real supersession was missed")
        ok = False
    else:
        print("  directional supersession      : citation ignored, real one caught")

    print("findings-baseline --self-test " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
