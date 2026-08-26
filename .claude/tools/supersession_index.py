#!/usr/bin/env python3
"""Generate the supersession index from the repo itself.

WHY GENERATED: a hand-maintained table of "what superseded what" goes stale the
moment a rename lands, and a stale supersession table is worse than none -- it
authorises work against a symbol that is already retired. This reads
COMPONENT-MAP.md (the verdicts), crates/ (where each symbol is actually live),
and plans/ (who still names it), so the table cannot disagree with the repo.

    python3 .claude/tools/supersession_index.py > .claude/board/SUPERSESSION-INDEX.md

Deliberately NOT used as a signal: git mtime on plans. 2026-07-24 is a bulk
import (one merge, 2,718 files) carrying 144 of ~180 plans, so git dates the
import, not the work.
"""
import re, glob, os, sys, subprocess

ROOT = subprocess.run(["git","rev-parse","--show-toplevel"],capture_output=True,text=True).stdout.strip()
os.chdir(ROOT)
CM = ".claude/v3/COMPONENT-MAP.md"
VERD = re.compile(r'\b(RETIRE[A-Za-z-]*|REPURPOSE[A-Za-z()\s]*|BLOCKED)\b')
SUCC = re.compile(r'→\s*`?([A-Za-z_][A-Za-z0-9_:]*)`?')

def count(pat, *dirs):
    n = 0
    for d in dirs:
        for f in glob.glob(f"{d}/**/*", recursive=True):
            if not os.path.isfile(f) or not f.endswith(('.rs','.md')): continue
            try: n += 1 if re.search(rf'\b{re.escape(pat)}\b', open(f,encoding='utf-8',errors='ignore').read()) else 0
            except Exception: pass
    return n

syms = {}
for line in open(CM, encoding='utf-8'):
    if not line.startswith('|'): continue
    c = [x.strip() for x in line.split('|')[1:]]
    if len(c) < 2: continue
    m = VERD.search(c[1])
    if not m: continue
    note = c[2] if len(c) > 2 else ''
    s2 = SUCC.search(note)
    for s in re.findall(r'`([A-Za-z_][A-Za-z0-9_:.]{3,})`', c[0]):
        s = s.split('::')[-1].replace('.rs','')
        if len(s) > 3:
            syms[s] = (m.group(1).strip(), s2.group(1) if s2 else '', note)

plans = {p: open(p,encoding='utf-8',errors='ignore').read() for p in sorted(glob.glob(".claude/plans/*.md"))}
board = "\n".join(open(f,encoding='utf-8',errors='ignore').read()
                  for f in glob.glob(".claude/board/entries/*.md")+[".claude/board/EPIPHANIES.md"])
AWARE = re.compile(r'v3/COMPONENT-MAP|\.claude/v3|RETIRE|REPURPOSE|D-PERT')
# Anchored to the status's LEADING token on purpose. An unanchored search matched
# a word that did not predicate the plan -- "ACTIVE ... Phase A/B COMPLETE",
# "PROPOSAL. v1 SHIPPED in PR #420" -- routing live plans to ARCHIVE?. 3/3 of the
# first ARCHIVE? batch were that error (2026-08-26 routing pass).
SHIPPED = re.compile(r'^\W*(SHIPPED|COMPLETE[D]?|SUPERSEDED|DONE|CLOSED|LANDED)\b', re.I)
# `(?!\s*legend)` so "Status legend:** OK SHIPPED (verified...)" is not read as a status.
STATUS = re.compile(r'[Ss]tatus(?!\s*legend):?\*{0,2}\s*([^\n|]{0,52})')
DID = re.compile(r'\b(D-[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)*)\b')

rows = []
for p, txt in plans.items():
    named = sorted(s for s in syms if re.search(rf'\b{re.escape(s)}\b', txt))
    if not named or AWARE.search(txt): continue
    dids = set(DID.findall(txt))
    cited = sum(1 for d in dids if d in board)
    st = (STATUS.search(txt).group(1).strip() if STATUS.search(txt) else '')
    retires = [s for s in named if syms[s][0].startswith('RETIRE')]
    route = ("ARCHIVE?" if SHIPPED.search(st) else
             "RESCOPE"  if retires else
             "READ")
    rows.append((route, len(named), os.path.basename(p)[:-3], named, st, cited, len(dids)))

out = sys.stdout.write
out("# Supersession index — GENERATED, do not edit\n\n")
out("> Regenerate: `python3 .claude/tools/supersession_index.py > .claude/board/SUPERSESSION-INDEX.md`\n>\n")
out("> This table is derived from `COMPONENT-MAP.md` + `crates/` + `plans/` on every run,\n")
out("> so it cannot disagree with the repo. A hand-kept supersession table goes stale at\n")
out("> the next rename, and a stale one is worse than none: it authorises work against a\n")
out("> symbol that is already retired.\n>\n")
out("> The commentary below is generated too: every measurement in it is interpolated from\n")
out("> the same values as the tables, so prose and table cannot disagree. Methods are\n")
out("> literal; measurements never are.\n\n")

# ---- generated commentary: measurements interpolated, methods literal ------
def blind_for(sym): return sum(1 for r in rows if sym in r[3])
def plans_for(sym):
    pat = re.compile(rf'\b{re.escape(sym)}\b')
    return sum(1 for t in plans.values() if pat.search(t))
stat = {s: (count(s,'crates'), plans_for(s), blind_for(s)) for s in syms}
top      = max(stat, key=lambda s: stat[s][0])                       # most live code
allblind = [s for s in stat if stat[s][1] and stat[s][2] == stat[s][1]]
inverted = [s for s in stat if stat[s][1] > stat[s][0] > 0 and syms[s][1]]  # successor required

out("## What this table says\n\n")
tc, tp, tb = stat[top]
out(f"**`{top}` is the shape of the problem.** Marked {syms[top][0]}, and simultaneously the\n"
    f"most-referenced symbol here: **{tc} crate files, {tp} plans, {tb} of them blind.**\n"
    f"That is a programme, not a cleanup.\n\n")
if allblind:
    s0 = max(allblind, key=lambda s: stat[s][1])
    out(f"**`{s0}` is the sharpest case: {stat[s0][1]} plans name it and *every one* is blind.**\n"
        f"Its COMPONENT-MAP note reads: {syms[s0][2][:120]}\n\n")
used = {s0} if allblind else set()
if [s for s in inverted if s not in used]:
    s1 = max((s for s in inverted if s not in used), key=lambda s: stat[s][1] - stat[s][0])
    succ = syms[s1][1]
    out(f"**`{s1}`{f' → `{succ}`' if succ else ''} gives the rule that needs no map at all:**\n"
        f"{stat[s1][0]} crate files against {stat[s1][1]} plans. The code moved; the plans did not.\n"
        f"**Plan-mentions exceeding crate-mentions is a staleness signal on its own.**\n\n")

out("### The limit of the mechanical route\n\n"
    "`RESCOPE` fires on \"names a RETIRE symbol\" alone. That is a *reason to look*, not a\n"
    "verdict: a plan mentioning the symbol once in background and a plan built around\n"
    "retiring it land in the same bucket. Separating them needs a read; this table's job\n"
    "is to make that read finite and ordered, not to replace it.\n\n")
out("### What cannot be found this way\n\n"
    "A rename that left no trace. `Blumenstrauss -> cognitive-shader-driver SoA` has zero\n"
    "hits in `.claude/`, `crates/`, plans, or board, so nothing connects prior reasoning to\n"
    "the current name and no script recovers what was never written down. An alias enters\n"
    "this table only if someone records it in COMPONENT-MAP -- the method's single\n"
    "dependency and single failure mode.\n\n")
out("### Deliberately not a signal\n\n"
    "`git log` dates on plans. 2026-07-24 is a bulk import -- one merge, 2,718 files -- so\n"
    "git dates the import, not the work. Routing uses self-declared status and board\n"
    "coverage instead.\n\n")

out("## Table 1 — ruled symbols: verdict, successor, and where each side actually lives\n\n")
out("| symbol | verdict | successor | live in crates | named in plans | blind plans |\n")
out("|---|---|---|---|---|---|\n")
for s in sorted(syms, key=lambda s: (syms[s][0], s)):
    v, succ, _ = syms[s]
    blind = sum(1 for r in rows if s in r[3])
    pat = re.compile(rf'\b{re.escape(s)}\b')
    in_plans = sum(1 for t in plans.values() if pat.search(t))
    succ_cell = f"`{succ}`" if succ else "—"
    out(f"| `{s}` | {v} | {succ_cell} | {count(s,'crates')} | {in_plans} | {blind} |\n")

out(f"\n## Table 2 — plans naming a ruled symbol without citing the ruling ({len(rows)})\n\n")
out("Route is **mechanical triage, not a verdict**: `ARCHIVE?` = the plan's own status says\n")
out("it shipped; `RESCOPE` = it targets a symbol marked RETIRE; `READ` = neither signal fires\n")
out("and a human read decides. Board coverage counts this plan's D-ids cited on the board.\n\n")
out("`ARCHIVE?` reads the status's LEADING token only. The first batch it produced was\n")
out("**3/3 false positives** -- an unanchored match on a word that did not predicate the\n")
out("plan (`ACTIVE ... Phase A/B COMPLETE`, `PROPOSAL. v1 SHIPPED in PR #420`, and a\n")
out("`Status legend:` defining the tick mark). All three were live, and archiving them\n")
out("would have retired work in flight. A route here is a prompt to read the plan, never\n")
out("a licence to act on it.\n\n")
out("| route | plan | ruled symbols named | self-declared status | board coverage |\n")
out("|---|---|---|---|---|\n")
for route, n, stem, named, st, cited, tot in sorted(rows, key=lambda r: (r[0], -r[1])):
    ss = ", ".join(f"`{s}`" for s in named[:4]) + (" …" if len(named) > 4 else "")
    out(f"| **{route}** | `{stem}` | {ss} | {st[:44] or '—'} | {cited}/{tot} |\n")

for r in ("ARCHIVE?","RESCOPE","READ"):
    out(f"\n- **{r}**: {sum(1 for x in rows if x[0]==r)}")
out(f"\n- ruled symbols tracked: {len(syms)}\n")
