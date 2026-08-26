## 2026-08-26 — E-THE-ARCHIVE-ROUTE-WAS-3-OF-3-WRONG-1

**Status:** FINDING (measured) · **Confidence:** HIGH

First routing pass over `SUPERSESSION-INDEX.md`, taking the `ARCHIVE?` batch
first because it was the smallest and looked the most decided. **All three were
false positives, and every one of them is live work.** Archiving on the route
would have retired a plan mid-flight.

The classifier searched the whole status string for a shipped-word:

```python
SHIPPED = re.compile(r'\b(SHIPPED|COMPLETE[D]?|SUPERSEDED|DONE|CLOSED|LANDED)\b')
route   = "ARCHIVE?" if SHIPPED.search(st) else ...
```

An unanchored search has no notion of **what** shipped. The three hits:

| plan | matched on | what the word actually predicated |
|---|---|---|
| `cognitive-substrate-convergence-v2` | `COMPLETE` | a *phase* — the status opens `ACTIVE`, and the very next word is `(pending merges)` |
| `odoo-savant-reasoners-v2` | `SHIPPED` | its **predecessor** — `PROPOSAL. v1 SHIPPED in PR #420`; this plan exists *because* v1's shape was judged wrong |
| `unified-soa-rubikon-integration-v1` | `SHIPPED` | a **legend** — `Status legend:` defining what ✅ means, captured as if it were a status |

Verified against `STATUS_BOARD.md` rather than the headers, anchoring the join on
each row's own leading cell (a loose `grep D-CSV-11` returns the D-CSV-14 row,
which *mentions* it — same bad-join class as the earlier filename-stem inflation):

- `cognitive-substrate-convergence-v2` — 12 Shipped, 4 **In PR**, 1 Queued. The
  four cite **#388/#390**; the repo is at #1044. Not archivable — its board rows
  are *stale*, which is a different defect and a different fix.
- `odoo-savant-reasoners-v2` — all five of its own D-ids **Queued**. Never
  started.
- `unified-soa-rubikon-integration-v1` — Queued, one OBE, one In PR.

All three name a RETIRE symbol, so all three route **RESCOPE**. `ARCHIVE?` → 0,
`RESCOPE` 47 → 50, `READ` unchanged at 12.

**The fix is an anchor, not a longer word list.** A stricter alternation cannot
help: any pattern matching `COMPLETE` fires on case 1, any matching `SHIPPED`
fires on case 2. Only the status's *leading* token predicates the plan.
`STATUS` additionally now refuses `Status legend:`.

Both halves falsified rather than assumed — it fires on `SHIPPED in #420`,
`DONE (2026-06-01)`, `✅ SHIPPED`, `Superseded by v3`; it is silent on all three
false positives. **The rule is not dead:** `odoo-source-extraction-v1` carries a
leading-shipped status tree-wide, so a category reading 0 inside the table is a
measurement, not a broken guard. (My first can-it-fire run reported a spurious
`False` because I fed `SHIPPED` a raw line; the generator only ever feeds it the
captured group. The test was wrong, not the code.)

**Two limits found while measuring, both still open:**

1. **`crates` hit counts include tombstone comments.** `CollapseGateEmission`
   shows 5 crate files and is genuinely **gone** — all six occurrences are
   comments recording its removal. The column measures textual mention, so a
   dead symbol reads as live.
2. **The stale-status class is real and unmeasured.** Four D-CSV rows still say
   `In PR (#388/#390)` hundreds of PRs later. Nothing reconciles a board row
   against its PR's actual outcome. This is the "claim-vs-reality drift" check
   already named as the highest-value unrun scriptable class.

**The generalizable rule:** a mechanical route is a *prompt to read*, never a
licence to act. The value of taking the smallest, most-decided-looking batch
first was that it was small enough to check every member — and checking every
member is what found that the category was empty.

---

**⊕ CORRECTION (same day, codex P2 on #1046 — verified, and it was right).**
The first fix skipped `Status legend:` **by name**. That is whack-a-mole, not a
fix: `STATUS` still searched the whole document and took the first hit anywhere,
so for `unified-soa-rubikon-integration-v1` the next match became the section
heading `## 6. Honest status (no overclaim)` at line 181 — and the table duly
reported **`(no overclaim)`** as that plan's self-declared status. Codex also
named the escalation I had not: any mid-document heading whose text after
`status` leads with a shipped token (`## 9. Status: SHIPPED items`) recreates
the false `ARCHIVE?` outright.

The class, not the instance: **a status is metadata, not prose.** Extraction now
runs over `plan_head()` — the preamble, extended through any *leading*
status-titled section (`## §0 — Status` is metadata; line 181 is not). A strict
preamble cut was measured first and was too tight: it lost
`self-reasoning-substrate-v1`, whose real status sits in exactly such a `## §0`
section. A 40-line window was also measured — its only marginal catch was itself
prose, so the principled rule beat the magic constant on the evidence.

Nine junk statuses disappear, not one: alongside `(no overclaim)` the table had
been reporting `` ` (:90), `QueryReference{server_id,ref_versi ``, `") vs `,
`, **code refs**, and where it `, and `. ` as statuses. Two more rows were
near-misses recovered by allowing a parenthetical (`**Status (§0):** PROPOSED`).
Twelve rows now read `—`; an honest absence beats a fabricated status.

Routes are unchanged (`ARCHIVE?` 0 · `RESCOPE` 50 · `READ` 12) and generation is
still idempotent. The falsifier suite gained codex's escalation case and stays
two-sided: fires on `SHIPPED in #420` / `DONE (2026-06-01)` / `SUPERSEDED by v3`,
silent on all five negatives.

**What this adds to the entry above:** the original finding was that a route is a
prompt to read, never a licence to act. This correction is the same error one
level up — I fixed the *instance* I had measured rather than the *class* it
belonged to, and shipped a table with a fabricated value in it. Naming the
failing input is not the same as naming the failure.

---

**⊕ THE GENERATED TABLE WENT STALE WITHIN THE HOUR (same day, post-merge).**
#1046 merged at 09:06 alongside #1045 and #1047. #1047 added
`grounding-descent-cognitive-maslow-v1`, which names `GateDecision` and
`ThinkingStyle` without citing the ruling — a 63rd blind plan. So the index
committed minutes earlier no longer reproduced: `RESCOPE` 50 → 51,
`GateDecision` 19 → 20 plans, `ThinkingStyle` 23 → 24.

Nothing was wrong with the generator. The table is generated from
`.claude/plans/` + `crates/` + `COMPONENT-MAP.md`, and **any PR that adds a
plan makes it stale** — generation was manual, so the freshness lasted exactly
as long as nobody else merged.

This is the same defect the index was built to solve, one level up. The
original argument was that *a hand-kept supersession table goes stale at the
next rename, and a stale one authorises work against a retired symbol.*
Generating it removed the transcription error and left the staleness — because
"generated" only means current at the moment someone last ran it. **A generated
artifact with no staleness gate is a hand-maintained artifact with extra
steps.**

Closed with `.github/workflows/supersession-index.yml`: regenerate, `diff`, fail
with the regenerate command on drift. It watches the generator's **inputs**
(`.claude/plans/**`, `COMPONENT-MAP.md`) and not merely the generator, since
inputs are what actually made it stale. Falsified both ways before committing —
appending one comment to the index makes it exit 1; regenerating makes it pass.

**Standing limit, unfixed:** the gate proves the table is *current*, never that
a route is *right*. The ARCHIVE? batch was current and 3/3 wrong.
