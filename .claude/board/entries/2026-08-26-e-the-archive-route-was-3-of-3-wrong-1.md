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
