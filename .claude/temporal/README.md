# `.claude/temporal/` — the temporal / delta / alpha reference

Created 2026-09-06. **Findings are recorded 1:1 as they were measured and
reported in session.** Nothing here is a plan unless the file says PLAN.
Every claim carries its `file:line` or its command; where something is
UNVERIFIED it says so and stays unverified.

## Why this folder exists

The 2026-09-06 session ran an archaeology + migration audit over the alpha
channel, the rung ladder, the octopus/SPOG addressing, lance 11's delta API,
and the surrounding temporal machinery. The result was large, cross-repo, and
almost entirely about things that ALREADY EXIST. Without a written home it
would have to be re-derived, which is exactly the tax this workspace's own
rules exist to prevent.

## Doc map

| file | what it answers |
|---|---|
| `01-delta-api-lance11.md` | Is delta read or write? What is it conditional on? What does MemWAL actually do? |
| `02-addressing-regimes.md` | The four addressing regimes, the census, and what makes a stable-row-id change non-breaking |
| `03-alpha-channel-state.md` | What the alpha overlay is, what it costs, who consumes it, what breaks if the resident form changes |
| `04-reusable-patterns.md` | pothole/`residue_band`, `revision.rs`, stockfish hindsight, surrealdb kv-lance, ternlog amortization, the convergent law |
| `05-deletion-and-tombstones.md` | Can we still go back after a delete? What would destroy that? Is a tombstone needed? |
| `06-callers-and-dormancy.md` | Who calls what; the dormancy census; the migration ledger; the `seed.clone()` ruling |
| `07-deepnsm-v2-old-vs-new.md` | v1 vs v2, brutally — including two corrections to this session's own claims |
| `08-memwal-vs-batchwriter.md` | FOLD verdict, the comparison table, and the property that would be silently lost |
| `09-plan.md` | PLAN. Staged, probe-gated, with non-goals and kill conditions |

## The one-line result of the whole audit

The stack's problem is not missing mechanism. Nearly every piece is built,
tested, doctrinally justified, and connected to nothing. The hypothesis that
migration "preserved the pieces while losing the wiring" was FALSIFIED: the
wiring was never there to lose.

## Standing rules this folder inherits

- Append-only. Correct in place with a dated `⊘` note; never delete a finding.
- A doc-comment claim is not a behaviour. If prose says X and no test exercises
  X, the prose is labelled *claimed, unverified*.
- Absence is a finding. "Searched X, found nothing" is recorded, not omitted.
