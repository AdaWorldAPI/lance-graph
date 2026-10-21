# D-TGB pre-registration (D-TGB-0)

Written before any thought-git code exists. Every later item is checked
against the oracle in `src/oracle.rs` on the fixtures in `src/fixtures.rs`.
Its falsifier and its disable are fixed here. A disable is a deliberate break
of the implementation; the falsifier must turn red under it. If an item
changes one of these, the change is recorded here with its reason, never made
silently.

## What D-TGB-0 itself pins (tests in this crate)

| rule | test | disable that turns it red (verified 2026-10-20) |
|---|---|---|
| only `Commit` lands | `only_commit_lands` | land every outcome |
| agreement with a given is not stored | `agreement_with_a_given_is_not_stored` | store agreeing writes |
| a different value for a given is refused and recorded | `a_different_value_for_a_given_is_refused_and_recorded` | let the offer overwrite the given |
| a divergence in one version stores nothing and is recorded | `a_divergence_in_one_version_stores_nothing_and_is_recorded` | last writer wins |
| an unwritten lane reads absent, not zero | `an_unwritten_lane_reads_absent_not_zero` | read a missing cell as 0 |
| squashing an unremarkable stack changes nothing | `squashing_an_unremarkable_stack_changes_nothing` | — (characterises the naive squash) |
| squashing a divergent pair hides the divergence | `squashing_a_divergent_pair_hides_the_divergence` | last writer wins (also red) |

Signals the plan relies on, each with a can-fire and a can-stay-silent case
on the real `lance-graph-contract` code (`tests/signals.rs`):

| signal | fires on | silent on |
|---|---|---|
| regression (`AlphaOverlay::regressions`, `visits > 1`) | `REGRESSION_SCANPATH` (row 3 returned to) | `SMOOTH_SCANPATH` |
| cold/hot surprise (`expected.xor(attended)`) | `SURPRISE_EXPECTED` vs `SURPRISE_ATTENDED` | expected equals a real overlay's attended mask |
| kanban route per outcome | Commit, Plan, Prune, pre-Rubicon veto, each a legal path through `advance` / `revise` / `veto` | — |
| MUL gate | `Block` at `Evaluation` → `Prune`; `Hold` → stays | `Flow` → `Commit` |
| Gadamer, replayed material | `Echo`, `NoIncrease`, routes to `Plan` even with a `Necessary` counterfactual | a new root plus `Necessary` reaches `Commit` |
| Gadamer, second landing | the same encounter against the first result has no new root | the first landing is `IncreaseEligible` |
| Gadamer, preserved contradiction | `ContradictionPreserved`, `Suspend`, the cycle holds, the tension is kept | — |

## Later items: falsifier, disable, fixture

| item | falsifier | disable | fixture |
|---|---|---|---|
| D-TGB-1 slot pool (built, `src/slots.rs`) | the 64th branch picks a candidate (FIFO oldest, LeastUsed idlest); a candidate with a live parent is squashed into it, one with a live child absorbs that child; only a chain-free candidate is evicted; the new thought's parent is never removed; stale handles are refused; a freed slot is reused without pressure | ignore the generation; LeastUsed keyed on age only; evict instead of squashing a chain; let the new thought's parent be the candidate | `tests/slots.rs` |
| D-TGB-2 lane labels | after `Commit` no live edge carries the landed slot; an edge from a recycled slot's old generation is refused | read W of a sealed row as meaningful (proposed here; the plan names none) | `rubicon_outcomes` (the Commit thought) |
| D-TGB-3 landing | equals `next_version` on every fixture; an empty cycle writes nothing; two thoughts in one cycle give one version; a re-submitted batch reconciles; a missing metadata row is detected at replay | one batch per thought (second `Commit` in a cycle must hit `HashConflict`) | all fixtures, landed together and one per version |
| D-TGB-3 write into 59..63 | landing the same thought twice leaves 59..63 unchanged; a thought supported only by replayed material lands with certification unchanged; `Block` leaves the version count unchanged; `pearl::hydrate` never changes certification, `pearl::revise` never changes topology | let `revise` lower a certification; let `hydrate` move certification | the Gadamer cases in `tests/signals.rs`, as edges (needs `lance-graph-planner`, added when D-TGB-3 starts) |
| D-TGB-4 receipt (built, `src/receipt.rs`) | replaying a landed or discarded thought from its receipt reproduces its writes cell for cell; missing or changed seed, changed config, changed source, stale/ABA slot, unsealed or torn metadata row all refuse | drop the seeds from the receipt (verified 2026-10-21, with nine further disables in `tests/receipt.rs`) | `rubicon_outcomes` cells (Plan and Prune thoughts are the discarded ones) |
| D-TGB-5 resident folds | `checkout`, `log`, `diff`, `blame` equal the oracle; `diff` across two versions equals Lance's `delta()` | read Lance on the hot path | `consecutive_versions_replace_without_divergence` shape |
| D-TGB-6 MemWAL arm | same answers as D-TGB-3 behind the switch | enable below the measured crossover | as D-TGB-3 |
| D-TGB-7 API + driver | two drivers making the same decisions give byte-identical cells, receipts, metadata rows | let the implementation read a `KanbanColumn` to decide a squash | all fixtures |
| D-TGB-8 abstraction | rule answers equal a direct scan of the metadata table; recorded intents replayed as verbs reproduce the run | answer one rule with a loop that skips the source relation | metadata rows produced by D-TGB-3 on all fixtures |
| D-TGB-9 planning vocabulary | planning answers equal a hand-written Quack `Query`; no allocation proportional to the number of cards | — (written when D-TGB-9 starts) | — |
| D-TGB-10 folded rules | the folded answer equals datafrog's bag of tuples | drop one presence plane from the address mask | as D-TGB-8 |

## Still OPEN (not decided here)

- Which anomaly signals stop a squash, and at what threshold. The fixtures
  make each signal fire; whether a single regression (`visits == 2`) is
  enough is not decided.
- Which revision kinds besides a contradiction count as an anomaly.
- What "satisfactory" means for a write (the `jc` statistic and threshold).
- The kanban card granularity (mailbox card or one per move).
