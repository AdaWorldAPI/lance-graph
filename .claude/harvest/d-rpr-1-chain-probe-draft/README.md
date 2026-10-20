# D-RPR-1 chain probe: draft, banked as differential-test evidence (2026-10-20)

**Status:** PROBE DRAFT, NOT MERGED, NOT RUN AS D-RPR-1. `chain-probe.patch` applies cleanly to `main` at `4dbd085` and, after #1623, at `8b570ed` (`git apply --check`). It is kept here as a patch, not in the crate, for two reasons: D-RPR-1 is "not authorized; waits on A1 and an operator go" (`recursive-perspective-rebinding-v1.md` §10), and one of its six tests is red (below).

Written by a second session from the instruction "use the existing substrate", before #1622 merged. It is an independent attempt at the chain D-RPR-1 describes, so it serves as a differential check of that design.

## What the patch adds

- `crates/mork-pathmap-probe/src/chain.rs` (171 lines): `ask` asks reachability on a prepared frame. While the answer has escapes, it brings those identities into the focus through `rebind` and asks again. A later question passes an earlier answer as its seeds. Slots are evicted only when they hold no identity an earlier answer still needs.
- `crates/mork-pathmap-probe/tests/chain.rs` (278 lines, 6 tests) and one `pub mod chain;` line in `src/lib.rs`.
- It calls only existing `rmap` functions: `Csr::prepare`, `rebind`, `Csr::reach`, `ScopedReach::global`, `Focus::replace`, `Tile::from_csr`, `TileExec::reach`, `ThoughtDelta`, `Witness`. It adds probe-local glue types (`Frame`, `Answer`, `Rebuild`, `ChainRefusal`), which the D-RPR-1 text ("adds no new types") does not allow for. That is a deviation to resolve before it could land.

## Result, run on `main` at `4dbd085` and re-run at `8b570ed` (release, AVX-512 host; same result both times)

```
cargo test --release --test chain   (in crates/mork-pathmap-probe)
5 passed; 1 failed
```

| test | result |
|---|---|
| `incremental_frames_equal_full_preparation_and_reuse_rows`: rebind and full prepare give identical answers, focus, edge set, escape set and tile; rows both reused and newly resolved | ok |
| `a_changed_world_refuses_the_frame`: bumped facts witness gives `Stale(FactsChanged)` | ok |
| `a_focus_too_small_for_the_answer_is_refused`: `FocusExhausted{needed: 1, free: 0}`; silent twin with one evictable slot answers `{0,1,2}` | ok |
| `an_earlier_answer_survives_the_later_steps`: the first question re-asked at the end admits nothing, same answer | ok |
| `thoughts_run_over_the_frame_the_chain_produced`: 64 retraction thoughts equal an in-focus oracle over edited facts; `ThoughtMismatch` for another id; shared tile unchanged | ok |
| `every_step_of_the_chain_equals_the_global_oracle` | **FAILED** at the anti-vacuity assertion `answers[0].admitted.len() > 1` |

In the failing test, the three per-step equalities with the global scan oracle pass, and the total admitted slots are above zero. Only the claim that the first question had to be re-asked fails. The cause is the fixture: the first focus is `discover` over all three predicates, 256 slots from the seed, and that already contains the 3-hop reach over predicate 0. So step 1 never escapes and tests nothing. The finding is recorded in the plan as sub-case D-RPR-1a's anti-vacuity rule. The fix is a narrower first focus, and nothing in the composition changes.

## Compared with the D-RPR-1 design (plan §8)

| D-RPR-1 element | draft |
|---|---|
| reuses `Focus`, `discover`, `Csr`, `rebind`, `ThoughtDelta`, `Witness`, `join.rs` | all except `join.rs`; adds glue types (above) |
| KJV corpus | not used; synthetic `rmap::graph(65 536, 3, 2, 512, 0.9)`, the plan's labelled fallback |
| escapes of angle k seed angle k+1 (H1) | escapes complete angle k (re-ask until global); the answer seeds angle k+1. H1 as worded is not tested |
| arms (c) full prepare, (d) rebind | both, pinned equal |
| arms (a) MORK / PathMap, (b) direct | not in the chain; they exist elsewhere in the crate |
| measures 1–10 | correctness only: reuse > 0 (3), 64 thoughts (7), changed premise (8), parity vs scan (10). No timings, no bags |
| refusals | `FactsChanged`, `IncompleteFocus` (loop condition), `ThoughtMismatch`; adds `FocusExhausted`. `OrderingMismatch`, `ScopeChanged`, `IndexResealed`, k = 256 not exercised; the dependency-cycle refusal stays OPEN |
| disable runs | not run |

## To reproduce

```
git apply .claude/harvest/d-rpr-1-chain-probe-draft/chain-probe.patch
cd crates/mork-pathmap-probe && cargo test --release --test chain
```
