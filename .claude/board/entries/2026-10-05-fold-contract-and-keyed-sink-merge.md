# 2026-10-05 — Fold contract (WHERE × WHEN) and the first rendezvous primitive: keyed partial-sink merge

**Status:** WORKING-MODEL (the contract) · TEST-PINNED (Step E, `crates/lance-graph-mask-risc/tests/keyed_merge.rs`) · OPEN (items at the end)
**Supersedes:** the wording of the fold-contract reconnaissance reported in-session (2026-10-05) on four points, listed under "Corrections". The reconnaissance itself is VERIFIED-IN-CODE and is not repeated here; its first finding is the gap Step E closes.

## The contract

A fold has two degrees of freedom: **WHERE** it lands (destination) and **WHEN** it merges or finalizes (rendezvous). Planning has a third: **WHETHER** it deserves to exist.

| verb | meaning | layer (WORKING-MODEL, operator 2026-10-05) |
|---|---|---|
| PREPARE / PLAN | may run speculatively and asynchronously, overlapping ingest, write or earlier execution; bound to schema + coordinate identity + expected version; validated before use; amortizable or bypassable | planner (JIT / loco) |
| BIND | makes a prepared route valid for one world/version | planner / ABI |
| RESOLVE | a row resolves its destination through functional references; no population state | R2IL |
| ACCUMULATE | fold the row's contribution into the destination's state | R2IL |
| RETAIN | keep RAW, unfinalized fold state | loco lifetime policy |
| RENDEZVOUS | the earliest point where independent fold states are needed together — a dependency fence, not an instruction | loco scheduling |
| MERGE | same fold kind: `S × S → S` | R2IL data instruction |
| COMPOSE | different retained fold states become inputs of a later computation (FoldSet) | loco program |
| FINALIZE | one-way: identity is lost (NULL from count, `sum/count`, normalize) | R2IL |
| MATERIALIZE | only on demand | boundary / adapter |

R2IL says WHAT operation; ogar-loco says WHEN / with whom / in which recipe; Quack says WHICH data computation is demanded; the ABI says IN WHICH world/version. Hard Rust kernels sit below R2IL and are added only when a measured fusion of R2IL steps beats the sequence — never because one application recipe is hot.

### Corrections to the reconnaissance wording

1. **ACCUMULATE / multi-terminal.** A shared traversal is defined by the compatible PHYSICAL source: source/coordinate world, pinned version, row extent, tile traversal. Each fold consumer independently owns its filter, its route / group-key resolution, its fold algebra and its sink. (Was: "share filter, route and traversal" — too narrow; the tile loop already has every slot live at terminal time, so per-consumer filters and keys are mechanically sound.)
2. **RETAIN.** Fold-state identity is separate from placement. A retained state must prove what it means — fold algebra, source coordinate/version identity, grouping/local coordinate identity, filter/presence semantics, exactness/overflow contract — but its eventual consumer/destination may stay symbolic until a later rendezvous. Compute now, retain raw, bind placement later, where semantics allow.
3. **MERGE.** The general contract is `merge : S × S → S`, associative wherever execution relies on arbitrary partitioning. Commutativity is a property of a particular fold, not of the contract. Count, Sum, Min, Max and `_sym` Sum are commutative (under their stated overflow/bound contracts), so Step E tests arbitrary order. `KeyRunCarry` is NOT thereby non-mergeable: it may be an ordered-segment fold (associative, not commutative, with boundary metadata). OPEN.
4. **MERGE ≠ COMPOSE.** `COUNT Berlin ⊕ COUNT Hamburg → COUNT Germany` is MERGE. License + Revenue + Risk + Headcount → annual report is COMPOSE. Heterogeneous folds are never forced into one merge algebra.

### Planning rules (operator, 2026-10-05)

- **Never wait longer to discover how to avoid work than it would take to do the work.** Use a plan only when `plan_cost + optimized_execution < simple_execution` over the expected lifecycle; `effective_plan_cost ≈ max(0, planning − overlapped_write) / expected_reuse`.
- **Plan may run ahead of authority; a prepared plan is not a valid execution world.** At seal: compatible → instant handoff; stale → rebind or discard.
- **Fire-and-forget planning, not fire-and-forget semantics.** Execution does not owe the planner a wait; the simple path runs now and the plan serves later batches.
- **JIT composes, it does not generate code.** It chooses existing instructions, order, fusion, where to retain, where a rendezvous is needed. A recurring sequence becomes a named loco program; it is not recompiled into Rust.

### The DO arm (operator, 2026-10-05) — the write half of the same algebra

The IR needs a `DO` arm symmetric to READ, whose natural form is a fold: source contribution → resolve target → fold into mutation state per target → retained action state → Rubicon/guards → seal → ActionHandler → Revision. **The DO arm does not execute one effect per source row. It folds source contributions by resolved effect destination into the minimal deterministic mutation state, and only that state may cross the action boundary.** Each DO class carries identity / accumulate / merge / conflict / finalize-to-effect (ADD+ADD idempotent; ADD+REMOVE and SET E5 + SET F3 are conflicts, policy-defined). `DO` means "construct the mutation state", never "write now" — that keeps replay, dry-run, before/after, dedup and batch amortization. NOT BUILT.

## Step E — keyed partial-sink merge (mask-risc)

**What changed:** `GroupSumI32`, `GroupSumViaI32` and `GroupReduce { Count, MinI32, MaxI32, SumSymI32 }` (Lane / Via / Pair keys) are admitted on a partial extent. Each extent gets a FRESH, re-seeded `Out::I64`; an edge word is clipped in a register (the arms previously read the unclipped mask — dormant while partial extents were refused). Partials combine with `Terminal::merge_group_sink`, which applies `GroupFold::merge` — the merge law lives on the IR type that already knows the fold kind. No new grouping engine, sink, address, terminal or type.

**Overflow algebra — each fold's existing semantics preserved, no conflict found (so no STOP):**

| fold | kernel accumulation | merge | exact when |
|---|---|---|---|
| Count | `wrapping_add(1)` | `wrapping_add` | always (a count ≤ its rows) |
| GroupSumI32 / Via | `wrapping_add` | `wrapping_add` | total rows ≤ `MASKED_SUM_I32_MAX_ROWS` (2^32); validate bounds the plane, disjoint extents of it cannot exceed it |
| MinI32 / MaxI32 | `min` / `max` | `min` / `max` (seed = lattice identity) | always |
| SumSymI32 | first row replaces ⊥, then `wrapping_add` | `⊥⊕x = x`, `x⊕⊥ = x`, `⊥⊕⊥ = ⊥`, else `wrapping_add` (`TD-SYM-SUM-MERGE-IS-NOT-ADDITION-1`) | TOTAL rows across partials ≤ `GROUP_SUM_SYM_MAX_ROWS` |

Report's `FoldState::merge` (`wrapping_add` / `min` / `max`) agrees with every row. ndarray's `checked_merge` (PowerSums) is a different state type and is untouched.

**Errors:** all refusals of these terminals are preflight — `precheck` (slot ceiling, extent range, extent support) and `validate` (lanes, out shape, row bounds) run before the sink is seeded. None of them fails during traversal (the only traversal-time error, `LaneNotOrdered`, belongs to `CountKeyRunsU32`, still refused on a partial extent). `execute_into`'s contract is unchanged; a partial sink is the caller's per-extent workspace, K-sized, never population-sized.

**Gates (all green; mask-risc suite, clippy `-D warnings`, fmt):** split composition equals the whole and the whole equals the row oracle, for 14 terminal × key combinations at two tile widths, over one partition, two-way cuts in a word / on word edges / on the tile edge, empty extents and ≥ 12 random uneven partitions, in four orders plus a balanced-tree grouping. Anti-vacuity asserted (a group live in two partials, a group live in exactly one, an empty group, an empty extent, a cut splitting a live word). `_sym` can-fire cases (⊥⊕x, x⊕⊥, ⊥⊕⊥, real zero ≠ ⊥) by value and through execution. `finalize(merge(raw))` correct; merging coalesced finalized MIN slots shown WRONG (0 for a true 5). Monoid laws on extremes; the `_sym` counterexample outside the row bound pinned.

**Disable runs, each red then restored:** `_sym` merged with plain `+`; MIN merged as MAX; empty `_sym` read as 0; GroupReduce edge unclipped; GroupSumI32 edge unclipped.

**What this makes true:** `population A → raw grouped state A ─┐ merge raw → finalize once` / `population B → raw grouped state B ─┘` — without source-row materialization and without A and B executing as one pass. First substrate implementation of RENDEZVOUS (same-type).

## OPEN

- **`TD-KEYED-SINK-MERGE-IDENTITY-1`.** Equal length is not semantic compatibility. A bare `&[i64]` cannot prove that it is raw, which fold produced it, its coordinate space or version, its filter, its destination universe, that the extents are disjoint, or that the combined row bound holds. Merging finalized slots is wrong but not REFUSED. Today these are caller preconditions documented on `merge_group_sink`. They close with the retained FoldState identity contract (correction 2), not with metadata bolted onto Step E.
- `KeyRunCarry` as an ordered-segment fold.
- Multi-terminal traversal sharing (one physical pass → N independent consumers).
- **A. Destination binding / coordinate resolution** — a frontend / binder concern (merged #1331: Report's `CoordSpec` cannot bind a functional reference, a composed route, or several coordinates to one destination ordinal). Its normal path lowers to the EXISTING `Local` / `Via` keyed fold; it is NOT a missing ndarray / fold primitive. A query-local interner is only a fallback for a genuinely unbindable destination universe. A destination ordinal is not a semantic identity by itself: **an ordinal has meaning only inside its destination-space / codebook identity** (WORKING-MODEL; the identity carrier is not built — see `TD-KEYED-SINK-MERGE-IDENTITY-1`).
- **B. Functional route composition beyond depth 2** — a substrate / R2IL question. Address chain (`lanes[k2][lanes[k1][fk[i]]]`) vs a precomposed resident lane over the intermediate table: unmeasured.
- The DO arm and its per-class conflict algebra.
- **rs-graph-llm as a semantic dependency graph over retained folds** (operator WORKING-MODEL, 2026-10-05; NOT BUILT). Quack is the deterministic IR for every stateful READ/DO computation; rs-graph-llm owns only dependencies, retained fold identity (FoldRef: algebra, coordinate/version, filter/provenance, raw state, optional destination) and the human/LLM boundaries. Consequences named: `Context` carries scalars, decisions, handles, FoldRefs and capabilities, never populations; an edge is a dependency / rendezvous, not row transport; a task may complete with an unfinalized FoldRef; LLM output is structured evidence folded (support / contradiction / provenance / frequency / confidence) before it becomes state; `NextAction::GoTo` gives way to declared readiness that loco schedules; agents contribute DO intents, and the DO fold — not the agent — produces ActionIntent. Loco needs only fold metadata (mergeable, ordered, finalized, dependencies, world/version, destination bound). Consistent with decision 3 of `2026-10-05-report-pair-key-and-stack-recon.md` (graph-flow emits facts; canonical Kanban + Revision owns lifecycle).
- Stale doc: `population-law-crosscheck-v1.md:39` says mask-risc has no moments fold; `GroupPowerSumsI32` / `GroupCrossPowerSumsI32` exist.
