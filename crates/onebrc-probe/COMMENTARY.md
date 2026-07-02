# COMMENTARY — one session's interpretation of the 1BRC probe

> Read `FINDINGS.md` FIRST — it is the agnostic record (environment,
> methods, numbers, invariants-with-code) and it is deliberately free
> of what follows. THIS file is the prime of the 2026-07-02 session
> that ran the probe: its architecture readings, the operator rulings
> it executed, and its doctrine mappings. A later session analyzing
> the findings from another angle should treat everything below as
> ONE interpretation to test, not as ground truth. Board homes of
> these readings: `EPIPHANIES.md` entries `E-1BRC-ADDRESSING-1`,
> `E-1BRC-KANBAN-UPDATE-1`, `E-1BRC-OWNER-GRANULARITY-1`,
> `E-1BRC-ORCHESTRATION-SWEETSPOT-1`, `E-1BRC-BATCH-PIPELINE-1`,
> `E-1BRC-GRIDLAKE-SWEETSPOT-1`.

## The arc (operator-driven, five follow-ups)

1. "Compare morton and the kanban vs without" → lanes F/R/G.
2. "One ractor mailbox per SoA" → lane G topology corrected (each
   owner's actor State IS its own table; "sharding one SoA" was an
   ownership inversion in prose, and full-size per-owner tables made
   the fine end unrunnable).
3. "The 65536 mailboxes had no Orchestration — find the sweet spot" →
   lane H (lazy activation + ahead-firing batch writer).
4. The batch-pipeline spec (64K upfront, aligned indices, codebook,
   double-cast, flush cache) → lane I.
5. "8 or 64 lanes? gridlake 64×64 = 4096×BF16?" → lane J knob matrix.

## Readings (interpretation — test me)

- **Route-and-write beats look-up-and-compare ~3×** (R vs C), and the
  semantic (Morton tile) address costs ~10% over plain radix at this
  LOW cardinality (F vs R). The prefix-local tile payoff is a
  high-cardinality claim the 400-key corpus cannot test.
- **The witness is free; the boundary is not.** Journaling
  (KanbanMove appends, version ticks) never showed up in any
  measurement (lane E: ~66 µs/card; sinks: invisible). What costs is
  the actor-model BOUNDARY: the one-time corpus copy, task
  oversubscription, and message fragmentation.
- **Producers must never address fine-grained owners directly.** The
  20× cliff at 65536 flat owners is fan-out, not ownership. Two cures
  measured: lazy activation + ahead-firing batching (lane H, 23×
  recovery), or whole-table batch double-cast (lanes I/J, messages ∝
  batches — independent of BOTH occupancy and address-space size).
- **Ownership does not need actors when it can be an index-aligned
  guarantee table.** Lane I/J's `row_owner[i] == i` binding +
  write-on-behalf at the sink preserved single-writer semantics with
  zero per-owner messaging. The standing 64K actor registry, by
  contrast, halves steady-state throughput by pure residency (knob-
  isolated in t7) — actors are for CONTENTION, not for address space.
- **Cache-match the batch unit: the 64×64 gridlake (4096 cells) is
  the measured sweet spot.** 46.2 Mrows/s — equal to the best
  streamed topology, above the orchestrator, with the strongest
  witness (double-WAL both ends). The 64K-cell batch unit loses ~15%
  purely to L2 pressure (~2.5 MB/worker working set). The literal
  4096×BF16=16 KB plane pair is ndarray #227's proven `VDPBF16PS`
  tier (~448 Mrows/s single-thread in its own probe) — the natural
  continuation once tile-GEMM enters the batch path.
- **Sink lanes scale with per-batch APPLY work, never with data**:
  1 suffices at O(400-dirty-rows) apply; 8 free; 64 over-lanes.
- **SIMD provenance note:** the delimiter scan (lane B) routes through
  `ndarray::simd::U8x32::cmpeq_mask` with the stride walk on
  `ndarray::simd::array_chunks` (the NON-overlapping walker;
  `array_windows` is the overlapping GEMM-style sibling and is
  deliberately not used — delimiter scanning never re-reads bytes).
  `simd_soa.rs`'s `SoaBytes` (Arc-backed typed iteration) is NOT yet
  wired — it is the natural carrier for vectorizing the sinks' dirty
  sweeps and the batch tables themselves; open follow-up.

## The composed recipe (this session's synthesis)

64×64 gridlake batch SoA + codebook CAM addressing (mint once, direct
index after) + 1–8 sink lane pairs + whole-table `Arc` double-cast +
refcount-gated flush cache; ownership as the index-aligned guarantee
table; the router/lazy-activation tier (lane H) only when fine-grained
ownership must live as actors; BF16 planes per ndarray #227 when
tile-GEMM lands.

## Flagged uncertainty (kept honest)

- All numbers: 4-core container, 400-key cardinality, 10M rows,
  no mmap, scalar parse. Run-to-run drift up to ~10% between rounds
  (same-session references quoted in FINDINGS wherever compared).
- The residency attribution at t6 was CONJECTURE until t7's registry
  knob isolated it — it is now a FINDING, but the *mechanism*
  decomposition (scheduler vs memory-footprint vs allocator) was NOT
  isolated further.
- High-cardinality behavior (where prefix-local tiles and per-tile
  bucketed sweeps should earn their keep) is entirely unmeasured.
- 100M container-scale runs, SWAR parse, mmap: priced and parked in
  `README.md` §1 / §5.3.

## Suggested lab sweeps for a next session

The 8 presets (`src/presets.rs`, feature `presets`) share one
signature and one parity harness. Obvious axes: preset × workers
(1/2/4) × corpus cardinality (regenerate with more syllable
combinations) × rows (10M/100M) × batch_rows (lane J's `1<<16` default
vs smaller) × morsel size. The archival recipe convention (README §2)
makes every cell reproducible; publish results as a new dated section
in FINDINGS (append, don't rewrite history).
