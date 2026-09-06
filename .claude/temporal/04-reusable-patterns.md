# 04 — reusable patterns found in the audit

Five patterns worth carrying forward, each measured, each with its provenance.

---

## A. The epistemic POTHOLE — `residue_band`, and the chain it starts

**"Pothole" is exact shipped vocabulary, not a metaphor.**
`OGAR/crates/ogar-dismech/src/lib.rs:417`:

```rust
pub const fn residue_band(candidates: u32, diffuse_floor: u32) -> u8 {
    if candidates == 0 { 0 }                     // Surface        <- POTHOLE
    else if candidates == 1 { 2 }                // Relation
    else if candidates < diffuse_floor { 4 }     // Counterfactual
    else { 1 }                                   // Association    <- POTHOLE
}
```

A cardinality ladder with four outcomes, **two of which are potholes**, and
they fail for opposite reasons (`lib.rs:125-142`):

- **0 candidates** — nothing addressable. *"The gap names its own cell, which
  is the reach-out hook: a failure that says what is missing points at what to
  fetch, instead of silently deriving nothing."*
- **`>= diffuse_floor`** — the term does not discriminate. Hub-shaped: *"a term
  with too many attachments is barred rather than guessed at."* `diffuse_floor`
  is calibrated per corpus, **never a shipped constant**.

The middle two are where a claim can be made: exactly one candidate is a
relation; a bounded fork is the branching material a counterfactual needs.

### The chain is a documented HANDOFF

Section title, verbatim: **"Pothole → rung degradation → revision"**. And the
boundary is drawn sharply: *"Revision is downstream and elsewhere: this band
produces the degraded rung; NARS revision consumes it."*

The degradation is principled: no branching material means no counterfactual
claim. MedCare's board carries the operator correction that keeps it honest —
a degraded rung must never read as worthless, because **the pothole is the
highest-value output of a search, since it is where the work is.** Same entry
carries the transferable rule: **a counter is not a population.** Any pipeline
that increments a variable where it should append a row has decided that class
is not worth addressing, and the decision is invisible until someone asks for
the file.

### Two things this settles

**Time lives in the pothole.** `lance-graph-contract/src/band_reading.rs:71`:
*"Temporal carries NO field here — time is implicit in the epistemic pothole
(Lance versions); explicit temporal lives only in its three sanctioned homes
(Rubikon revision window, `CausalEdgeV3`'s TE byte, a future attention-v3
reading). `EdgeProvenance` is layout epoch, never time."*

**The rung stack is free.** *"Projecting rungs as a stack of alpha layers costs
zero bits here: each rung is its own table over the same address space, and the
3-bit sample is what sits inside one layer."* Which validates the rung×tenant
mask stack — with the trap above (never map the band onto the rung ladder).

**Dormancy:** `residue_band` has NO consumer outside `ogar-dismech` in OGAR,
lance-graph or MedCare-rs. Band and revision are both built; the handoff is
unwired.

---

## B. `revision.rs` — the epistemic-pothole revision, already modelled

`lance-graph-contract/src/revision.rs`, 665 lines, LANDED 2026-08-29.
Placed by its own doc beside `temporal.rs` and `counterfactual.rs`:

- temporal remembers the awareness horizon and durable arrival;
- counterfactual explores sealed hypothetical timelines;
- **revision records whether an encounter changed the interpretive horizon or
  merely returned inherited assumptions.**

Explicitly: *"no production write capability — the output types stop before
actual-world mutation by design."*

### The mask trait

`EvidenceMask`: `empty`, `is_empty`, `union`, `intersection`, `difference`,
`is_subset_of`, `intersects`. Implemented for `u64` and fixed `[u64; N]`
arrays. Its doc: *"The canonical mask type can implement this trait at
integration time."* Live consumer: `lance-graph-planner/src/dismech_candidates.rs`
(`EvidenceItem`, `apply`, `evaluate`, `is_informative` — a candidate set is a
bitmask over chains).

### The types map onto the operator's four questions

| the question | the shipped type |
|---|---|
| what did the old version know vs what I know | `InterpretiveHorizon` before/after + `BasisView` bounded ancestry (caller-supplied, no graph walk) |
| what did the difference cause | `CounterfactualVerdict::{Necessary, Dispensable, NotRun}` — remove it, does structure collapse; `NotRun` is the deliberate DEFAULT, *"the reason this type exists"* |
| what indirect intermediate unknowns | `unresolved_tension`, accumulated by UNION and **never cleared** |
| the epistemic pothole revision | `RevisionKind` (9) -> `EvidentialEffect` (3) |

`RevisionKind`: `IndependentConfirmation`, `Reinterpretation`,
`HorizonExpansion`, `HorizonFusion`, `AssumptionExposed`,
`ContradictionPreserved`, `Suspended`, `Echo`, `ClosedCycle`.
`EvidentialEffect`: `IncreaseEligible`, `NoIncrease`, `Suspend`.

### Hindsight blindness IS its central invariant

*"Echoes and closed cycles remain observable history but receive zero
additional evidential weight."*

The separator between a preserved contradiction and an earned fusion is ONE
condition — `has_new_root`, a genuinely new independent root not in ancestry.
Without it: `ContradictionPreserved` -> `Suspend`. With it: `HorizonFusion` ->
`IncreaseEligible`. The anti-laundering invariant in executable form:

> *"no amount of re-reading inherited material can produce a synthesis, because
> re-reading yields no new independent root. Fusion of two horizons requires
> that at least one of them actually touched the world."*

And it is **Gadamer, not Hegel: the tension is PRESERVED, never dissolved.** A
synthesis that resolved its antithesis is the false synthesis the module
refuses.

---

## C. stockfish-rs hindsight — the type-enforced causal window

Two probes, and the first failed instructively.

**D-SF-RUNG-1 (`examples/rung_hindsight.rs`) — INCONCLUSIVE by its own header.**
It built the hindsight oracle from `sign(evaluate(net, final_position))`. NNUE
is not mate-aware, so its own anchor game (Morphy's Opera Game,
`Qxb8+ Nxb8 Rd8#`) reads as the mating side being materially down. The deeper
fault, verbatim: *"reading `evaluate` at ply N to judge a claim ABOUT ply N is
not independent evidence — it is the same measurement instrument re-applied,
wearing a hindsight costume."*

**D-SF-HINDSIGHT-1 (`examples/hindsight_stream.rs`) — the fix, two changes.**
1. **Independent oracle** — the game's own PGN `Result` header. *"A won game is
   a won game regardless of whether the mating tactic makes the second-to-last
   static eval look upside down."*
2. **Structural gate** — each game streamed as a
   `lance_graph_contract::temporal_pov` version stream where **ply index `v` IS
   the Lance version**. A present reader at ply v is `TemporalPov::at(v,
   STRICT_RUNG)` admitting exactly `[0, v]`; every multi-ply read filters
   candidates through `pov.admits(u)`, so *"the reader physically cannot
   construct a window that reaches into `v+1..`."*

`STRICT_RUNG = 0` because `EpistemicMode::for_rung` (`temporal.rs:67-73`)
partitions rungs `0..=4` as `Strict`. The file is honest about scope:
`TemporalPov::admits` implements only the version-range HALF of admission, not
the per-row `Spoiler`/`Anachronistic` classification.

Its own claim, modest and correct: *"Mechanism vs framing: the GATE mechanism
below is ordinary statistics. The FRAMING is what's new — the causal-window
admission is enforced by a typed range check (`VersionRange::contains`), not by
manual discipline ('only look at earlier plies, I promise'). A read that would
violate the window is excluded by construction"* — and it asserts zero
violations are ever admitted.

### `TemporalPov` is the ONE non-dormant piece

`lance-graph-contract/src/temporal_pov.rs`, 314 lines, **zero-dep**, with two
independent real consumers:

| consumer | proven on |
|---|---|
| `deepnsm-v2` (`wave.rs`, `lib.rs`, `bible_wave.rs`) | the whole KJV, 23,145 verses; the read that retired the fixed ±5 window |
| `stockfish-rs` (`hindsight_stream.rs`, `expert_iteration_stream.rs`) | real lichess PGN games with an independent outcome label |

**Being zero-dep is the decisive property:** the version-range admission gate
needs NO lance dependency, because it is a typed range check over integers.

The gate exists at two layers and only the lower one is used:

| layer | status |
|---|---|
| `contract::temporal_pov::TemporalPov::admits(u)` — version-range half | **LIVE**, two repos, two corpora |
| `planner::rung_horizon::claim_admitted` — full `TemporalStatus` classification then claim | **DORMANT**, no production caller |

---

## D. surrealdb kv-lance — MVCC hand-rolled, and what became native

`AdaWorldAPI/surrealdb` @ `8f5adb23`, `surrealdb/core/src/kvs/lance/`
(3,955 lines: `mod.rs` · `tests.rs` 1,662 · `timeline.rs` 267 · `schema.rs` 237
· `tx_buffer.rs` · `background_optimizer.rs` · `cnf.rs`). Header pins it:
*"all Lance 7.0.0 surface — the mandatory pin."*

`KvSchema` (`schema.rs:59-63`), five columns, deliberately byte-minimal:

| column | type | job |
|---|---|---|
| `key` | Binary | the KV key |
| `val` | Binary | the value |
| **`version`** | UInt64 | the MVCC version AT WRITE TIME |
| **`tombstone`** | Boolean | a delete recorded as a ROW, never a removal |
| **`seq`** | UInt64 | ordering WITHIN a version |

That is a hand-rolled append-only MVCC delta log on a flat table.

| hand-rolled on lance 7 | lance 11 native |
|---|---|
| `version` column | `_row_created_at_version` / `_row_last_updated_at_version` |
| `tombstone` column | `DatasetDelta::get_deleted_row_ids` |
| scan-and-compare per version | `get_inserted_rows` / `get_updated_rows` |
| `seq` column | **nothing** |

**The hand-rolled columns are the exact shape of what became native. The design
was not wrong, it was early.**

### The same pattern at three layers, one missing feature

| layer | how it answers "what changed" | cost |
|---|---|---|
| surrealdb `Timeline` (lance 7) | `view_at(v).scan()` per version, compare | a full SoA per version |
| lance-graph `VersionedGraph::diff` (TODAY, `versioned.rs:538`) | `read_all_batches` BOTH versions, HashSet-diff by `node_id` | two full materializations |
| alpha overlay (TODAY) | a 512-byte row per touched address | 512 B × touched |
| **lance 11** | `Dataset::delta(v1,v2).get_inserted_rows()` | the delta |

### `seq` is the finding: three designs needed it, lance provides it nowhere

1. surrealdb kv-lance — a `seq: u64` column.
2. lance-graph `persist_sink` — `order_cycle_stably` by `stream_position`
   BEFORE the single WAL append, so reads never sort.
3. alpha — `seq: u32` inside `AlphaStamp`.

A delta reports WHICH rows changed between versions; it never reports the ORDER
they were touched within one. So the irreducible application payload,
confirmed by three independent discoveries rather than by argument:

```
seq     order within a version    (no native equivalent, anywhere)
visits  attention regressions     (alpha only)
```

**Six bytes.** Everything else alpha materializes is recoverable from versions.

### Two decisions worth taking

1. **Byte-minimal, no second clock.** The schema keeps only `version`;
   wall-clock comes opportunistically from Lance's own `Version` records, and
   a consumer needing a guaranteed wall clock writes its OWN `timestamp_micros`
   column as `VersionedGraph` does — additive, never a default.
2. **Read-only enforced by the TYPE SYSTEM.** `TimelineView` has no
   `set`/`del`/`commit`: *"it owns no write path, so 'SurrealDB never mutates
   the SoA' is enforced by the type system, not by convention."*

**Dormancy:** `timeline.rs:2` — `#![allow(dead_code)] // unwired, test-covered
read surface (future kanban/replay consumer)`, plus seven more in `mod.rs` and
two in `tx_buffer.rs`. Built, 1,662 lines of tests, connected to nothing.

---

## E. ternlog chaining amortization

**The law** (`gemm-ternlog-mask-consolidation-v1.md:144`): a 3-input `IMM` is a
whole truth table, so any Boolean over three masks is ONE pass, and an n-input
predicate is **⌈(n−1)/2⌉ chained passes**. That ratio IS the amortization.

**Shipped surface:** `ternlog<const IMM: i32>` on every backend (avx512, avx2,
neon, wasm, scalar) + `mask_ternlog` / `mask_ternlog_assign` over `&[u64]`.
Named tables: `AND3 0x80`, `AND2 0xC0`, `MAJ3 0xE8`, `XOR3 0x96`, `OR3 0xFE`,
`AND2_ANDNOT 0x40`, `AND_ANDNOT2 0x10`, `OR2_AND 0xA8`.

**Measured caveat, must not be dropped:** *"the ternlog wire is one of three
mask passes and cannot account for 5×"* — the bulk was `eq_u32_strided_to_mask`
at `stride_bytes == 4`. So **ternlog chaining is a correctness/shape win first,
a speed win only where a measurement says so.**

**Consumer census:** `lance-graph-java`'s lgj-abi consumes it correctly by tier
(`kernels.rs:100` re-exports `ndarray::simd::ternlog`; `exports.rs` names
`kernels::ternlog::AND3`, never `ndarray::simd`). **lance-graph and OGAR have
NO caller** (post-teardown survey row B1).

### The blocker, and it ties E to the mask synergy

`AlphaMask` is `Box<[u64]>` + `len` — byte-for-byte what `mask_ternlog_assign`
wants. But **`lance-graph-contract` is zero-dependency BY DESIGN**, and its
Cargo.toml says why: as a workspace member, ANY path dep (even optional) is
resolved at workspace-load time, so a missing sibling breaks every cargo
invocation in CI (learned 2026-07-07, an optional `../../../OGAR` dep killed
the whole PR pipeline).

Therefore: `AlphaMask`'s and/or/xor/and_not are hand-rolled SCALAR word loops
(the private `zip`), while a shipped all-backend SIMD ternlog is unreachable
one crate away. **Three mask algebras over the same u64 words** —
`AlphaMask` (bit ops), `EvidenceMask` (set algebra), `mask_ternlog` (any
3-input truth table) — the first two shape-compatible with the third's input.

A stacked temporal awareness is exactly that: one mask per version, combined
pairwise. `known_then AND NOT known_now` is ONE ternlog pass instead of three.
**But the cached, ternlog-accelerated stack must live one crate OUT of the
contract**, consuming its mask words.

**Instrument, not re-run here:** `ndarray/examples/ternlog_amortization_probe.rs`
(D-GTM-0n) measures the curve with four arms (T0 materialized `Vec<u32>`, T1
`mask_and_assign`, T3 `mask_ternlog_assign::<AND3>`, T5 sorted intersection), a
correctness gate across arms, and bandwidth-derived residency rather than
nanosecond aesthetics. Cite that run, not this paragraph.

---

## F. THE CONVERGENT LAW — the same finding, three domains, none citing another

| where | statement |
|---|---|
| stockfish D-SF-RUNG-1 | reading `evaluate` at ply N to judge ply N is the same instrument re-applied, "wearing a hindsight costume" |
| `alpha.rs` `attended_mask` doc | *"an expected mask derived from the same traversal that claimed would make expected ≡ observed by construction and the diff vacuously empty"* |
| `revision.rs` anti-laundering | re-reading inherited material mints no evidence; fusion requires that a horizon actually touched the world |

**Re-applying your own output is not new evidence.** Any falsifier in this
family owes an INDEPENDENT route for the expected side, the way the PGN result
header replaced a re-evaluated final position.
