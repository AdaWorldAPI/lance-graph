# RP-SEAL Consolidation — First Independent Pass (2026-08-19)

> Consolidation of the 15-researcher independent pass mandated by
> `.claude/plans/erasure-seals-compaction-research-v1.md` (§12–§13).
> Run `wf_ca974718-1b4`: 15/15 cells completed, 0 errors, 0 empty results,
> ~5.59M subagent tokens, 842 tool uses. Full per-cell reports committed as
> **Appendix H** under `docs/lotus/rp-seal-v1/` (A1–E3 plus
> `A0-orchestrator-prepass.md`, the orchestrator's private pre-dispatch
> cross-check, unseen by any researcher).
>
> **Independence discipline held.** First pass ran with no cross-talk, blind
> to `docs/lotus/**` and all board files. Builder/adversary/scout roles per
> domain. **Scope-pivot filter applied and PASSED**: the operator's mid-run
> rulings (rustynum struck — everything ndarray; `crates/symbiont`
> deprecated) bind this consolidation as a hard filter; audit of all 15
> reports found zero findings *sourced from* either — the only two mentions
> (D1, C3) cite the exclusion ruling itself, and D1 converts it into a
> finding (see M9). Nothing required re-anchoring or re-running.
> **Model-tier discipline held**: 0 cargo invocations by any researcher.
>
> Charter maxim, §17: *do not make the beautiful idea true; make it survive
> fifteen people trying to make it false.* This document records what
> survived, what died, and what was refined into a sharper question.

---

## 1. Evidence matrix

Columns per charter §12. "Independent?" marks genuine independent
rediscovery (high-value) vs single-source claims. Cells cite their own
file:line evidence in the Appendix H reports; only the consolidated claim
and its strongest anchors are repeated here.

### M1 — Lance 9.0.0 already ships the apparatus; the consumer uses none of it

**Claim.** Deferred remap via Fragment-Reuse-Index, rank-based
O(#fragments) row-address remap (`RowAddrRemap`), stable row IDs,
size-bounded incremental compaction, `IndexRemapper` (caller-receivable
old→new remap), and per-row `created_at_version` / `last_updated_at_version`
provenance (RLE, compaction-correct) are all public and un-feature-gated in
the pinned 9.0.0. lance-graph's production write path
(`LanceCycleWriter`/`cycle_sink.rs`) is bootstrap-create-then-forever-append
and invokes none of it; grep for the machinery across `crates/` returns
zero non-vendored hits. `excluded_fragment_ids` / `max_source_rows` /
`max_source_bytes` are upstream-main-only (11.0.0-beta.14).
**Support:** A1 (two independent runs, concordant), A2, A3, E1, E3 — five
cells, independent. **Dissent:** none. **Confidence:** HIGH (settled
archaeology). **Falsifier:** none needed; source-anchored.
**Next:** consume, don't rebuild — see Tier table.

### M2 — Locality-repair-by-compaction is structurally constrained, twice over

**Claim.** (a) Compaction at both versions cannot reorder rows; under
`IndexRemapMode::Compact` order preservation is a *correctness contract*
(ascending-id guard, `row_addr_remap.rs:35-45,139-160`), so
locality-repairing compaction is prohibited in the cheap mode — the escapes
are Direct's O(rows) coordinator RAM or stable row IDs. (b) Stable row IDs
hard-conflict with `defer_index_remap` (thrown `Err` by design) and delete
the FragReuseGroup permutation witness. The two obvious remap-cost levers
cannot be composed, and the witness is anti-correlated with the cheap
configuration.
**Support:** A2 (primary), A1 (the defer/stable conflict, independent).
**Dissent:** none. **Confidence:** HIGH.
**Falsifier:** attempt a reordering compaction under Compact remap → guard
rejection. **Next:** any layout work routes through fragment *rewrite with
an ordering key* (M3), never through remap.

### M3 — The one open, precedented upstream seam: a compaction-time layout key

**Claim.** A caller-supplied physical-layout ordering key applied at
compaction-rewrite time is the genuinely open seam: Lance already ships the
mechanism scoped to one index (R-tree Hilbert-sorted leaf pages), Delta
ships the base-table analogue (Z-order), and `IndexRemapper` is a shipped
inversion-of-control trait that lets an external consumer receive the exact
old→new remap on every compaction commit today — a free first "permutation
witness" experiment with zero upstream changes. The secondary-index remap
cost that motivates all of this traces to one documented upstream gap:
indices key on physical row addresses instead of the stable row IDs Lance
already has.
**Support:** A3 (primary), E3 (LSM mapping, independent), B1 (zero-cost
sort-by-own-key arm, independent). **Dissent:** none. **Confidence:** HIGH
for the seam's existence; the *benefit* is unmeasured (see M10).
**Falsifier:** F-AMPLIFY. **Next:** Tier 2 IndexRemapper experiment before
any Tier 3 RFC.

### M4 — The SFC default died; the question moved

**Claim.** (a) Measured: at every 4^j page size this architecture uses,
Morton and Hilbert induce the IDENTICAL page partition (they differ only in
page order), so pages-touched is identical for every query — the
Morton-vs-Hilbert debate is moot at power-of-4 granularity. (b) Measured:
the linear (non-rank) quantizer produces 20–45× petal hot-spots and 70–84%
empty fields on clustered identities; `morton_slot` over a hash is a bit
permutation, not a locality code. (c) Literature: an unbounded clustering
gap for near-full-extent queries, the Arrwwid d≥3 disadvantage for the
whole cube-recursive family, and a proven impossibility of one curve
serving mixed query shapes. (d) Measured against expectation: Hilbert's
mean code-span is 18–22% WORSE than Morton on square queries even where
its median is better; Z-order beats Hilbert 2.0× on dyadic-parity
selections.
**Support:** B2 (measurement) + B3 (literature) — independent routes to the
same kill; B1 concurs (the project's own non-interleaved tiered key beat a
real Morton variant on the range-query metric in the one prior in-repo
experiment). **Dissent:** none on the kill; B2 retains Morton as *minimax
default* only. **Confidence:** HIGH.
**Falsifier:** F-SFC is now partially discharged — the surviving form is
query-mix- and dimensionality-conditioned selection, with the quantizer
(rank vs linear), not the curve, as the first-order variable.
**Next:** B1's pre-registered arms (own-key-bytes sort first; true
bit-level Morton, Hilbert, spectral as open arms) under E1's harness.

### M5 — The comma/phase coefficient schedule is DELETED

**Claim.** The project-specific phase/comma coefficient schedule fails its
own charter gate: 0 of 6 necessary conditions non-trivially met;
cross-level syndrome separation is impossible under permutation (an
XOR-fold seal is permutation-invariant, so completion-order scrambling —
which the architecture explicitly declares non-semantic — erases exactly
the structure the schedule would need). Locus+version-bound per-chunk hash
+ flat MDS RS strictly dominates cascade/product codes on every detection
axis in the C2 truth table.
**Support:** C2 (adversary, primary). C1 independently bounds cascade
parity below at ≥33% overhead (Gopalan–Huang–Simitci–Yekhanin locality
bound) vs 6.25% for row/column P+Q; C3 independently shows hierarchy is
purchased with distance (generalized Singleton, hierarchical VIII.5,
availability bounds), never free. Three cells, three routes, one verdict.
**Dissent:** none. **Confidence:** HIGH.
**Falsifier:** X-C2-2 (the DELETE gate) remains pre-registered if anyone
wants the corpse re-examined; the burden is now on the schedule.
**Next:** nothing — per charter §17 the schedule is removed from the
program. The seal baseline is C1's recommendation (M6).

### M6 — The seal baseline: shipped hash + row/column P+Q over the native 64×64

**Claim.** The low-risk seal is the already-shipped content hash (~0.07%
overhead) plus row/column P+Q parity over the native 64×64 field grid:
6.25% overhead, 63× repair amplification vs flat-RS's 4095×. Repair-
bandwidth optimizations layered on an UNMODIFIED MDS seal
(Guruswami–Wootters trace repair, piggybacking) are proven and free;
hierarchical/cascade structure is a priced escalation, not a default. The
GF(2^8) SIMD idiom needed for RS constant-multiply (nibble-split + PSHUFB
+ XOR) already ships in `ndarray/src/simd_avx2.rs` (Harley–Seal popcount),
per the everything-ndarray ruling. One program-level question gates real
deployment: whether the target object store already erasure-codes one
layer down (double-coding wastes the overhead).
**Support:** C1 (primary), C3 (bounds, independent), C2 (flat-RS dominance,
independent). **Dissent:** none. **Confidence:** HIGH for the
recommendation shape; MEDIUM for the numbers until X-C2-1/X-C2-3 run.
**Falsifier:** F-RS + X-C2-3. **Next:** X-C2-1 injection harness (the
prerequisite), then X-C2-3.

### M7 — Query-locality × repair-locality: anti-synergy found, novel question refined

**Claim.** The naive cross-domain hope — one grouping serving both query
locality and repair locality — is an ANTI-synergy: Morton-style locality
manufactures exactly the correlated failure that destroys locality-aligned
parity groups (C2). Independently, C3 and E1 both searched and found NO
prior art posing the joint "one address order minimizing both query
scatter and repair scatter" question. The refined open question is
therefore: an address order serving query locality while parity groups
deliberately ANTI-align. This is the program's strongest Tier 4 candidate.
**Support:** C2 (anti-synergy), C3 + E1 (novelty, independent searches).
**Dissent:** none — the three compose rather than conflict.
**Confidence:** HIGH that the naive version is dead; the refined question
is open by construction. **Falsifier:** the E1 harness's repair-scatter
axis vs query-scatter axis, jointly. **Next:** only after Tier 0/1.

### M8 — The coalescing/seal implementation inverts its own headline (E2)

**Claim.** Amortization is real at the durable boundary — one fsync/cycle
pays above C≈100 casts — and false everywhere else in the current
implementation: (a) the coalescing path writes MORE payload-column bytes
than no coalescing at all, (1+C+D)·512 > C·512 always, with physical
amplification equal to b+1 — 65× at the repo's own b=64 — caused by 512 B
of null padding per landing row; (b) 52% of seal CPU is a byte-at-a-time
FNV-1a hash that is invariant under coalescing (47.5 ms at b=1, 4, 64
alike) and structurally forbids an incremental variant; (c) the parity
GRANULE, not the seal granule, sets the incremental-vs-batch crossover
(≈1.46% dirty-row density for 8 KiB units at the stated geometry).
**Support:** E2 (adversary, measured on the real code). **Dissent:** none;
no other cell touched these paths. **Confidence:** HIGH (measured), single
source — re-verification is cheap and required before the fix lands.
**Falsifier/next:** Tier 0 re-run of E2's measurements, then Tier 1 fixes:
sparse landing rows (kill the b+1 padding term) and a blockwise,
incrementally-composable seal hash (keeping fail-closed semantics).

### M9 — The temporal ladder: Layer 1 real, Layer 2 inert, one genuine novelty

**Claim.** `temporal.rs` Layer 1 (causal deinterlacing / crash recovery) is
production-wired and tested. Layer 2 (STRICT/AWARE/RETRO epistemic
projection) has exactly one non-test exerciser, whose own source admits the
interesting axes are inert; `T_now` (the hypothesis's third coordinate)
exists nowhere as a type; `hlc_tick` is HLC-named but not HLC-implemented
(no physical component, no merge rule). Lance itself already stamps the
W_write/A_last primitives per row (M1), so the missing coordinates need no
external stamping layer for the single-writer case. The version→kanban
OUT-bridge (`LanceVersionScheduler`) is tested library code whose only
non-test caller was in operator-deprecated `symbiont` — it now has zero
live callers (D1, converting the scope ruling into a wiring fact).
Literature: the reader-rung-selectable STRICT/AWARE/RETRO admission tier
has no counterpart in the surveyed systems (XTDB's current/history split
is binary; classical isolation is per-transaction) — the program's second
Tier 4 candidate. The retroactive-structures bound Θ(min(√m, n·log m)) is
a forward falsifier only (no retroactive-write mechanism exists).
**Support:** D1 (code), D3 (literature) — independent; A1 (upstream
primitives). **Dissent:** none. **Confidence:** HIGH.
**Next:** Tier 1 — introduce `T_now` as a type and a real HLC merge rule,
or explicitly rule both out; re-home or delete the caller-less OUT-bridge.

### M10 — Random placement is cheaper than assumed; irreversibility is the real cost

**Claim.** E2's own adversarial cache-tiling thesis was self-DISPROVED:
random identity placement costs ~nothing at the design's 32 MiB geometry
(0.97× vs sequential) and only 2.30× at 512 MiB. B1: today's arrival order
is architecturally closer to a random control than to any spatial order.
E3: vendor cost warnings (Snowflake/Iceberg/Delta) triangulate that
unconditional locality-key ordering is known-costly maintenance. A2: the
real cost of random placement is IRREVERSIBILITY (M2), not read latency.
Consequence: physical-layout work needs a measured read-pattern
justification at real geometry before any engineering; E3's
entropy/scatter "locality debt" compaction trigger — the one genuine
literature gap in the LSM mapping — is the cheap first experiment.
**Support:** E2, B1, E3, A2 — four cells, independent, convergent.
**Dissent:** none. **Confidence:** HIGH. **Falsifier:** E3's pre-registered
locality-debt experiment; F-NOCOMPACT. **Next:** Tier 0.

### M11 — The integrity surface has defective seals wearing strong names

**Claim.** Five shipped seal/integrity paths, three defective in
false-accept terms (C2 §B1–B5): `firefly_frame.rs` advertises
"Reed-Solomon ECC" while `compute_ecc` is a 4-way XOR fold with zero
detection capability (independently found by C3); `merkle_tree.rs` (the
in-tree multi-scale syndrome) carries four defects including a
100%-false-accept path wearing a BCH decoder's name; `unified_audit.rs`
chains FNV-1a as tamper evidence (trivially forgeable — though its
callcenter threat model is drift, not adversaries); `seal.rs` is 48-bit
truncated BLAKE3 with an alpha-mask caveat; `persist_sink.rs`'s cycle
seal is FNV-1a-based (see M8's CPU finding). Hash-only seals false-accept
on exactly the faults this system generates (wrong-slot/wrong-version
writes of valid bytes) — locus+version binding is the fix, not wider
hashes.
**Support:** C2 (primary), C3 (firefly, independent). **Dissent:** none.
**Confidence:** HIGH (source-anchored). **Next:** Tier 0 — fix or
truthfully re-document each path; X-C2-1 harness pins false-accept rates.

### M12 — Eleven order leaks into durable coordinates; L2 escalates first

**Claim.** D2's leak table (report §"The leak table", L1–L11): 11
arrival/physical-order leaks into durable coordinates, 10 unpinned. The
escalation is **L2**: durable semantic replay order IS Lance physical scan
order (`cycle_sink.rs:971-980` `scan_in_order(true)` with no `order_by`;
`persist_sink.rs:665-666` "no sort is done here") — so compaction here is
a semantic-order-MUTATING operation, not locality repair. Close behind:
L3 (same-row coalescing resolved by arrival under a non-injective
`row_of`), L6 (`QueryReference::default()` = `u64::MAX` admits future rows
even under Strict), L8 (deinterlace sort key mixes HLC ticks and Lance
versions as commensurable). The one pinned leak (batch_hash, F-ORD-REAL)
is the safest member of its family — it fails closed.
**Support:** D2 (primary); A2 independently establishes the
compaction-order contract that makes L2 unfixable-by-accident (M2).
**Dissent:** none. **Confidence:** HIGH (file:line-anchored).
**Falsifiers (pre-registered in D2):** F-PHYS-ORDER (highest value),
F-ORD-2, F-RESTART, F-HINDSIGHT family. **Next:** Tier 0 — these gate ALL
physical-layout work: layout experiments on a substrate whose replay order
is physical order would corrupt semantics silently.

### M13 — Measurement infrastructure: 80% exists; two hazards recorded

**Claim.** The harness family is specifiable today from shipped primitives
(`measure_wal_curve.rs`, 3,558 lines, plus Lance's own compaction/remap
machinery); the one real gap is hardware cache-miss counting (perf-event /
iai-callgrind, neither wired). Hazards: (a) the 512-byte witness payload
sits between the cloud (1000 B) and local (10 B) early-materialization
thresholds, so file:// benchmarks INVERT production S3 behavior — every
probe must run both storage schemes; (b) charter §14's "do not optimize a
metric that was not measured" currently strikes cache misses until the
counter lands.
**Support:** E1 (primary), A2 (threshold hazard, independent).
**Dissent:** none. **Confidence:** HIGH.
**Next:** Tier 0 — wire perf-event or strike the metric; add the
dual-scheme rule to every probe design.

### M14 — Cleanup and temporal horizons are one resource (design coupling)

**Claim.** `cleanup.rs:498-514` deletes exactly the manifests that
checkout_version-based STRICT/AWARE/RETRO horizons resolve to:
stale-version retention bytes are pinned by the cognitive layer's temporal
window, not by an operational retention policy. Retention must be derived
from the temporal window. Related seed: `FragReuseIndexDetails.Group.
changed_row_addrs` already records which rows changed per version, so a
retained per-version parity delta can triage stale-vs-corrupt (C2).
**Support:** A2, C2 — independent halves of one design coupling.
**Dissent:** none. **Confidence:** HIGH for the coupling; the parity-delta
triage is a design seed, unmeasured. **Next:** Tier 1 design note in the
temporal work (M9).

### M15 — Orchestrator pre-pass reconciliation (disagreement record)

The pre-pass (`A0-orchestrator-prepass.md`) claimed "prepared fragments →
one manifest publication" via `execute_uncommitted` + `CommitBuilder::
execute_batch`. A2 found compaction's own path commits twice
(`reserve_fragment_ids` is itself a `commit_transaction`; an abort between
the two permanently advances `max_fragment_id`). Both are true: the
two-phase INSERT path exists exactly as the pre-pass reports; the
COMPACTION path does not get the same shape. The prepared-artifact
publication doc (D-LOTUS-6) stands for inserts; any compaction design must
budget two versions per pass. No other pre-pass claim conflicted with the
fleet; the fleet exceeded the pre-pass everywhere else.

---

## 2. Cross-domain relationship graph

Edges marked ⊕ are independent rediscoveries (high-value per §12);
⊗ marks composition of findings that never met in one cell.

- **M2 ⊗ M12 (A2 × D2):** two independent reasons the current substrate
  blocks layout work — compaction cannot reorder (correctness contract),
  and if it could, it would mutate semantic replay order (L2). Fixing L2
  (an explicit durable order key) is therefore a PREREQUISITE for M3's
  layout-key seam, not a parallel track.
- **M5 ⊕ (C1 × C2 × C3):** three independent routes (truth table; locality
  bound; distance bounds) to the same kill of hierarchical/cascade coding
  as a default. Genuine independent confirmation.
- **M7 ⊗ (C2 × C3 × E1):** the novelty (no joint prior art) survives, but
  C2's anti-synergy inverts its naive form — the composition is sharper
  than any single cell's claim.
- **M4 ⊕ (B2 × B3 × B1):** measurement, literature, and the one prior
  in-repo experiment all against an unconditional SFC default.
- **M1 ⊕ (A1 × A1' × A2 × A3 × E1 × E3):** the double-run of A1 (resume
  artifact) plus four other cells — concordant archaeology; treated as one
  claim with unusually deep support, not six discoveries.
- **M9 ⊗ (D1 × D3 × A1):** code inertness + literature novelty + upstream
  per-row provenance compose into the temporal work order: the novel tier
  is worth building precisely because the primitives are already stored.
- **M8 ⊗ M6 (E2 × C1):** any parity overhead percentage is meaningless
  until the b+1 padding amplification is fixed — 6.25% parity on top of
  65× padding is noise. E2's fix precedes C1's seal.
- **B1 ⊕ E3 ⊕ B3 (convergent shape):** the recursive-bisection →
  HEEL/HIP/TWIG address shape was independently built four times in this
  workspace (B1), matches the packed-memory-array amortized bound lineage
  (E3), and is the same recursive-block decomposition as cache-oblivious
  layouts (B3). The workspace keeps rebuilding one structure; naming it
  once is a documentation deliverable, not new engineering.

Shared-ancestry caution: all five E/A cells read the same Lance source
tree, so their agreement on M1 is expected (same primary source), unlike
the M5 triple, whose three routes are methodologically disjoint.

---

## 3. Tiering (charter §13)

**DELETED (survived nothing):**
- The phase/comma coefficient schedule (M5). C2's X-C2-2 remains
  pre-registered should anyone want to re-litigate; the default is gone.
- Unconditional Morton (or any SFC) as a layout default (M4). Survives
  only as a query-mix-conditioned choice with a rank quantizer.
- "Compaction = one manifest publication" (M15). Two commits, budgeted.
- The naive "one grouping serves query AND repair locality" (M7).

**TIER 0 — measurement/falsifiers only (gates everything):**
1. Pin D2's unpinned leaks: F-PHYS-ORDER (L2, highest value), F-ORD-2
   (L3), F-RESTART (L1/L5/L9), F-HINDSIGHT family (L6-L11 subset). (M12)
2. Re-verify E2's three measurements (b+1 amplification, FNV CPU share,
   parity-granule crossover) on the committed tree. (M8)
3. X-C2-1 injection harness; pin false-accept rates for the five seal
   paths of M11 (fix or re-document each).
4. E3's locality-debt (remap-entropy) trigger metric — the cheap
   falsifier before ANY layout/movement engineering. (M10)
5. Wire perf-event cache-miss counting or strike the metric; adopt the
   dual-storage-scheme rule for every probe. (M13)

**TIER 1 — lance-graph internal experiments/fixes:**
1. Sparse landing rows — kill the b+1 null-padding amplification. (M8)
2. Blockwise, incrementally-composable seal hash with locus+version
   binding (replaces FNV; keeps fail-closed semantics). (M8, M11)
3. An explicit durable order key for replay (fixes L2/L9; prerequisite
   for all layout work). (M12)
4. `T_now` as a type + a real HLC merge rule, or their explicit
   rejection; retention policy derived from the temporal window;
   re-home or delete the caller-less OUT-bridge. (M9, M14)

**TIER 2 — generic Lance prototypes (no upstream changes needed):**
1. `IndexRemapper`-consumer permutation witness (A3's free experiment).
2. Row/column P+Q seal over the 64×64 grid as an external layer, after
   Tier 0 №2/№3 (C1's 6.25%/63× numbers, verified by X-C2-3).
3. B1's layout arms under the E1 harness: own-key-bytes sort (zero-cost
   arm) vs arrival baseline vs bit-level Morton/Hilbert/spectral — only
   if Tier 0 №4 shows measurable locality debt.

**TIER 3 — credible upstream RFC/PR candidates (each must answer the
§13 questionnaire before filing):**
1. Caller-supplied compaction-rewrite ordering key (M3; precedents:
   in-repo R-tree Hilbert leaves, Delta Z-order).
2. Stable-row-id keying for secondary indices (closes the documented
   remap-cost gap A3 identified upstream).
3. Overlay-staleness as a compaction trigger (A3's novel-candidate seed;
   the docs already name it, no code consumes it).

**TIER 4 — paper-worthy questions (after Tiers 0–2 produce data):**
1. The refined joint question: address order for query locality with
   deliberately ANTI-aligned parity groups (M7).
2. Reader-rung epistemic admission tiers (STRICT/AWARE/RETRO) as a
   temporal-database primitive (M9; D3 found no counterpart).
3. The b+1 coalescing amplification analysis as a negative result (M8).

---

## 4. Work order

Tier 0 items 1–3 are independent of each other and of Tier 1 №4; they can
run as separate probe PRs. Nothing in Tier 2+ starts before its named
Tier 0 gate is green. The charter's phase structure (probes → design →
paper) resumes from here; §16's paper skeleton acquires its §10 (negative
results) and §11 (cross-domain discoveries) content from this document.

## 5. What the maxim did

Fifteen researchers were pointed at the beautiful ideas. The comma schedule
died in one afternoon (M5) — with three independent proofs, which is a
cheaper funeral than one shipped defect. The SFC default died (M4). The
amortization headline inverted (M8). What survived is better than what was
proposed: a seal that is boring and correct (M6), a layout seam with real
upstream precedent (M3), two genuinely novel questions with no prior art
(M7, M9), and a Tier 0 list that fixes real, file:line-anchored defects
before any new structure is built on them.
