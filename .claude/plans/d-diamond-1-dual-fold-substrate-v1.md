# d-diamond-1-dual-fold-substrate-v1 — can one 8×2×8-shaped carrier support both point-peek and population-mask traversal?

**Status:** ACTIVE — operator-directed 2026-09-18, one probe arc, worked from `main`
`a2a51012` (post-#1248) on branch `claude/d-diamond-1`. Commit 1 = rulings + tests +
this plan; commit 2 = the probe; commit 3 = results + verdict.
**D-ids:** D-DMD-R1 · D-DMD-R2 · D-DMD-L · D-DMD-P1 … D-DMD-P4 · D-DMD-F (rows on
`STATUS_BOARD.md`).
**READ BY:** truth-architect · falsifier-auditor · measurement-skeptic · v3-envelope-auditor ·
zero-copy-warden · baton-handoff-auditor · dto-soa-savant.
**Extends (does not supersede):** `.claude/knowledge/three-prefix-fold-carriers.md`
(three carriers, three folds — this arc adds the FOURTH fold, *bound*, and measures the
unmeasured whole-facet 8-tile cell); `E-BYTES-ARE-STORED-INTEGERS-ARE-PROJECTED-1`
(#1248: the image is bytes, the hierarchy lives in the projections); D-MRX-7
(`Pred::Range` in the mask-risc IR, whose doc named the missing ordering knowledge).
**Not touched, by rule (the fence):** GridLake deterministic landing · address-derived row
placement · dense-basin allocation · `NodeGuid` · `CausalEdge64` · the JC clippy red
(`TD-JC-CLIPPY-RED-ON-BASE-2`) · DAG folding · Hamming/similarity folds · value-slab
decoding · temporal alpha-layer · planner cost model. Those may become follow-ups only if
this arc earns them.

---

## 0. The question

> Can one canonical 8×2×8-shaped carrier support both point-peek and population-mask
> traversal, with semantic hierarchy reduced to prefix/bound folds, while async writes
> remain invisible to sealed readers?

- **point universe = peek** — a pairwise fold over one `FacetCascade` (carrier 3,
  byte-addressed; `three-prefix-fold-carriers.md` §2).
- **field universe = mask** — a population fold over the same ordinal lane.
- **same canonical carrier, same ordinal population**; semantic traversal expressed as
  folds over that carrier, never as a graph walk.

Verdict vocabulary, fixed in advance: **PROVEN** (dual substrate works and witnessed bound
lowering materially wins in its valid region) · **BOUNDED** (works only above/below a
measured crossover or in a narrower cache/population regime) · **FALSIFIED** (the shape
does not earn the traversal advantage, or ordering/seal costs erase it). *Do not rescue a
disappointing result by expanding scope.*

## 1. Locked rulings

### R1 — tile 0 is canon (D-DMD-R1, shipped commit 1)

`FacetCascade::shared_prefix_tiles` must express semantic hierarchy coarse→fine. The stored
LE image places `custom` (APP_PREFIX) at bytes `[0..2)` and `canon` (the concept) at
`[2..4)` because `ClassidOrder::CanonHigh => (canon << 16) | custom`; the pre-R1 lens ran
`trailing_zeros` straight over `as_u128()` and therefore counted the app BEFORE the
concept — a latent semantic inversion, not an alternate traversal order. **The canonical
LE storage image is unchanged.** Only the projection was fixed: the two classid tiles of
the XOR are swapped (`rotate_left(16)` on the low 32 bits) before counting.
Recorded as `ISSUES.md` `ISS-SHARED-PREFIX-TILES-CLASSID-INVERSION`.
Red-first: `diamond_tests::f1_le_byte_order_is_not_semantic_tuple_order` and the F5
depth test both FAILED against `a2a51012` (F1 at *"same concept, different app ⇒ canon
tile shared"*, lens said 0; F5 at depth 1) before the fix, and pass after. One existing
expectation flipped (`redout_is_granularity_free_and_orthogonal`: a flipped bit 0 is a
`custom` flip, so the corrected lens reports 1 shared tile, not 0 — annotated in place).

### R2 — the ordering witness (D-DMD-R2, shipped commit 1)

The normative lane ordering is **lexicographic unsigned order over the numeric
projections** `(facet_classid, tiers[0].as_u16(), …, tiers[5].as_u16())`, `facet_classid`
compared as its projected `u32` (canon-high preserved): **«numeric projection order over
the canonical LE image»**. "BE order of LE projections" is explanatory only, never the
contract. Proven equal to lexicographic order over `semantic_tiles()`
(`r2_numeric_projection_order_is_semantic_tile_lexicographic`), and proven DIFFERENT from
byte-wise order over the image.

The witness is **«storage-attested, planner-consumed»**: `ordered_lane::SealedFacetLane`
seals (sort-at-seal is the reference) or attests an already-ordered lane (refusing an
unordered one), minting `OrderedLaneWitness { version: LanceVersion, n_rows, digest }`;
`quack::Filter::prefix_facet` consumes it and lowers a prefix to `Cmp::Range` only when
`SealedFacetLane::bound` validates it. The planner never infers order from column
placement or schema. **No witness → sweep** (`MatchU64` over the two semantic `u64`
planes). **False/stale witness → `WitnessError`**, the bound path unavailable — never a
plausible wrong mask.

## 2. The four arms (probe: `crates/d-diamond-1-probe`, workspace-excluded)

| arm | universe | measures | against |
|---|---|---|---|
| **P1** D-DMD-P1 | point | `is_ancestor(a,b) := LCP(a,b) ≥ depth(a)` on 8 semantic tiles; latency per pair class: equal · canon mismatch · custom mismatch · early tier (t0) · late tier (t5) · unrelated | corrected 8-tile `shared_prefix_tiles` (tzcnt + swap) vs the peek chain over `semantic_tiles()` — **the previously unmeasured whole-facet cell; do not assume 1.7 ns** |
| **P2** D-DMD-P2 | field | `descendants_of(prefix)` = witnessed `lower_bound + upper_bound + mask_set_range`; bound cost · mask-write cost · sweep cost · total · crossover N · hit-count sensitivity · L2-resident vs L2-evicted | the `MatchU64` sweep (ndarray `ternary_match_u64_to_mask`) over the semantic `u64` planes |
| **P3** D-DMD-P3 | fold ∩ | ontology prefix ∩ correlated L4 `6×(8:8)` tenant prefix over the SAME ordinal: A = witnessed bound on the ontology lane + the tenant sweep NARROWED to `[lo,hi)` + AND; **the tenant lane cannot carry an order witness over the ontology ordinal** (it is not sorted — the probe asserts `attest_sorted` refuses it), so "two witnessed bounds" is measured as bound + narrowed sweep and the refusal is a recorded finding | B = two full sweeps + AND. Absolute and relative. No join structure. |
| **P4** D-DMD-P4 | seal | sealed peek latency and sealed bound latency, each with no writer and with an open writer appending out of order; seal-sort cost; attest (digest) cost; publish/swap cost. Readers pin one sealed `Arc<SealedFacetLane>` (the `at(version)` path) | required property: «open-lane producer arrival order must not perturb reads from the sealed image» |

Inputs: 1M synthetic keys, an 8-tier ontology with Zipf-skewed branching at every tile so
subtree sizes differ by orders of magnitude; keys sorted by the normative order; a second
lane in L4 shape correlated by construction (tile-wise dependence with noise) so the
intersection is non-trivial. Deterministic SplitMix64 seed `0x9E3779B97F4A7C15`.

## 3. Falsifiers (D-DMD-F)

| id | claim it can kill | where |
|---|---|---|
| **F1** | LE byte order is semantic tuple order — same concept/different app vs same app/different concept ranked correctly by a raw byte scan | `facet.rs` `diamond_tests::f1_…` (contract, CI) — raw scan ranks them inverted; the corrected projection reverses it; bytes unchanged |
| **F2** | the ordering witness is decorative — a shuffled lane's unwitnessed bound still agrees with the oracle | `ordered_lane.rs` `f2_…` — `attest_sorted` refuses the shuffle; `bound_unwitnessed` on it is caught by the oracle |
| **F3** | a false/stale witness can execute the bound and return a plausible mask | `ordered_lane.rs` `f3_…` (version / row-count / digest / re-seal) + quack `invalid_witness_lowers_to_the_sweep_and_says_why` — every mismatch is an error before any range; the lowering emits no `Range` leaf |
| **F4** | the benchmark populations are vacuous | probe runtime asserts on every timed range/intersection: `kept > 0`, `kept·3 < total`; for ∩: neither mask contains the other, AND ≠ ∅, AND ≠ A, AND ≠ B |
| **F5** | an off-by-one hides behind a depth | depths 0 · canon (1) · custom (2) · every tier (3..=7) · exact (8): `facet.rs` `f5_…`, `ordered_lane.rs` `f5_…`, quack `witnessed_bound_and_sweep_lower_the_same_predicate_at_every_depth` |

Every arm is oracle-checked before it is timed (the existing probe's discipline).

## 4. Deliverables and the report

Commit 1: this plan · `STATUS_BOARD.md` rows · `INTEGRATION_PLANS.md` entry · `ISSUES.md`
entry · R1 fix + R2 witness + lowering, with tests (contract 1356 → 1367, quack 14 → 17).
Commit 2: the probe crate. Commit 3: results into §5 below, `EPIPHANIES.md`,
`three-prefix-fold-carriers.md` (fold 4), STATUS_BOARD statuses, supersession index
regenerated LAST.

Final report must carry: commit SHA · exact test counts · disable/falsifier runs ·
benchmark environment · raw timings · crossover N · whether sealed-read silence held ·
seal-sort cost · cache sensitivity · **the smallest ruling the measurements support**.

## 5. Results (commit 2, N = 1M, this branch, `rustc 1.98.1 (48a229cea 2026-09-01)`, 4-core Xeon @ 2.10GHz, L1d 192KiB/4, L2 8MiB/4, L3 260MiB)

### The no-sweep-in-a-fold correction (mid-arc)

The first probe pass (commit-2 draft, not landed) put a materialization on both
timed fold paths and was rejected before commit: **P2** sized its mask
destination to `words_for(n_rows)` — the WHOLE lane — so `mask_set_range`'s own
zero-before/ones-inside/zero-after write touched O(N) words regardless of how
narrow `[lo, hi)` was, hiding an O(N) cost inside what was reported as a fold.
**P3**'s "fold" arm A called `ternary_match_u64_to_mask` (a per-row sweep)
narrowed to the bound's row range — still a sweep, since narrowing the range a
sweep runs over does not change what it is. The rule the rewrite follows: a fold
must not build a population- or lane-sized buffer, and must not run a per-row
predicate.

Both were rewritten before any number below was taken:

- **R2 gained a `SemanticLens`.** Storage is a content-blind ordinal; "sorted"
  is meaningful only under a named projection, and one physical sequence is
  monotone under exactly one lens at a time (F1 had already shown this at the
  byte level — the LE image's own order disagrees with its numeric
  projection's order). `OrderedLaneWitness` now carries
  `lens: SemanticLens` (one variant shipped, `CanonHighTiles8`) and
  `SealedFacetLane::bound` rejects a lens mismatch before searching
  (`WitnessError::LensMismatch`), the same discipline as version/rows/digest.
  Contract: 1367 → 1368 tests.
- **P2's write is now `touched_write(lo, hi) -> (w0, dst)`** — a base word
  index `w0 = lo / 64` plus `words_for(hi) - w0` words, so the cost is
  O((hi − lo) / 64) at any position; `n_rows` does not appear in its signature.
  (The first version of this fix sized to `words_for(hi)` and wrote from word 0
  — cost O(hi), the range's end position. See §5's position table.) The
  full-lane sweep is kept ONLY as an explicitly separate `reference_sweep_ns`
  column, never summed into `bound_fold_total()`.
- **P3's fold arm is a `JointIndex`** — a probe-only (not shipped) Morton-style
  interleave of the two lanes' semantic tiles (`A0 B0 A1 B1 …`, capped at
  `JOINT_MAX_DEPTH = 4` tiles per side to fit a `u128`), built once
  (`JointIndex::build`, timed separately as a real cost) and then bounded with
  exactly two `partition_point`s. Verified structurally, not just by report:
  `ternary_match_u64_to_mask` appears nowhere in `JointIndex`; the only sweep
  call sites in the probe crate are inside the `reference_sweep_ns` timing
  block. Limitation: equal-depth prefixes only (`a_depth == b_depth`); unequal
  depths were not attempted rather than shipping a complicated scheme.
- **P4**: the writer's clone is structurally necessary (the open buffer must
  keep accumulating independently of the immutably-published snapshot); what
  changed is sorting `open` in place before cloning, so pdqsort sees a mostly-
  sorted prefix with a short unsorted tail on repeat seals, cheaper than
  sorting a freshly-cloned, fully-shuffled-since-last-seal buffer.

### P1 — point universe (64K pairs/class, min of 7)

The previously unmeasured whole-facet 8-tile cell does **not** inherit the
1.72 ns axis-chain number: all three arms cluster at 1.7–4.2 ns, with the
byte-peek arm (the shape that WAS 1.72 ns on the 6-tier axis chain) actually
SLOWEST on identical/late-tier pairs (~4.0 ns) because it pays two byte loads
per tile with no early exit until 8 tiles in. The 8-tile classid-first
layout does not give the axis chain its early-exit advantage back.

| pair class | tzcnt+swap | peek u16 | peek bytes |
|---|---|---|---|
| fully equal | 2.02 | 2.38 | 4.18 |
| classid canon mismatch | 2.01 | 1.73 | 1.73 |
| classid custom mismatch | 2.01 | 1.73 | 2.07 |
| early tier (t0) mismatch | 2.01 | 1.73–2.28 | 2.08–2.28 |
| late tier (t5) mismatch | 1.98 | 2.30–2.36 | 3.94–3.98 |
| unrelated | 2.00 | 2.84–2.93 | 2.93–2.94 |

`is_ancestor(a,b) := LCP(a,b) >= depth(a)` verified against all three arms and
the class's expected LCP on every generated pair before timing (oracle-first).

### P2 — field universe: bound + touched-write vs the full-lane sweep

Two falsifiers, because the answer has two axes and the first one alone
certified only one of them.

**(a) Size axis** — fixed absolute range `[500,600)`, N varied 1000×:

| N | touched_write_ns | old whole-lane-sized_ns |
|---|---|---|
| 1,000 | 22.11 | 34.45 |
| 16,000 | 23.14 | 54.48 |
| 256,000 | 21.98 | 334.92 |
| 1,000,000 | 22.57 | 4,619.80 |

**(b) Position axis** — fixed 100-row width, position moved 8000×:

| range | width | touched_write_ns |
|---|---|---|
| `[500, 600)` | 100 | 21.68 |
| `[64,000, 64,100)` | 100 | 20.50 |
| `[1,000,000, 1,000,100)` | 100 | 20.75 |
| `[3,999,900, 4,000,000)` | 100 | 20.49 |

**Flat within noise on both axes; the old whole-lane buffer grew ~134× on (a).**

Table (b) is the correction this plan's own first fix needed. That fix sized
the destination to `words_for(hi)` and wrote from word 0, so `mask_set_range`
zeroed every word before `lo` and the cost was O(`hi`) — the range's END
POSITION, still population-shaped for a range near the lane's end. Table (a)
could not see it, because holding `(lo, hi)` fixed across N holds `hi` fixed.
The plan text at the time even argued that moving the position would be an
*invalid* measurement; that argument was the defect defending itself. The
shipped `touched_write` returns `(w0, dst)` with `w0 = lo / 64`, making the
cost O((hi − lo) / 64) at any position. Falsifier
`f_touched_write_is_position_independent`, disable-verified red against the
old shape.

At N = 1M: `bound` 238–265 ns (flat, O(log N)), `touched_write` 234–4,287 ns
(scales with the RANGE width, not N — d=1..4 prefixes here keep large
populations, `1.2%–3.4%` of the lane, so their touched write is wide;
d≥5 prefixes are near-singleton, 1 row, and their touched write is one word),
`reference_sweep` 429,100–910,240 ns. Speedup (bound+write vs reference sweep)
119×–707× at 1M, 200×–564× at 256K, degrading toward parity below N≈256–512
where the reference sweep is itself cheap. L2-evicted (64 MiB stream before
each round; L3=260MiB is not evicted): bound and touched-write both grow
(2.4–8.2 µs) but the reference sweep grows more (up to 1.51 ms), so the
speedup ratio survives the cache-cold regime, just narrower.

### P3 — fold intersection over one ordinal (JointIndex, equal depths only)

The tenant lane cannot be attested over the ontology ordinal (`WitnessError`,
first inversion at row 1) — the structural finding the plan predicted: one
sequence, one lens-order; a second, independently-generated lane over the
same rows is not sorted under that lens and gets no bound of its own.

Joint index build (sort 1,000,000 interleaved `u128` keys, once): **61.2 ms**.
Fold cost thereafter — two `partition_point`s and, for the comparable column,
the O(kept) remap back to the world's own ordinal. Zero per-row work over the
population in either:

| depth | kept A | kept B | kept ∩ | bound ns | + materialize ns | reference (2 sweeps + AND) ns | speedup |
|---|---|---|---|---|---|---|---|
| 3 | 1,984 | 194 | 64 | 79 | 98 | 797,268 | 8,135× |
| 4 | 1,666 | 66 | 65 | 69 | 89 | 745,473 | 8,376× |

**Quote the `+ materialize` column, not `bound`.** The comparator produces a
full original-ordinal mask; `JointIndex::bound` produces two offsets into the
JOINT index's own order. Timing only the bound compares inequivalent outputs
and inflated this ratio to ~10,700× in the first write-up. `materialize_rows`
is O(kept) — proportional to the answer, so still a fold — but it is not free.

(depth 2 found no F4-passing pair in 20,000 draws at this seed — not a defect,
the intersection floor of 32 rows was simply not hit at that depth's
population sizes here.) F4 anti-vacuity (`kept ∩ >= 32`, neither side a
subset) holds on both rows.

### P4 — sealed reader under an open writer

Peek and bound distributions (min/median/p90) essentially unchanged with an
active writer publishing 20K-row batches: no perturbation beyond measurement
noise; the pinned `Arc<SealedFacetLane>`'s digest and validation were
unchanged throughout (`SealedFacetLane::validate` still `Ok` against the
witness taken before the writer started, after the writer had published a
strictly higher version). **Property held: open-lane producer arrival order
does not perturb reads from the sealed image.**

### Falsifier status

F1/F2/F3/F5 shipped and green at commit 1 (contract + lowering level). F4
enforced at runtime throughout P2/P3 (anti-vacuity: `kept > 0`, `kept·3 <
total`; intersection floor 32, neither side a subset). No falsifier failed.

### Verdict: **BOUNDED**

The dual peek/mask substrate over one 8×2×8-shaped carrier works, and the
bound fold materially wins in its region — but the region has a real edge,
named here rather than smoothed over:

1. **P1 does not transfer.** The 1.72 ns axis-chain result was specific to
   that carrier's 6-tier, 2-tile-excluded shape. The whole-facet 8-tile
   compare is ~2 ns regardless of arm — a real number, not the number that
   was extrapolated for it in the plan's own framing ("do not assume 1.7 ns").
2. **P2's crossover is real and depends on depth.** Below N≈256–512 the bound
   fold does not clearly beat a full sweep (the sweep itself is cheap at
   small N); above it, the win is 100×–700× and grows with N. The two
   populations shipped by this generator (wide at shallow depth, singleton at
   deep depth) both cross this line by N=1M but the WIDTH-dependent
   `touched_write` term matters at the wide end.
3. **P3's win is CONDITIONAL on a prebuilt `JointIndex`, and its applicability
   is further conditional on the ordering witness's lens matching — which a
   second, independently-written lane almost never satisfies for free.** The
   ~8,200× fold speedup is real once a `JointIndex` exists, but building one is
   a ~61 ms up-front cost that amortizes only across repeated queries at fixed
   depths on a fixed pair of lanes — this is a cache/index the caller must
   choose to build, not a free property of the substrate.
4. **The `SemanticLens` correction is load-bearing, not decorative**: it is
   the reason P3 could not simply reuse the ontology lane's witness, and it
   is the mechanism that will let a second, differently-lensed order (e.g. a
   joint key) coexist with the first without either silently validating
   against the wrong one.

Not touched, per the fence: no GridLake placement, no `NodeGuid`/`CausalEdge64`
change, no JC clippy fix, no DAG folding, no Hamming fold, no value-slab
decode, no planner cost-model work.
