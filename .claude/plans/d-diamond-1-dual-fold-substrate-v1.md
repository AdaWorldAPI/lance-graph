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

## 0. The question (operator, verbatim)

> «Can one canonical 8×2×8-shaped carrier support both point-peek and population-mask
> traversal, with semantic hierarchy reduced to prefix/bound folds, while async writes
> remain invisible to sealed readers?»

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

## 5. Results

_(commit 3 — not yet measured; nothing below this line is a claim until it is.)_
