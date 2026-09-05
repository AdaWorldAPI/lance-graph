# nexgen-mask-histogram-thresholds-v1 — the exposure meter is a nested mask set

**Status:** IN PROGRESS (2026-09-05). §5 steps 1–4 GREEN; D-NXG-1/4/5 shipped in #1181 (`planner::nested_bands`, `contract::shape_rank`, `jc::stats::fisher_2z`); D-NXG-3 lives inside `NestedBands::split` (not promoted to T1); rooms 9–27 of §3 remain PROPOSAL and unmeasured.
**D-ids:** D-NXG-1 … D-NXG-12 (rows on `.claude/board/STATUS_BOARD.md`).
**Evidence:** `.claude/nexgen/harvest/` (11 verbatim agent reports, 2026-09-05).
**Board block:** `.claude/board/EPIPHANIES.md` 2026-09-05 `E-NXG-*` (16 entries).
**Location note:** this plan lives under `.claude/nexgen/plans/`, not
`.claude/plans/`. `supersession_index.py` and `plan_dids.py` scan only
`.claude/plans/*.md`, so the D-ids here are visible to the STATUS_BOARD and
invisible to the index's coverage column. That is a known gap, recorded on
purpose: moving the file is a one-line change once the folder earns a second
plan. Do not read the coverage column's silence as "untracked".

---

## 0. The one sentence

A Belichtungsmesser reading over N rows is a chain of nested row masks
`M_1 ⊆ M_2 ⊆ … ⊆ M_16`; its histogram is 16 popcounts; a row's Prozentrang is
the index of the innermost mask containing it; bucket i is `M_i ∧ ¬M_{i−1}`
(`vpternlogq` immediate `AND_ANDNOT2 = 0x10`); bucket overflow is a popcount
test; rollover is a mask split keyed by Lance version. Threshold, band, bucket,
rank, cache entry and basin are the same object read five ways.

## 1. What exists (from the harvest, file:line as of 2026-09-05)

| mechanism | where | shape today | gap |
|---|---|---|---|
| reject floor | `ndarray/src/hpc/cascade.rs:137` | `mu + 3σ`, k fixed | Gaussian-tail assumption; Jirak note at `rolling_floor.rs:25` says it is wrong |
| bands | `cascade.rs:162`, `rolling_floor.rs:140` | quarters of t | derived from t only, no rank |
| preheat | `rolling_floor.rs:93,230` | copy mu/σ/n coarse→fine | two scalars where a survivor set exists |
| early exit | `rolling_floor.rs:239`, `width_16k/search.rs:483` | first Alarm / kth-best tightening | two rules for one idea |
| buckets | `hdr_cascade.rs:200` (`calculate_sweet_spot`), `:625` (`classify_signal`), OGAR `residue_band` | hand `match` tables | no overflow handling, no rollover |
| Prozentrang | doctrine §2, D-BLW-5 design | 16-bucket Fisher-2z histogram | **unbuilt**; `statistics::percentile` is a batch sort |
| bucket rollover | `legacy_outliers.rs:27`, `identity_quad.rs:46` | warnings only | **absent everywhere** |
| Mexican hat | `hdr_cascade.rs:128` | piecewise-linear ramp | Pillar-15 DEFERRED, placeholder `passed=true` |
| entropy | `thought_atoms::normalized_entropy` (ruled home, #1153/#1154); popcount-histogram form at `spectroscopy/features.rs:89` | scattered, consolidating | not wired to thresholds |
| EWA sandwich | `contract::sigma_propagation::ewa_sandwich` ≡ `jc::ewa_sandwich` (#1160) | certified byte-identical | `PILLAR_6_PSD_THRESHOLD` = 0.10 placeholder; σ_step=0.2 denormals in <30 hops |
| mask primitives | `ndarray/src/simd_int_ops.rs:562-1015` | `eq_*_to_mask`, `mask_{and,or,andnot,ternlog}[_assign]` | complete for this plan |
| ownership shape | OGAR `BasinCodebookBuilder::seal()` (#295) | build-freeze-no-`&mut` | reusable as-is |
| iteration shape | OGAR `CallMask` inline `[u64;3]` + `set_indices()` (#288) | word scan + `trailing_zeros` | reusable as-is |
| surviving combinator | `ogar_loco::TERNLOG = FnIndex(0x86)` (#296/#298) | value byte = truth table | 0x87–0x8B RETRACTED, never resurrect |

## 2. Architecture entropy — the T0..T3 shape

The membrane doctrine (`.claude/knowledge/membrane-tiers.md`) says a tier may
only know the vocabulary of the membrane beneath it, crossing by NAME. Applied
here, the histogram collapses vocabulary at every tier. "Entropy" below is the
count of distinct concepts a tier must name; the plan's goal is to lower it
at every membrane without losing a reading.

### T0 — substrate (bytes, lanes)
- Owns: `u64` words, `vpternlogq`, popcount, the Lance version stamp.
- Knows nothing of bands, ranks, or thresholds.
- Entropy before: same. After: same. T0 is already minimal; nothing lands here.

### T1 — primitive (`ndarray::simd`, `lgj-abi/kernels.rs`)
- Owns today: `mask_*`, `eq_*_to_mask`, `ternlog::{AND3,…}`, `masked_strided_group_sum`.
- Adds (D-NXG-2): `mask_bucket::<IMM=0x10>(m_i, m_prev, dst)` is NOT a new
  primitive; it is `mask_ternlog::<AND_ANDNOT2>` by name. The only genuinely new
  T1 op is `popcount_words(&[u64]) -> u64` if it is not already exposed as a
  named primitive (harvest did not find one under that name; verify before
  minting). Everything else composes.
- Adds (D-NXG-3): a `bisect_column_by_mask(col: &[u16], within: &[u64], target_popcount) -> boundary`
  partial-popcount bisection. This is the one primitive that reads a value
  column, and it must stay a T1 NAME so T2 never spells it out of a sort.
  *(2026-09-05: shipped as the private bisection inside `NestedBands::split`,
  NOT as a T1 name — it is a gather-and-popcount loop, not a lane op; see §6
  and E-NXG-21. The T1 claim above is the original proposal, kept for the record.)*
- Entropy: +1 or +2 names. Every T2 concept below is expressible in the existing
  eight immediates plus these.

### T2 — selection (`where`/`hop`/`plan_eval`, the slab cache)
- Owns: `NestedBands { boundaries: [u16; B], masks: [MaskHandle; B], popcounts: [u64; B], version }`,
  sealed per `BasinCodebookBuilder::seal()` shape (D-NXG-1).
- Replaces, by name, four T2 concepts with one:
  `Cascade::Band`, `FloorBand`, `QualityTracker` buckets, `residue_band` are
  all `bucket_index(row) = partition_point(masks, row)`.
- Rank (D-NXG-4): `prozentrang(row) = bucket_index(row)` scaled to `[0, B)`.
  Shape = `popcounts`. This IS the doctrine's `shape × rank`, computed, not
  designed.
- Rollover (D-NXG-5): when `popcounts[i] − popcounts[i−1] > budget`, call
  `bisect_column_by_mask` inside bucket i, insert the boundary, mint the new
  mask under the current version. Old masks are never rewritten; the old
  `NestedBands` stays addressable by its version.
- Reject floor (D-NXG-6): `boundary[B−1]` at a chosen rank, not `mu + kσ`. σ
  becomes `RollingFloor::z()` diagnostic output only.
- Preheat (D-NXG-7): `M_fine_domain = M_coarse[weak]`; the fine tier's initial
  `NestedBands` is the coarse one restricted by AND.
- Early exit (D-NXG-8): one rule, `popcount(survivors) ≤ k_wanted`, replacing
  first-Alarm and kth-best tightening.
- Entropy: T2 drops from ~6 threshold vocabularies to 1 (`NestedBands`) plus
  the rank read. The T1/T2 warden's NAMED test passes by construction: every
  op above is a `mask_ternlog` immediate or one of the two T1 names.

### T3 — intent (Java facade, R2IL, planner, consumers)
- Sees: `Rank(u8)`, `Shape([u64; B])`, `Version`. Never a boundary value, never
  a σ, never a mask word (BBB rule: names cross, byte positions do not).
- `WideFieldMask`/`FieldMask` are untouched (operator ruling 2026-09-04: no
  removal, no demotion without permission). The histogram is additive beside
  them.
- Entropy: T3 gains one name (`Rank`) and loses every threshold constant it
  currently pins (`threshold_l0/l1/l2`, `DEFAULT_EXCITE/INHIBIT`, `k`). Those
  become T2-internal, learned from the slab.

### R2IL / low-code
- `where rank(col) <= 3` is the whole surface. The compiler lowers it to a
  bucket-index compare, which is one AND against a cached mask. This is the
  "Java becomes low-code" consequence the membrane doctrine promised, arriving
  through thresholds rather than through field masks.

## 3. Folding the epiphanies — what falls out that nobody asked for

Each row: the epiphany it folds from → the unforeseen consequence → the
falsifier that would kill it. Ordered by how many rooms ahead it sits.

| rooms | folds from | what falls out | falsifier |
|---|---|---|---|
| 1 | E-NXG-1 + E-NXG-3 | **The 16-bucket histogram is a 16-bit rank code per row.** Store `bucket_index` as a u8 column when the masks are cold; the masks regenerate from it by `eq_u8_to_mask` in one sweep. Cold storage = 64 KB for 65 536 rows, not 128 KB. | regenerate masks from the u8 column and assert bit-equality with the sealed masks |
| 2 | E-NXG-9 | ⊘ **REGRADED 2026-09-05 (E-NXG-19): entropy LAGS the budget test by 5 steps on a real shift.** The row claimed entropy triggers rollover earlier; measured, the max-bucket extremum leads the global average. Entropy is the did-the-split-help read (0.834 → 0.912 across one split), not the is-a-split-due timer. | measured: budget step 16, entropy step 21 of 24 — PROBE-NXG-ROLL-1 |
| 3 | E-NXG-4 + E-NXG-11 | ⊘ **REGRADED 2026-09-05: midpoint σ over-reads 12 % on quantile buckets** (E-NXG-17, heavy top bucket) — needs per-bucket means. The stronger result came from the other side: `k` does not name a rate at all and at k=3 a real column's floor is unreachable (E-NXG-20), so σ is demoted regardless of how well it is recovered. | per-bucket-mean variant, unmeasured |
| 4 | E-NXG-7 + E-NXG-8 | **The eight named immediates are a complete cognitive ISA over masks.** AND3 narrow, AND_ANDNOT2 bucket/known-false, MAJ3 quorum, XOR3 disagreement, OR2_AND gated union. A "thinking style" at the mask level is a sequence of immediates, i.e. a byte string, i.e. a `Vocabulary` entry. This is `TERNLOG(0x86)` with its value byte read as a program. | any style in `contract::thinking` that cannot be expressed as ≤ 8 immediates over ≤ 3 masks |
| 5 | E-NXG-12 | **Proprioception is a histogram of histograms.** Seven anchors × 16 buckets = a 7×16 popcount matrix per window. State = the row with maximum mass; drive_ratio thresholds = two boundaries in that row's `NestedBands`. The 11-dim vector and nearest-anchor distance disappear. | a window where the vector classifier and the histogram classifier disagree, resolved by which one the operator-ruled anchor set calls correct |
| 6 | E-NXG-6 + E-NXG-2 | **Top-k and alarm are the same query.** Search asks "survivors ≤ k"; monitoring asks "survivors ≤ floor". The rolling floor IS a top-k with k = floor. `perturbation-sim`'s early-warning and `holograph`'s search share one T2 function. | one `NestedBands` driving both `stack_early_exit` and `width_16k::search` with identical exits on a shared fixture |
| 7 | E-NXG-14 | **The narrowing trajectory is a path; sigker compares trajectories.** Popcounts per tier = a length-4 path in ℝ⁴. `signature_kernel_pde` on two such paths is an order-sensitive "did these two queries think alike" scalar. Cheap enough (T=4) to run per query pair. Feeds the prefetch planner (E-NXG-*, prefetch frontier): prefer frontier masks whose trajectory signature is close to a hot one. | a pair of queries with identical final survivor sets but different tier trajectories must score < 1.0; identical trajectories = 1.0 |
| 8 | E-NXG-10 | **The Mexican hat is two boundaries; κ is a rank ratio; Pillar-15 certifies on the histogram.** DoG unimodality on a discrete 16-bucket histogram is a finite check (one local max, one annular min) that runs in the T2 seal, not a deferred continuous pillar. Pillar-15 can activate without `dog_eval`. | the seal must refuse a `NestedBands` whose excite/inhibit ratio leaves κ ∈ [1.5, 3.0] |
| 9 | E-NXG-13 | **A basin is a band, so the zero ladder is a missing mask.** `resolve(0) → None` becomes "no mask at this key"; the slab cache and OGAR's `BasinCodebook` are the same sealed object under two names. One `seal()`; one owner; version-keyed. | a basin lookup and a band lookup for the same `(classid, version, idx)` return the same handle |
| 10 | E-NXG-1 + version keying | **Time travel is free.** Because every `NestedBands` is version-keyed and never rewritten, "what was the rank of row r at version v" is a lookup, not a recomputation. The Prozentrang gains a temporal axis; the payload law's "frozen at V₀" is literally the key. | rank at V₀ read after 10 rollovers equals the rank sealed at V₀ |
| 11 | E-NXG-8 XOR3 | **Cross-version surprise is one instruction.** `M_v ⊕ M_{v−1}` per band = which rows changed rank between versions; its popcount is a drift meter with no σ. `ShiftAlert` becomes `popcount(xor) > budget`. | a version pair with identical masks yields zero drift; a rewritten column yields drift proportional to rows moved |
| 12 | rooms 4 + 11 | **Learning is a diff of programs.** If a style is a byte string of immediates (room 4) and version drift is a mask XOR (room 11), then "which style reduced drift most" is a scan over ≤ 256 byte strings against one XOR mask. NARS-style self-reinforcement without a model: keep the program whose survivors moved least. | two styles applied to the same version pair; the one with lower XOR popcount must be the one the existing NARS revision would also prefer, on the fixtures where NARS has a verdict |
| 13 | E-NXG-3 rollover | **Buckets can merge, not only split.** When two adjacent buckets both fall under budget/entropy, drop the boundary between them; the merged mask is `M_i ∨ M_{i+1}` (already nested, so just `M_{i+1}`). The histogram breathes with the distribution. B is not fixed at 16; it is bounded by the L2 budget. | on a collapsing distribution, B must fall; on a widening one, rise; never exceed the slab budget |
| 14 | E-NXG-5 + room 1 | **Preheat across mailboxes is a u8 column copy.** Because the cold form is a rank column, a sibling mailbox inherits a histogram by copying 64 KB and re-minting masks under its own version. No shared mutable slab, no baton, one-writer preserved. | two mailboxes seeded from the same rank column produce identical `NestedBands` under different versions |
| 15 | rooms 5 + 12 | **Proprioceptive drift is stylistic drift.** If state is a histogram row and style is an immediate program, then "the agent's state changed" and "the agent's thinking changed" are XORs on the same substrate. The MUL layer's homeostasis gate reads one popcount. | a held-constant style over a moving state must show state drift with zero style drift, and vice versa |
| 16 | E-NXG-2 + CallMask | **`set_indices()` over a bucket mask is a rank-ordered iterator.** Walking `M_i ∧ ¬M_{i−1}` for i = 0.. yields rows in Prozentrang order without a sort. Top-k is the first k yields. `partition_point` insertion disappears. | iteration order equals stable sort order by distance, within-bucket order unspecified and asserted so |
| 17 | rooms 10 + 16 | **A time-ordered rank walk is episodic recall.** Iterate buckets at version v, then v+1: the rows that appear earlier in the later walk are "what came into focus". This is the ±5 window of the Markov trajectory expressed as two mask walks. | on a synthetic stream where focus is known, the walk-diff must recover it |
| 18 | E-NXG-7 retraction | **Named epistemic calls come back as *derived* reads, never mints.** INFO_GAIN = `H(shape_v) − H(shape_{v−1})`; SIGMA_TENSION = the sandwich-propagated Σ (room 3); STANCE_ENTROPY = `H` of the 7×16 proprioception matrix (room 5). All three exist without a FnIndex, satisfying #298's family-separation ruling because each reads exactly one family's masks. | any derived read that has to AND masks from two semantic families is a violation and must be rejected at the seal |
| 19 | membrane doctrine | **`where` compiles to a rank compare; the Java facade never sees a threshold again.** Every `threshold_l0/l1/l2`, `k`, `DEFAULT_EXCITE` constant currently pinned above T2 is deleted from T3 signatures (additively: new signatures beside old, per the no-removal ruling). | `ApiSurfaceTest`-style reflective fence: no public T3 signature carries a numeric threshold parameter |
| 20 | rooms 4 + 19 | **R2IL gains a `RANK` macro, not a threshold macro.** The R2IL vocabulary (#285, "execute-never-convert") lowers `rank(col) <= n` to `CallMask` AND; the interpreter never sees a distance. | a lifted R2IL program that compares a distance directly is rejected by the vocabulary |
| 21 | room 13 + palette FULL | **Bands are the palette's growth path.** #1126 says the 256-entry palette is FULL. A `NestedBands` per classid is 16 entries per classid living in the slab, not in the palette; the 256:256-by-classid ruling (#1130) already says the classid swaps the whole palette. Band index = a per-classid sub-palette that costs no palette slots. | zero new palette lanes minted by this plan, const-asserted |
| 22 | room 2 + room 11 | **Entropy of the XOR is a novelty meter.** `H(popcounts(M_v ⊕ M_{v−1}))` is high when change is spread across ranks, low when concentrated. Concentrated change = one basin moved = an event; spread change = noise or drift. This is the Belichtungsmesser's `adaptive_resolution(query_entropy, corpus_cv)` match table replaced by two numbers. | on injected single-basin change, XOR entropy must fall; on uniform noise, rise |
| 23 | E-NXG-16 | **Correlation between bands is measurable from the slab.** `popcount(M_a ∧ M_b) / min(popcount)` for two predicates is their overlap; the prefetch planner's independence assumption is checked, not assumed, by one AND. Store the overlap matrix (B×B u16) beside the histogram. | greedy prefetch order with the overlap correction must beat the naive order on a fixture with known correlated predicates |
| 24 | room 23 + MAJ3 | **Quorum over correlated bands is a MAJ3 with a correction term.** Three predicates that overlap heavily vote as one; the overlap matrix tells the quorum how many independent votes it actually has. `jc::quorum::QuorumLevel` gets a mask-native input. | three identical masks must yield quorum 1, not 3 |
| 25 | rooms 12 + 24 | **Self-reinforcement without a model, with humility.** The style whose survivors moved least wins (room 12) only if its votes were independent (room 24). The φ⁻¹ ceiling from the free-energy doctrine has a mask reading: never let a program win on fewer than `1/φ` independent votes. | a style that wins only via correlated bands must be blocked by the independence floor |
| 26 | everything above | **The whole stack is one sealed object per (classid, version): `NestedBands` + overlap matrix + rank column.** Search, monitoring, rank, basin, proprioception, drift, novelty, quorum, style selection, episodic recall and time travel are reads of it. There is no second object. | any of the above requiring state outside this object is a plan defect |
| 27 | room 26 | **The object is the Think struct's `trajectory` field, read as masks instead of `Vsa16kF32`.** CLAUDE.md's Think struct carries a trajectory; the 2026-07-10 ruling moved the Markov trajectory off VSA onto the temporal stream. The sealed histogram IS a temporal-stream projection (versions = time). Think.trajectory becomes a `&NestedBands`, zero-copy, one owner. VSA keeps its ≤32-item niche (I-VSA-IDENTITIES); nothing else changes. | the four VSA tests (register laziness, bundle size, orthogonality, cleanup codebook) all answer "no" for the trajectory field, confirming it was never a VSA workload |

## 4. What is NOT free (carried from the consolidation)

- Independence assumption in popcount selectivity (room 23 is the fix, unmeasured).
- 16 × 8 KB per histogram per version hot; room 1's u8 column is the cold form.
- Rollover bisection reads the distance column (the one non-mask read).
- `PILLAR_6_PSD_THRESHOLD` 0.10 placeholder blocks room 3 in production.
- `columnar_hop_bench` is the only bench in scope; every row above needs its own.
- No repository code is touched by this plan. It is a proposal with falsifiers.

## 5. Sequencing (probe-first, per truth-architect)

1. D-NXG-1 seal shape + D-NXG-2/3 T1 names (edit-only spec, warden pre-spawn).
2. PROBE-NXG-HIST-1: build one `NestedBands` from a real facet column; assert
   bucket masks are disjoint, nested, and sum to N; assert rank = partition
   point (rooms 0, 16).
   **GREEN 2026-09-05** on a real audio column (94 572 rows), not a facet
   column — `crates/lance-graph-planner/examples/probe_nxg_hist_1.rs`,
   E-NXG-17. Room 3 regraded there (midpoint σ over-reads 12 % on quantile
   buckets).
3. PROBE-NXG-ROLL-1: rollover on a bimodal column; entropy-triggered vs
   budget-triggered split order (room 2, D-NXG-5).
   **GREEN 2026-09-05 with C3 RESTATED** — `probe_nxg_roll_1.rs`. Budget
   LEADS entropy by 5 steps, the reverse of room 2 (E-NXG-19). The first run
   also falsified the ladder itself: the top band must be the universe or rows
   above it are lost silently (E-NXG-18) — corrected and asserted against.
4. PROBE-NXG-FLOOR-1: rank-derived floor vs `mu + 3σ` on the HHTL tiers
   fixture; measure false-alarm rate under Jirak-rate tolerance (D-NXG-6).
   **GREEN 2026-09-05 with C1+C2 RESTATED** — `probe_nxg_floor_1.rs`, on three
   real columns rather than an HHTL fixture (none is on disk). `k` does not name
   a rate; at k=3 one column's floor is unreachable; the rank floor is the best
   ACHIEVABLE boundary, exact only where the column has no ties (E-NXG-20).
5. Only after 2–4 are green: rooms 4–8. Rooms 9–27 stay PROPOSAL until then.

## 6. D-id table

| D-id | deliverable | status |
|---|---|---|
| D-NXG-1 | `NestedBands` sealed shape (T2), version-keyed, one owner | Shipped 2026-09-05 (`planner/src/nested_bands.rs`, E-NXG-21) |
| D-NXG-2 | T1 name audit: `popcount_words` present or minted; bucket = `mask_ternlog::<0x10>` by name | Queued |
| D-NXG-3 | T1 `bisect_column_by_mask` partial-popcount bisection | Lives in `NestedBands::split`; not promoted to T1 (not a lane op, E-NXG-21) |
| D-NXG-4 | Prozentrang = bucket index; `shape × rank` computed from the slab | Shipped 2026-09-05 into the D-BLW-5 payload (E-NXG-22) |
| D-NXG-5 | rollover: split on budget or entropy, merge on collapse, never rewrite | Shipped 2026-09-05 — both arms, budget leads (E-NXG-19/21) |
| D-NXG-6 | rank-derived reject floor; σ demoted to diagnostic | Queued |
| D-NXG-7 | preheat by mask inheritance | Queued |
| D-NXG-8 | one early-exit rule for search and alarm | Queued |
| D-NXG-9 | histogram entropy via `thought_atoms::normalized_entropy` as rollover timer | Queued |
| D-NXG-10 | Mexican hat as two boundaries; Pillar-15 on the histogram | Queued |
| D-NXG-11 | EWA sandwich on histogram-recovered Σ; blocked on Pillar-6 calibration | Blocked |
| D-NXG-12 | overlap matrix + corrected prefetch order + mask-native quorum input | Queued |
