# Engineering parity: DuckDB literature against DR / Quack / mask-risc / Lance 13

Date: 2026-11-03. Lab research, READ-ONLY. This replaces the earlier draft (scope change: literature first).

Grades:
- **VERIFIED-IN-CODE:** I read the file:line myself.
- **CROSSWALK:** taken from the #1788 crosswalk, `.claude/plans/duckdb-dr-quack-machine-room-reuse-v1.md` Appendix A, and not re-read.
- **CLAIMED:** a publication says so; not checked in source.
- **UNVERIFIED:** I could not check it.

DuckDB C++ source is **not on disk** (`selection_vector.hpp` not found). DuckDB symbols are therefore CROSSWALK at @1641475.

## Sources actually read

| # | publication | status | what I read |
|---|---|---|---|
| P1 | DuckDB docs, "Vector" internals (`/internals/vector` redirects to `docs/2.0/internals/vector.html`) | **SHIPPED-FEATURE** (2.0 docs) | the page |
| P2 | "Optimizers: The Low-Key MVP" (blog, 2024-11-14) | **SHIPPED-FEATURE** (blog about the shipped optimizer) | the page |
| P3 | Liu et al., *Selective Late Materialization*, PVLDB 18(11):4616–4628 | **PUBLISHED-RESEARCH-PROTOTYPE** | the PDF, §1–§2.3 and §4.4–§5.1 (pp. 4616–4621) |
| P4 | Qiao & Zhang, *Data Chunk Compaction*, SIGMOD 2025 | **PUBLISHED-RESEARCH-PROTOTYPE** (shipping version UNVERIFIED) | library-page abstract only |

Notes on P3 and P4:
- P3's implementation is the fork `github.com/yhliu918/duckdb@latest` on DuckDB v1.1.0 (p. 4616, §5.1.2).
- Shipped DuckDB does have a *different* late-materialization rule, `optimizer/late_materialization.cpp:159` `TryLateMaterialization` (CROSSWALK A.2 #15). It must not stand in for P3.
- P4's PDF returned 403 from ACM and from the authors' site. Its mechanism details are UNVERIFIED.

## Engineering-assimilation matrix (one row per principle)

### 1. Reusable selection (P1; P3 §2.2)

- **(a) Problem.** Avoid copying survivors after a filter.
- **(b) DuckDB mechanism and lifetime.** DuckDB keeps a selection vector or dictionary over a child vector. P3 §2.2 says the selection vector holds "DataChunk-local row IDs", shared by columns. Its lifetime is **one chunk** (2,048 rows, `STANDARD_VECTOR_SIZE`, P1). P1 does not state ownership; that part is UNVERIFIED.
- **(c) Our equivalent.**
  - Packed `u64` masks plus the sealed `u16` map (CROSSWALK A.2 #1/#2).
  - The DRF lease, lifetime **one version × scope × seal** (CROSSWALK A.1).
  - mask-risc tiles, `TILE_WORDS = 256` (`mask-risc/src/exec.rs:110`).
- **(d) Status.** ALREADY WORKS, and lives longer: it is reused across queries, not rebuilt per chunk. There is no `Vec<row_id>` by design (CROSSWALK #1).
- **(e) Falsifier.** D-DR-FOLD part B (`REPORT-BENCH.md`, "B: amortization"):
  - the split policy reaches near-hot cost at Q≈256 queries per generation;
  - witnessing a generation costs 502–758 µs.
  - If generations churn faster than about 1/Q, the reuse advantage disappears.

### 2. Late materialization (P3)

- **(a) Problem.** Avoid copying a payload through joins and filters when few rows survive. P3 §2.2–2.3: in modern engines the only remaining cost is replacing virtual IDs with values. Plain late materialization (LM) is no longer a win on average (§1).
- **(b) DuckDB mechanism and lifetime.** P3 makes the choice **per attribute, per query, at plan time**. For each attribute it picks one point between the scan and first use (Alg. 1, §3), using a trained cost model of fetch cost plus memory-copy cost. It carries row IDs and `disable_column` until that point. It fetches in batches at row-group granularity (120K rows) (§4.5).
- **(c) Our equivalent.**
  - Lance projection plus `with_row_addr_prefilter` (CROSSWALK A.3).
  - DRF MAP / BORROW / GATHER (CROSSWALK A.4).
  - There is no hash join carrying a row payload (CROSSWALK #14).
- **(d) Status.** INTENTIONALLY DIFFERENT. P3's biggest win, a smaller hash-join payload (§2.2, Fig. 2), has no target here. What remains missing is the **cost-based choice** between BORROW and GATHER and the masked Lance fetch. That choice is today a measured crossover (2 folds, DRF-P2), not a planner decision.
- **(e) Falsifier.**
  - H14c: a uniform 1 % mask reads 1.00× the bytes; a clustered one reads 0.24× (CROSSWALK A.3). This is P3's "out-of-order probing" in storage form.
  - P3 §5: LM wins only at high selectivity or with long payloads.

### 3. Evaluation direction / join order (P2)

- **(a) Problem.** Avoid huge intermediate results.
  - P2: an unoptimized join produced about 5·10¹² intermediate rows; the optimized one ran in 0.769 s.
  - P2: join filter pushdown, i.e. pushing the build side's min/max into the probe scan, gave about 10×.
- **(b) DuckDB mechanism and lifetime.** Join order is chosen **per query** from table statistics. P2 does not describe the build-side rule (UNVERIFIED).
- **(c) Our equivalent.**
  - Frontier masks vs CSR vs a d-anchored gather, all as lab arms (P0 `REPORT.md:115,122`; D-DR-FOLD).
  - There is no chooser in Quack (D-DR-FOLD Recommendation 1: "a lowering that chooses evaluation direction … from presence density").
- **(d) Status.** **MISSING CAPABILITY (strategy, not kernel).** DR does have presence density as a versioned, popcount-cheap fact. That is a better input than a sampled estimate.
- **(e) Falsifiers.**
  - P0, `multi_skewed` W4: DuckDB produced 3,726,779 join rows for an answer of 1,164 (`REPORT.md:99`).
  - D-DR-FOLD: top-down frontier won 54/54 sweep cells against the anchored gather.
  - P0: `csr` loses 2–14× on dense data (`REPORT.md:115,122`). Picking the wrong direction is the cost.

### 4. Short-circuit for EXISTS (P2 filter pushdown and LIMIT; the P2-probe evidence)

- **(a) Problem.** Stop as soon as the answer is known.
- **(b) DuckDB mechanism and lifetime.** EXISTS ran in about 0.9–2.7 ms, flat in K (`results/cw.log:25-51`). This suggests a scan that stops at the first qualifying chunk. **Lifetime: per query.** The mechanism is CLAIMED-from-timing; I did not read the operator.
- **(c) Our equivalent.**
  - `Terminal::Any` (`mask-risc/src/ir.rs:292`).
  - The executor loops over every tile (`exec.rs:1632`) and does `any |= mask_any(..)` with **no break** (`exec.rs:1793-1794`). VERIFIED-IN-CODE: the loop and the OR; I did not trace every Any path.
  - ndarray `mask_any` has no early exit (local `ndarray/src/simd_masking_ops.rs:4864-4870`; this is gap G8, CROSSWALK).
- **(d) Status.** **MISSING WIRING.** It is one break per tile, after the first tile with a set bit. No new capability is needed.
- **(e) Falsifier.** P2 probe, `cw.log`:
  - At K ≥ 8 our `any` costs about the same as our `count` (1.4–3.9 ms), while DuckDB EXISTS costs 0.88–2.7 ms.
  - 17 LOSS rows, e.g. `k24_large_and` at a ratio of 0.36.
  - After the break, `any` should fall with the position of the first hit, not with N.

### 5. Physical-read avoidance (P2 filter pushdown and statistics propagation; CROSSWALK #12; Lance 13)

- **(a) Problem.** Avoid reading blocks or rows that cannot qualify.
- **(b) DuckDB mechanism and lifetime.**
  - Filters move into the scan. DuckDB's own count plan shows `"Projections": ""` with 8 pushed filters (P2-probe `results/where/n1000000/explain/k8_medium_and.txt`).
  - Zone maps are **per row group, persisted** (`RowGroup::CheckZonemap`, CROSSWALK #12).
  - Statistics folding answers `c0<1000 OR c3<5` with no operator at all (`duckdb/bench_where.py:98-101`).
  - P2 never mentions zone maps.
- **(c) Our equivalent.**
  - Lance prunes storage, at a chunk of ≤4,096 values (mini-block) (CROSSWALK A.3).
  - `with_row_addr_prefilter` applies a DR mask to the read.
  - Resident data needs no pruning: it is already in memory.
- **(d) Status.** ALREADY WORKS (storage), with **MISSING WIRING**:
  - one coordinate→`_rowid` authority, plus bulk word→`RowAddrTreeMap` (CROSSWALK A.5; S-INGRESS OPEN);
  - statistics folding for tautologies or contradictions: the case `adv_contradictory_17` lands at PARITY rather than a win (`cw.log:61`).
- **(e) Falsifier.**
  - H14d: an empty R reads 0 B; clustered 1 % reads 0.19×; uniform 1 % reads 1.00× (CROSSWALK A.3).
  - The integrated DR64→Lance→Quack path is **unmeasured**.

### 6. Chunk compaction (P4)

- **(a) Problem.** After a selection or a join, chunks become sparse and vector loops run underfull.
- **(b) DuckDB mechanism and lifetime.** Compaction is triggered per chunk, with a threshold learned at run time. "Logical compaction" in hash joins minimizes data movement (abstract only). The article reports up to 63 % speedup.
- **(c) Our equivalent.**
  - A mask fold costs per word, not per surviving row, so a sparse mask does not run underfull. It costs N/64 words regardless of density.
  - `materialize_rows` is egress only (CROSSWALK #16).
- **(d) Status.** INTENTIONALLY DIFFERENT: the representation removes the problem. The open question is the sparse end. With very few survivors, whole-word passes are wasted: D-DR-FOLD found "A's per-row lane term" and that C wins when "the chain dies early". That remedy is item 3, not compaction.
- **(e) Falsifier.** At very low survivor density, a fold of K predicates costs K·N/64 words while survivors≪N/64. The `tiny` rows in `cw.log` show our count flat in population (e.g. k24 tiny 3.8 ms vs large 4.2 ms). Prediction: compaction would not help that; switching direction (item 3) would.

### Routing (P2-probe composition)

- **Evidence.** The `route` column dominates: for example, 35,914 µs out of a 50,178 µs total at sel=50,000, D=1 (`results/comp_both.log`, N=1M table).
- **DuckDB.** It builds a per-query hash table, which it does not reuse across queries (UNVERIFIED for shipped DuckDB).
- **Ours.**
  - Arm C, the derived sorted `(global, coord)` index with a `StaleIndex` refusal (SPEC.md:38-49).
  - Status: **MISSING WIRING**, not MISSING CAPABILITY.
  - Its reuse lifetime is the frame (one generation), which is longer than DuckDB's single query.

## WHEN vs WHETHER

DuckDB, both shipped and in P3, optimizes **when** a property is recalled. It already avoids **whether** in one case: COUNT on pushed filters projects nothing (`"Projections": ""`). It still has to read every *filter* column on every query.

We can skip the recall entirely where the answer is a fact about presence, not about values:

1. **COUNT / EXISTS / ALL over an already-bound DR mask.** For a predicate that is already a versioned plane, there is no value read at all. `Terminal::Count`/`Any` read a mask only (`ir.rs:290-292`). DRF's MAP-only path holds no value (CROSSWALK A.4, path 1). DuckDB re-scans the filter columns every query (`rows_scanned: 1000000` in the k8 EXPLAIN). The gain lasts for as long as the plane's generation does.
2. **Field-presence questions.** `FieldMask`/`WideFieldMask` are "pure presence bits" (`contract/src/class_view.rs:60-70`, `:243`), so the question "has property p" never touches p's value.
3. **Empty or contradictory demand.** An empty R reads 0 B from Lance (H14d). If an empty mask skipped `with_row_addr_prefilter` (G8), we would never open the storage at all.
4. **DISTINCT over a sealed map.** DISTINCT is the population of the distinct ordinals in the domain mask (CROSSWALK A.1, `membership.rs:103-125`); no value is needed.

We **cannot** skip it:

- **The first evaluation of a value predicate.** The plane has to be computed once; that is the P2 `discover` scan, N-bound by design (`comp_both.log`).
- **SUM / MIN / GROUP BY on a payload.** These need BORROW or GATHER.
- **Any non-resident source.** It needs a Lance masked read with projection, so the cost is bytes per chunk touched (H14c), not density.
- **After a generation change.** Witness or rebuild: 502–758 µs per generation (D-DR-FOLD B).

**Falsifier for WHETHER.** Re-run the 17+ K-group `count` cases with the K predicate planes pre-bound at the same version (a resident plane per criterion). The cost should drop to K·N/64 word ANDs plus a popcount, with no lane reads, and stay flat across repeated queries. DuckDB's cost should stay per query: `rows_scanned` equals N each time. If ours does not stay flat across repeats, the "versioned fact" claim fails.

## Ranked assimilation candidates (producer → missing edge → consumer → test → owner gate)

1. `Terminal::Any` → per-tile early break (`exec.rs:1793`) → Quack EXISTS → `cw.log` K≥8 EXISTS cases, plus a can-stay-silent test on an empty mask → mask-risc owner.
2. Presence density (popcount of resident planes) → choice of evaluation direction in the lowering → Quack multi-hop / W4 → P0 `multi_skewed`/`multi_dense` W4 + the D-DR-FOLD sweep → Quack owner (D-DR-FOLD D1 ruling).
3. P1 Arm C derived index → a frame-scoped binding reused across queries → composition / route stage → `comp_both.log` route column at sel 50k → DRF/D-GLOBAL-WHERE owner.
4. DR mask → one coordinate→`_rowid` authority plus bulk `RowAddrTreeMap` → Lance masked fetch → H14d cells + the integrated path (unmeasured) → Lance-seam owner, gated on the S-INGRESS ruling.
5. Plane statistics (empty/full per tile) → constant-fold of tautologies and contradictions at bind time → Quack → `adv_contradictory_17` (PARITY today) → Quack owner. Rejected as a resident zone map (CROSSWALK #12).

## Draft verdict

**Engineering parity is reached by representation on selection reuse and on compaction.** A versioned DR mask outlives DuckDB's per-chunk selection vector, and word folds remove the problem of underfull chunks.

**It is not reached on strategy.** DuckDB's per-query optimizer chooses join order, pushes filters into scans, folds statistics and short-circuits EXISTS. Every P2 benchmark loss maps to one of these, never to a kernel:
- EXISTS at K≥8 (no break, `exec.rs:1793`);
- W4 wrong-direction arms;
- the route stage.

The assimilation is to feed DR's *already-versioned* facts (presence density, an empty mask, a bound plane) into decisions the lowering does not yet make. DuckDB's representations are not to be copied. P3's per-attribute late materialization is a research prototype on a v1.1.0 fork. Its main payoff, a smaller hash-join payload, has no target here.

## Not verified

- P4's mechanism beyond the abstract.
- The build-side rule and EXISTS operator in DuckDB (no source on disk).
- Whether shipped DuckDB includes P4.
- Every `Terminal::Any` code path, beyond the loop I read.
- The integrated DR→Lance read.
