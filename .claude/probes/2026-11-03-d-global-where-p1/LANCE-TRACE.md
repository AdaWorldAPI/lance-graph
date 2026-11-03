# Lance 13 selective-read trace — D-GLOBAL-WHERE vs established contracts

> Paths under `probe/`, `duckdb/` and `results/` refer to the research branch [`research/d-global-where-p2`](https://github.com/AdaWorldAPI/lance-graph/tree/research/d-global-where-p2/.claude/probes/2026-11-03-d-global-where-p1) (commit `1422b2b92`); they are not on `main`.

Repo `/home/user/lg-p1`, branch `ccr-85acaff0-oszq40-global-where-p1` (HEAD `3fb41149d`). Read-only. No cargo run. Labels: **VERIFIED-IN-CODE** (read first-hand, file:line), **VERIFIED-IN-DOC** (read in a committed plan/entry/PR body), **HYPOTHESIS**, **OPEN**.

## Step 1 — established contracts

| PR | established | evidence |
|---|---|---|
| #1489 (D-TELEPORT-2) | Lance 13's own scanner/index paths prune source reads; "No new primitive is needed"; remaining seam = "a cost choice between full scan and index take". Shuffled 50 % B-tree is ~9× slower than full scan (scattered take). Index-only COUNT reads no data columns. | PR body (MCP); `crates/lance-graph-benches/examples/lance_source_probe.rs:1-41` |
| #1573 (D-HOLO H14c) | `with_row_addr_prefilter` reads; `CoordMap` (coord→`_rowid` at one version) + `ReadError::StaleMapping`. Uniform 1 %: bytes 1.00×, rows scanned 692/65,536; clustered 1 %: 0.24×; MaterializationStyle changed bytes in 0/120 cells. Stale map → silent omission; stable ids survive update+compaction, not overwrite. | PR body; `crates/holo-lance-masked-io-probe/src/main.rs:146-218`; `.claude/board/entries/2026-10-15-d-holo-h14c-thought-first.md:17-19` |
| #1575 (D-HOLO H14d) | `logic → address mask R (mask-risc) → with_row_addr_prefilter(R) → terminal`. Rows scanned == \|R\| in every cell. Bytes: uniform 1 % 1.00, clustered 1 % 0.19, clustered 10 % 0.99, apertures 0.01–0.12. "Bytes are page-bound, so the saving needs survivors that are page-local." | PR body; `crates/holo-h14d-probe/src/main.rs:449-533,752-770` |
| #1633 (D-SRAM-2) | Stage 3 `lance_store` reads each side under a relalg-built mask; Arm B `addressed` = version-attested ordered lane (`SealedFacetLane::attest_sorted`), ranges via `seek_from`, `insert_range` masks, refusals `Stale`/`OtherDataset`/`NotSorted`/`NotContiguous`/`NotLeadingColumn`. Arm B reads exactly `3k` rows; 2.5–12× faster ≤1 % centers; full scan wins from 10 %. | PR body; `crates/crepe-plan-probe/src/lance_store.rs`; `crates/crepe-plan-probe/src/addressed.rs:1-330` |
| #1788 (D-DQM crosswalk, local `10291a8cc`) | Coordinate domains are four spaces; exactly one version-checked coord→`_rowid` authority required; S-INGRESS open. | `.claude/plans/duckdb-dr-quack-machine-room-reuse-v1.md:146-330` |
| #1789 (D-DQM P1, local `ab1f3a83f`) | Chunk-touched predictor FAILS (8.8 % / 16.4 % cells > 0.10; clustered 1 % predicted 0.44 vs 0.24/0.19 measured; over-predicts 59/73). No estimator proposed. | PR body (MCP) |

### Verbatim decisions (#1788, plan file)

- **S-INGRESS** (plan `:256-260`): "**Copy status of the handoff (seam S-INGRESS, OPEN, operator decision).** The words → `RowAddrTreeMap` step re-encodes an elected `Keep` set into a container Lance owns, and Lance copies it twice more. The waben ruling (`waben-fold-execution-loop-v1.md:1967-1979`, "where a set IS elected … the container is a free choice") covers our own carriers; whether it covers a foreign-owned ingress container is not ruled. Until the operator names the seam (as D-DRF-1 did for its cache), nothing on this path is called zero-copy or a fold."
- **A.0-1 correction** (plan `:158-162`): "Building a Lance `RowAddrMask` is an **ingress copy outside the fold** … Nothing on that path may be called zero-copy or a fold (see A.3, seam S-INGRESS)."
- **Single authority** (plan `:182-186`): "`DR ordinal ≠ _rowid ≠ _rowaddr ≠ output position.` No code today composes `ord` with any `_rowid` table (closed search space `crates/**/*.rs`). There must be exactly one version-checked coordinate→`_rowid` authority; today there are two lab ones with different witnesses (`holo-lance-masked-io-probe/src/main.rs:148-156` `CoordMap{version}` + `ReadError::StaleMapping`; DRF `check_seal` on `index_seal`), and the crepe handoff has none (`crepe-plan-probe/src/lance_store.rs:264`, keyed on `_rowaddr`)."
- **Register rows** (plan `:318-320`): ONE coordinate→`_rowid` authority = ADAPT (wiring), duplication risk HIGH ("a third map"); DR ordinal → `_rowid` composition = "look up through both maps (never store `row_ids∘ord` as a third table)", ADAPT gated on S-INGRESS; bulk words → `RowAddrTreeMap` = `insert_fragment` / `from_lsb0_bytes` / `insert_range`, ADAPT gated on S-INGRESS.
- **A.3 headline** (`:254`): "**Logical DR64 density does not predict Lance I/O; chunk locality of the selected `_rowid`s does.**"
- **A.3 "do not rebuild"** (`:228-232`): the API set `with_row_addr_prefilter`, `RowAddrMask::{from_allowed,from_block}`, projection, `filter` refine, `MaterializationStyle`, `scan_stats_callback` is "Already implemented and measured; do **not** rebuild".
- **Timing rule** (`:262-264`): storage results reported as bytes, ranges, requests, rows scanned, not wall time.
- A.4 end-to-end map (`:280-289`): nonresident path = `C_map ord→_rowid via ONE version-checked map [INTEGRATION GAP]` → `C_mask words→RowAddrTreeMap — S-INGRESS [OPEN]` → `C_io with_row_addr_prefilter + projection of demanded columns only (Lance owns)` → re-attach by stored coordinate column. `:311`: "**the integrated DR64 → Lance → Quack query is unmeasured**".

## Step 2 — the caller path, hop by hop

| hop | symbol / location | status |
|---|---|---|
| ClassView → WideFieldMask | `lance-graph-contract/src/class_view.rs:243` `WideFieldMask` (contract type exists) | EXISTS (contract) |
| WideFieldMask → required Lance columns | none found. Closed search: `rg '\.project\('` over `crates/**/*.rs` excluding the five lab probes: every hit is `ClassView::project`/`SubstrateView::project`/math `project` (class_view.rs:1122, causal_witness.rs:685, …), not `Scanner::project`. Production Lance scans hard-code column names: `lance-graph/src/graph/cycle_sink.rs:440,608,1079,1162,1211`, `graph/versioned.rs:443`. | **MISSING EDGE** (VERIFIED by closed grep + read of candidate lines; whole-file reads of cycle_sink/versioned NOT done → PARTIAL for those files) |
| WHERE / DR selection → mask words | Quack `lower` + mask-risc `execute_into` (probe `lib.rs:63-64`); relalg `run_with_program` in `crepe-plan-probe/src/lance_store.rs:233-250` | EXISTS (resident) |
| mask words → version-attested row ids | three different lab authorities: (1) H14c `CoordMap{version, id_of}` + `StaleMapping` (`holo-lance-masked-io-probe/src/main.rs:146-218`); (2) H14d `row_ids` positional `Vec<u64>`, **no version** (`holo-h14d-probe/src/main.rs:448-471`); (3) crepe Arm A `_rowaddr` column read at the same `Arc<Dataset>`, **no explicit witness** (`lance_store.rs:184-205,264-269`); (4) crepe Arm B `SealedFacetLane` attested at `ds.version()` + `uri`, refuses `Stale`/`OtherDataset`/`NotContiguous` (`addressed.rs:101-116,284-309`). DR side: dt64 `RouteVersion`/`Frame` (`lance-graph-dt64/src/dr.rs:153-176`) — a DR route-log witness, not a Lance dataset version; dt64 has no Lance dependency (`lance-graph-dt64/Cargo.toml:10`). | **COORDINATE GAP**: no code composes DR ordinal → `_rowid` (confirms plan `:182`) |
| → Lance `RowAddrMask` (AllowList) | `RowAddrMask::from_allowed` at `lance_store.rs:268`, `addressed.rs:270`, H14c `main.rs:216`, H14d `main.rs:532`, `lance_source_probe.rs:357`. **BlockList (`from_block`) is called nowhere** in `crates/**/*.rs` (closed grep `from_block`, 0 hits). | EXISTS (lab only) |
| → `Scanner::with_row_addr_prefilter` | `lance_store.rs:217`, H14c `main.rs:234`, H14d `main.rs:490`, `lance_source_probe.rs:379` | EXISTS (lab only) |
| → `Scanner::project` (required values only) | `lance_store.rs:191,215` (`c0`/`c1` literal); H14c `main.rs:231` (`coord` + lane names); H14d `main.rs:488`; `lance_source_probe.rs:373-392`. Columns are literal names, never derived from a FieldMask. | EXISTS, but FIELD selection is hard-coded, not ClassView-driven |

Closed search space for the Lance calls: `rg 'with_row_addr_prefilter|RowAddrMask|RowAddrTreeMap|from_allowed|from_block|take_rows|with_fragments|scan_stats_callback'` over `crates/**/*.rs` → hits only in the 5 files above (+ `crepe-plan-probe/src/lib.rs:23` doc). All are lab: `crepe-plan-probe`, `holo-lance-masked-io-probe`, `holo-h14d-probe` are in the root `exclude` list (`Cargo.toml:51` onward, entries at `:55,:137,:206`); `lance-graph-benches` is a member but the hit is an example. `take_rows` / `with_fragments`: 0 hits.

### Field-mask vs row-mask confusion check
- No site passes a WideFieldMask/FieldMask where a row mask is expected, or vice versa (VERIFIED in the files read). In the D-GLOBAL-WHERE P1 probe, `WideFieldMask` appears only as `MethodBinding.on/reads` (field positions) in `falsify_p1.rs:199-201` — correctly field-side.
- **Naming hazard (VERIFIED):** `project.rs:1-62` calls property-layout matching a "projection probe", but it projects nothing through Lance or ClassView; it measures packing 64 logical properties into 32 resident i32/u32 lanes. That is a third meaning of "projection" next to `Scanner::project` and `ClassView::project`.
- crepe `lance_store.rs:282-307`: builds the row mask from a full one-column read (`read_column`, no prefilter) — i.e. a full column scan per side per query to produce the row mask, then a second masked scan. Rows→mask is correct domain use (`_rowaddr`), but the mask's input read is unpruned.

## Step 3 — answers

**a) Paths that invoke Lance masked reads (prefilter + projection)** — VERIFIED-IN-CODE, all lab:
1. `crepe-plan-probe::lance_store::read_rows` (`lance_store.rs:209-229`), called from `prepare_lance` (`:274-364`), Arm A.
2. `crepe-plan-probe::addressed::read_side` (`addressed.rs:316-330`) → `read_rows`, Arm B (version-attested, range masks).
3. `holo-lance-masked-io-probe::read_with` / `masked_read` (`main.rs:201-253`).
4. `holo-h14d-probe::read` (`main.rs:482-520`).
5. `lance-graph-benches/examples/lance_source_probe.rs::run` Arm C (`:375-381`).
No production (workspace `src/`) caller of `with_row_addr_prefilter` exists in `crates/**/*.rs`.

**b) Bypasses**
- **The D-GLOBAL-WHERE probes never touch Lance.** Both `Cargo.toml`s depend only on contract, dt64, quack, mask-risc, ndarray (+ `dr-math-binding-probe` in P0) — `.claude/probes/2026-11-03-d-global-where-p1/probe/Cargo.toml:10-15`, `.../d-global-where/probe/Cargo.toml:10-16`. Closed grep for `lance::|Scanner|RowAddr|_rowid|take_rows|Dataset` over both probe trees: 0 hits. Both SPEC.md files: no `drf|SealedMap|Focus|CoordMap|RowAddr|_rowid|lance::|Dataset` hit. Data is generated in memory: P0 REPORT `:18` ("datasets are deterministic (SplitMix64) and are regenerated by `bench`, not committed"); compose.rs `:92-134` (`where_val`, `a_of`, `b_of`, `Graph::edges_of` closures); `fetch` is "edges … from the generator, standing in for a global relation index" (`compose.rs:38-40`) and its doc disclaims "that `fetch` equals a real relation-index read" (`:57-58`). Lane values reach folds via `WorkingSet::gather_lane(frame, source: Fn(u64)->i32)` (`lib.rs:405-416`) — a copy into a new coordinate-ordered `Vec<i32>`, i.e. a GATHER over a closure, not a storage read.
- Inside those probes, shapes the workspace already rules against (VERIFIED-IN-CODE): `discover` extracts mask → `Vec<u64>` ids (`compose.rs:175-179`); `fetch_edges` returns `Vec<(u64,u64)>` tuples (`compose.rs:128-134`); `partition` builds `Vec<Vec<u64>>` (`:206-213`). Per CLAUDE.md these are the tuple/row-id-list shapes; A.2 row 1 marks SelectionVector-style row-id lists INTENTIONALLY ELIMINATED.
- crepe Arm A: full unmasked column scan to build each mask (`lance_store.rs:184-205`, called at `:292,:336`). Unmasked full reads also at `:347-348` when no centers.
- `project.rs:33-41`: packed ordered compares done with a hand-written Rust "extraction loop" because "Quack has no ordered compare over an unsigned lane". The plan's register (`:322`) already names the fix: ndarray `*_u8_to_mask` / `*_u64_to_mask` exist (`simd_masking_ops.rs:3569-3724,3787-3936`, cited by the plan, not re-read here) but have no `Pred` variant — ADAPT (wiring). The probe re-derives the kernel instead.
- No `take`/`take_rows` loops anywhere (0 hits).

**c) DR → Lance coordinate mapping + version witness: MISSING (not reused, not re-derived)**. The DR side (dt64 `MapTable`/`Frame`/`RouteVersion`) and the Lance side (H14c `CoordMap{version}`, crepe `AddressedRelation{uri, lane.version()}`) never meet. dt64 has no Lance dependency. D-GLOBAL-WHERE adds its own `WorkingSet` + `CoordIndex` (arm C "derived index … built FROM `MapTable` views under one frame", `lib.rs:4-9,34-36`) — derived under the DR frame, but with no Lance version, so it is neither of the existing Lance witnesses. HYPOTHESIS: if this `CoordIndex` were later keyed to `_rowid`s it would be exactly the "third map" the register forbids (plan `:318`).

**d) Mask construction frequency** (VERIFIED-IN-CODE):
- crepe Arm A: `row_addr_mask` (`lance_store.rs:264-269`, per-bit `for_each_set` → `Vec` → `from_iter`) once per side **per `prepare_lance` call** (per query), after a full column read.
- crepe Arm B: `range_mask` (`addressed.rs:265-271`, `insert_range` per run) once per side per query; `attest` once per table (cold, reused while `ds.version()` matches, `:284-309`).
- H14c: `CoordMap` built once per dataset (`main.rs:338`); `masked_read` rebuilds the `RowAddrMask` from a coord list on every call (`:215-216`), called 5× per cell and once per lane in the independent strategy (`:366-393`).
- H14d: `row_ids` once per dataset (`main.rs:643`); `row_mask` once per (pattern,d,width,k) cell (`:755`), per-bit enumeration (`:523-533`).
- D-GLOBAL-WHERE: no Lance mask at all.

**e) Bytes attributed to physical layout, not missing filtering** — yes, consistently:
- H14d PR #1575: "Bytes are page-bound, so the saving needs survivors that are page-local."
- Plan A.3 `:251-254`: uniform mask touches a 4,096-row chunk with prob `1 − (1 − d)^4096` (≈1 at 1 %) → "chunk locality of the selected `_rowid`s does" predict I/O.
- `.claude/probes/2026-11-03-d-dr-fold/D-DR-FOLD-MAP.md:47`: "Lance 13 `with_row_addr_prefilter`: a scattered mask cuts rows decoded, not bytes fetched; a page-local mask cuts both (1 % uniform 1.00×, 1 % clustered 0.24×, 0.1 % uniform 0.77×)." — LAB, H14c-4.
- `.claude/probes/2026-11-03-d-dr-fold/REPORT-BENCH.md:129`: "The H14c result stands: a scattered mask cuts rows decoded, not bytes; a clustered one cuts both." (this is the #1945 / D-DR-FOLD quote's source).
- `.claude/board/entries/2026-11-03-d-dr-fold-map.md:5`: "A mask cuts bytes fetched only when presence is page-local. Every recorded `bytes_read` was served from the page cache, not read from disk. The physical read-planning contract is missing and has no owner."
- Rows scanned == |R| in every H14d cell, so filtering is not missing; bytes are layout-bound.
- #1789: chunk-count predictor fails; lance-io merges at 4 KiB on local temp dirs vs 64 KiB cloud default.

**f) Where time goes**
- Storage: not timed by rule (A.3 `:262-264`); bytes from page cache only (dr-fold-map `:5`). #1633: full read 9,810–30,710 µs, Arm B warm 370–38,980 µs, attest 6,700–9,800 µs (quoted from the PR body, not gated).
- DR routing: P0 REPORT `:112` — cold losses on every `multi_*` dataset come from "**addressing**: … `MapTable::locate` per edge endpoint … O(tiles) per id"; P0 entry `:18`: "Routing once and reusing costs 5 µs against 650–970 µs." P1 `comp_both.log:18-28` (N=1M): `route` 480.6 µs → 282,447.8 µs, `fetch` (in-memory generator) 68–14,086 µs, `fold/set` 60–4,358 µs, `discover` 209–641 µs; FAILED-GATE on 6 of 10 rows. Route dominates every case and grows with scatter (route +7,397.6 µs/set scatter vs +215.2 contig, `:42-43`).
- Mask ingress: never timed on the Lance side except `lance_source_probe` `mask_ms` (`:376-378`); A.4 lists `C_mask` as an OPEN seam.
- Fold: ns/word scale in H14d (2–10 ns/word stage 2, ~6 ns/lane load); in P1 compose, fold/set exceeds 1,000 µs only because sets are 50k–200k ids (`comp_both.log:20-28`).

## Step 4 — classification

1. **NOT APPLICABLE (primary)** for the D-GLOBAL-WHERE probes: they never touch Lance (manifests + closed grep, step 3b). They cannot regress or reuse a Lance read path; their numbers say nothing about Lance selective reads. Their costs are DR routing + in-memory generator "fetch".
2. **MISSING WIRING**: (i) DR ordinal → `_rowid` composition (no code; plan `:182`, re-confirmed by dt64's dependency set); (ii) WideFieldMask → `Scanner::project` column set (no caller; all projections literal); (iii) one version-checked authority — still four lab witnesses (H14c `CoordMap`, H14d none, crepe A none, crepe B `SealedFacetLane`). All gated on S-INGRESS / owner decisions D4 (dr-fold-map `:14`).
3. **COORDINATE MISMATCH (latent)**: H14d `row_ids` and crepe Arm A carry no Lance version; the D-GLOBAL-WHERE `CoordIndex` carries a DR frame only. Not an active bug (each is used within one dataset handle), but none can be composed safely across versions.
4. **GENUINE PHYSICAL-I/O LIMITATION**: scattered masks cut rows, not bytes (H14c/H14d, A.3, dr-fold REPORT-BENCH:129). Not a filtering defect; chunk locality decides.
5. **CALLER REGRESSION**: none found. No production caller ever used `with_row_addr_prefilter`, so none regressed. The lab callers from #1573/#1575/#1633 are present and unchanged in this tree.
6. **Re-derivation inside the probe (not Lance)**: `project.rs` extraction loops re-derive ndarray u8/u64 compare kernels (register row `:322`); `gather_lane` re-derives the DRF GATHER (path 3) without DRF's `FocusKey`/`check_seal`; `discover`/`fetch_edges`/`partition` materialize ids and tuples that A.2 row 1 eliminates.

## Step 5 — minimal corrective proposal (prose only, no edits)

1. If D-GLOBAL-WHERE is meant to say anything about storage, it must read through an existing Lance path rather than a generator. The reusable one is crepe-plan-probe Arm B: `addressed::attest` (`addressed.rs:101-116`) once per table, then `ranges`+`plan_read`+`read_side` (`:213-330`), which already carries `ReadPolicy` (the #1489 "cost choice") and the `Stale`/`OtherDataset` refusals. Its mask is `insert_range` per run, the register's bulk path. Use it, do not copy it.
2. The missing edge is DR ordinal → Lance row position: H14d's positional `row_ids` (`holo-h14d-probe/src/main.rs:448-471`) shows the lookup shape, but the authority must be ONE witness that records both the DR `Frame`/`RouteVersion` and the Lance `ds.version()` (+uri) — the register's ADAPT row `:318`, looked up through both maps, never stored as a composed table (`:319`). This is gated on S-INGRESS and owner decision D4; it should not be built by a probe.
3. Field side: map the ClassView/WideFieldMask positions a query demands to the column list passed to `Scanner::project` (today literal at `lance_store.rs:215`). That edge is absent everywhere.
4. Inside the probe: replace `project.rs` extraction loops by adding mask-risc `Pred` variants that call the existing ndarray u8/u64 compare kernels (plan `:322`, ADAPT when a consumer asks — this probe is the consumer); stop measuring `route` per query (P0 REPORT already shows route-once costs 5 µs).
5. Report any Lance arm in bytes / rows scanned / ranges, not wall time (A.3 `:262-264`), and state that page-cache bytes are not disk reads.

## Could not access / not done
- Lance 13 crate source (`scanner.rs:1663`, `mask.rs`, `filtered_read.rs`, `miniblock.rs`): not read; cited from the plan.
- ndarray `simd_masking_ops.rs` lines: cited from the plan, not re-read.
- PR file lists (`get_files`) not fetched; local files read instead.
- PR #1945 not fetched; quote located in local `REPORT-BENCH.md:129` and `D-DR-FOLD-MAP.md:47`.
- Whole-file reads not done for `cycle_sink.rs`, `versioned.rs`, `lance_cache.rs`, `audit.rs`, `lance_vector_search.rs`, `addressed.rs:330-514`, `holo-*` beyond the cited ranges, `where_k.rs`, the P1 `lib.rs` beyond `1-200` and `405-430`. Claims about those files are PARTIAL.
- `drf-focus-probe` source not read; its `SealedMap`/`check_seal` cited from the plan.
