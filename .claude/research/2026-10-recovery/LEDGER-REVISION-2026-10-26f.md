# Ledger revision, 2026-10-26 (f): #1727–#1784, ndarray #368–#376

Maps the merges since revision (e) (`LEDGER-REVISION-2026-10-23e.md`, #1635–#1726)
onto the open list, and folds in one more list: the open items a session
carried from 2026-09-18 (the five "doors", the hop into quack, the 64k slab,
the DuckDB differential, Mississippi Queen, the duplicate 34-recipe vocabulary,
four doc corrections). Revision (e) and its predecessors are not edited; this
file supersedes the rows it names. Board entry:
`.claude/board/entries/2026-10-26-ledger-revision-1727-1784.md`.

**Scope.** lance-graph: 61 first-parent commits, #1727–#1784, `main` at
`18fbe43a`. ndarray: 8 PRs, #368–#376, `master` at `195844bd` (revision (e)
read `37a3b59`). OGAR: none (`68cb29b`, same as revision (e)). Method as in
(e): a script extracted per-PR facts (status, top directory, new files); seven
Haiku runs (five `harvest-scout`, one `quote-harvester`, one re-check scout)
over named search spaces. Every file:line below was read on the main thread
at `18fbe43a` unless marked *navigation only* (a search hit, not read).

## 1. The 2026-09-18 list, revised

| id | item (2026-09-18) | read at `18fbe43a` | now |
|---|---|---|---|
| S1 | **Duplicate 34-recipe loco vocabulary**: `ogar-loco::nars::NarsVocabulary` and `lance-graph-ogar::recipe_vocab::RecipeVocabulary` both claim `DOMAIN_FLOOR + id − 1` | both exist and still disagree: `nars.rs:142-147` arity 2 for `Datapath`, 1 for `Gate`/`Control`, no value codebook; `recipe_vocab.rs:142-144` arity 1 for all, `:171-176` `CB_ATTENTION_FOCUS`. `NarsVocabulary`'s only hits are its own definition. `lance-graph-ogar` is an EXCLUDED crate with its own workspace and pulls `ogar-loco` at `branch = "main"` (`Cargo.toml:140`) | OPEN, latent (same as 09-18). Needs a ruling on which arity is right (Q7) |
| S2 | **The hop into quack** (`src_mask → edge lane → dst_mask`), "PR5" | landed three ways. (a) fk semijoin: `Filter::semijoin` lowers to `MaskOp::Gather` (`quack/src/lib.rs:2296, 2390`); mask-risc's own doc says *"`hop` (PR5) LANDED as `MaskOp::Gather` … and `Terminal::ScatterOrU32`"* (`mask-risc/src/lib.rs:24-27`). (b) Edge-table one-hop in `lance-graph-cypher-quack` (`lib.rs:552` `one_hop`, re-anchored on the edge table), #1376. (c) `OutgoingHop` (#1616), quarantined behind the off-by-default feature `quarantined-gather-hop` (`mask-risc/src/lib.rs:132-137`) | **Reframed.** One hop exists. Two or more hops are refused (`cypher-quack/src/lib.rs:583-587`: *"the walk count needs a foreign-value sum"*). What remains is a lawful multi-hop, which is revision (e)'s RECURSION + GATHER-CODE. Merged into those rows |
| S3 | Phase 7: the 64k slab benchmark | D-QCK-6 *"the 64k vertical slice across five arms"* Shipped (`18c1d85`, STATUS_BOARD:1156); `count_probe` 4855 at 64k on four arms, 0 B per execute (STATUS_BOARD:1199); the lane-fold capstone pins the numbers (605 / 626 / 636 ns); freeze@64k measured (#1276) | **CLOSED.** Its remaining edge is the capstone's points 3 and 6 (tile skip, single-tile probe), §3c |
| S4 | Phase 8: DuckDB semantic differential + critical-falsifier report | semantic differential shipped: `quack/tests/duckdb_differential.rs` + `tests/duckdb/{fixture.rs,oracle.py}` (#1256); PostgreSQL K3 / NULL parity (#1668, #1677, #1682) | **Semantic half CLOSED.** OPEN: a timed DuckDB column (capstone point 5); D-CML-8, the DataFusion support differential, Queued (STATUS_BOARD:1185) |
| S5 | Mississippi Queen (W2): the weighted arm | entry `2026-09-25-window-scheduling-and-two-level-ternlog.md`: *"Mississippi-Queen windows lose; a two-level ternlog fold wins 1.7–1.8× for 4–5 planes"*; Tern2 shipped (#1287). `causal-plane-inventory.md:265`: no implementation | **CLOSED, rejected by measurement.** Tern2 is what replaced it |
| S6 | EPIPHANIES entry (16) missing the `−32` correction | `EPIPHANIES.md` is now a 355-line index (split 2026-09-20); the prose is in `EPIPHANIES-ARCHIVE-2026-09-20.md` | **EXCLUDED.** Historical prose is frozen by the 2026-09-20 closeout rule (`CLAUDE.md` § Closeout); the code (`facet.rs`) carries the correction |
| S7 | `three-prefix-fold-carriers.md` "A wins everywhere" | still `:122`; `:129-131` call depth 5 "a tie, not a win". One later reproduction (three runs per arm) had C ahead of A at depth 5 / identical (C 5.22 / 5.22 / 5.22 against A 5.59 / 5.69 / 5.59) | OPEN, low. The claim needs a host and run count beside it; the sign of the depth-5 result is not settled |
| S8 | ndarray #317: G8's `>> 2` justified by `4⁴` (which justifies `>> 1`), and a falsifier that cannot catch a wrong shift | `masking-ops-state.md:225-237` names both factorizations and says *"Do not cite 4⁴ as the justification for `>> 2`; it justifies `>> 1`"*; `:282-287` adds the third arm with an independently planted divergence | **CLOSED** (corrected 2026-09-18) |
| S9 | `LATEST_STATE.md` says `ruff#115` is still open | `:1017` unchanged; ruff#115 was closed 2026-09-18 and its 8 docstrings landed as ruff `1b26e67` | **EXCLUDED.** Dated append-only entry; recorded here instead of edited |
| S10 | The five doors | (1) the facet as one register: `shared_prefix_tiles` is one `xor` + `tzcnt` over `as_u128()` with the canon/custom tile swap (`facet.rs:327-338`); the 16-byte facet match predicate shipped (#1326, `ternary_match_strided16_to_mask`). (2) `TruthU8` in rail slots: no code (*navigation only*: 0 hits in contract, mask-risc, ndarray simd/hpc). (3) `pack<const L>`: ndarray `simd_masking_ops.rs:4688` `fn pack<T, const L>`, the one predicate tail (#313). (4) generation-keyed mask tile: no code under that name; D-DRF's facet-keyed focus with eviction is the nearest thing (K26). (5) random-access width: the `u16` address is settled by the lane-fold capstone; gather along the `u16` map exists in the D-DRF lab | (1) and (3) **CLOSED**. (2) OPEN, unbuilt. (4) OPEN, partly met by D-DRF (K26). (5) **CLOSED as a ruling**, its executor is lab-only |
| S11 | Decisions 1 / 1b / 3 (alpha channel first; quack as the spine; belief needs proper wiring) | alpha: `contract::alpha` + `spog_tenants` shipped; D-SPOG-W-0 / D-ALPHA-G-0 (#1371, #1372). Quack: the Cypher → Quack seam exists (#1376), `cypher-mask-lowering-v2` is a PROPOSAL. Belief: `RevisionKind` (`contract/src/revision.rs:199`), `BeliefArena` (`planner/src/nars/belief.rs:124`); `Opcode::Revision` pools a duplicated source (TD-NARS-REVISION-UNGUARDED-CAUSAL-EDGE-ISA-1, #1733) | Alpha **landed**. Quack **landed as a seam**, multi-hop open (S2). Belief **merged into** the TD row: "proper wiring" now has a measured defect to start from |
| S12 | The #1240 post-merge board commit | written 2026-09-18 (`d52cc2086`), never merged; by 2026-10-26 it was about 500 PRs behind `main`, and its LATEST_STATE / PR_ARC entries described a state long superseded | **EXCLUDED.** The branch was restarted from `main` without it. Its one live finding, the duplicate vocabulary, is S1 and has a fresh ISSUES row |

## 2. New closed ends (C73–C84)

| id | what closed | evidence |
|---|---|---|
| C73 | **D-DRF lab arc:** immutable DR64 focus lease (P1), focus-vs-direct break-even (P2), focus keyed on its facet (G), staged writeback coalesced per canonical row (W1), ordered same-facet apply (W3), flushed row images handed to the real `CycleWriter` (W4), eviction to a resident budget (W8), MRU vs LRU under churn, gather cost against distinct pages (TLB) | #1762–#1774; entries `2026-10-24-drf-*`, `2026-10-25-drf-*`. TEST-PINNED, lab; the W8 entry says no CI job runs the crate |
| C74 | D-DRAV-AV1: a map-only membership mask over a bound DR64 map | #1776; `2026-10-25-drav-av1-membership.md` |
| C75 | `lance-graph-onnx`: ONNX operator names as aliases of native ndarray functions; `Tanh` / `Exp` as ULP-bounded aliases (`onnx::bounded`) | #1754, #1758; `lance-graph-onnx/src/lib.rs:64-92` re-exports `ndarray::hpc::vml::{vsabs, vssqrt, vsexp, vstanh}` and `ndarray::simd::*_i32_to_mask` |
| C76 | ndarray numerics across every backend: `exp` / `tanh` within 1 ULP (#368, #369), `mul_add` rounds once (#374), saturating narrows down the 512-bit ladder (#372, #373), TFLite / gemmlowp requantization bit-exact (#375), Codec8 phase a (#370), `masked_sum_i32` dense-branch-free / sparse-walk hybrid (#376) | ndarray PR titles; C75 consumes #368/#369 |
| C77 | D-NFR-6: native operations over a `SoaEnvelope` borrow, no table, no runtime frame guard | #1755, `crates/soa-view-probe`, CI job `soa-view-probe.yml` |
| C78 | D-MPF-1: the Stage 1 math polyfill already exists; one expression, oracle, refusals, cost split | #1752, `crates/math-polyfill-probe`, CI job `math-polyfill-probe.yml` |
| C79 | D-DRMC L1: a D-DRMC-5 certificate reused on a fresh frame (X8 partly); the certificate keeps egglog's `ProofStore` + `ProofId`, not its rendering | #1740, #1742 |
| C80 | Thought-git squash: chain-of-origin squash into the most-updated lane, gated on contradiction, blame survives; protected readers of a squash (D-TGA-5D) | #1736, #1739 |
| C81 | EXO-129: a learned 256×256 table does not compose where the codebook is not closed → torque T-2026-10-23-1 KILLED | #1731 |
| C82 | Board ledgers: `BREADCRUMBS.md`, `TORQUE.md`, `GOLDSTAUB.md`, generated `FRUITS.md` (`fruits_index.py`, CI-gated) | #1729 |
| C83 | Board catch-ups through #1758 (arc, INTEGRATION_PLANS, LATEST_STATE, stale In-PR rows) | #1727, #1744, #1759 |
| C84 | bgz-tensor: symmetric signed-i8 probe and Fisher-Z error decomposition (lab, no wire change) | #1757. The silent clamp itself is unchanged (`fisher_z.rs:67`), so ISS-FISHERZ stays open |

**Plans with no code in the window:** D-NAM-0..5 + PZL (#1778–#1784, 7 PRs),
D-ONNX-RELALG-CAP S0–S5 + convergence (#1744–#1751, 7 PRs, two competing
capstones: #1745 is a HOLD on #1744 vs #1750), D-DTM-1 (#1777), D-SAG-6
(#1760), D-DRF-1 (#1756), D-NFR / D-NFC (#1753), D-DRAV-1 (#1775). About 22
plan files against five new code crates.

## 3. Open items, revised

Fruit: **L** low, **H** high, **Heavy**. Tier: O = operator decision, S =
Sonnet, H5 = a Haiku card (named).

### 3a. The six decisions of 2026-10-20: still not executed

Re-checked at `18fbe43a` by one scout; `action.rs:266, 517`,
`lance-graph/Cargo.toml:55-58` and `supervisor.rs:203-212` re-read on the main
thread. No PR in the window touched any of them.

| id | status at `18fbe43a` | fruit | tier |
|---|---|---|---|
| ACTION-GATE | `commit` (`action.rs:266`) unchanged; `commit_via_no_required_role_proceeds_regardless_of_rbac` (`:517`) still pins fail-open; no `Public`/`Restricted` declaration in `action.rs` (the `Marking::Restricted` in `property.rs:782` is a data marking, a different concept) | H | S |
| FLOAT-REFS | `lance-graph-r2il` unconditional (`Cargo.toml:58`), which pulls OGAR `branch = "main"` (`lance-graph-r2il/Cargo.toml:19-20`); no `fold` feature in any manifest | L–H | H5 audit, then S |
| WIRE-50 + ISA-LEGACY | `Maturation3` only in `planner/tests/maturation_ladder.rs`; no `ISA_LEGACY` / `legacy_opcode` / `LegacyRead` / `activation_i4` (*navigation only*) | H | S, one PR (K20) |
| O30 / PROVENANCE | the only `struct SourceRef` is `lance-graph-report/src/plan.rs:284` | Heavy | S after K21 |
| ADMISSION code half | GATHER-CODE classification not started | H | H5 quotes, then S |

### 3b. Carried, unchanged

SUP-DISPATCH (`supervisor.rs:203-212`), SUP-CAPACITY (`:42`, `:389`),
SUP-DRAIN (`drain.rs:97`); DURABLEREF (still doc comments only; `BaseRef`
stand-in now 29 lines in `thought-git-probe`); D-CGW-4 (`counterfactual_verdict`
callers only in `pearl.rs` tests); ISS-BF16-POLYFILL-PER-ELEMENT-ALLOC
(ndarray `simd_ops.rs:640`, `vec![0.0f32; k]` inside the `j` loop, 256 times
per call); ISS-FISHERZ (`fisher_z.rs:67`); TD-NARS-REVISION-UNGUARDED
(`causal-edge/src/isa.rs:197, 214`); ISS-GATHER-HOP-1616 (quarantined, CI tests
it under its feature: `rust-test.yml:130`); RECURSION; D-CMR-5; CERT-COMPILER;
DRMC-PROBES (X6, X7, X16, X18; X8 partly done, C79); FRAME-SEAL-OWNER;
PAL256-G5; THOUGHT-GIT-END; TUPLE-SITES; everything revision (e) marked
*carried, not re-checked*.

### 3c. The lane-fold capstone's ten points (`.grok/board/CLAUDE_LANE_FOLD_CAPSTONE.md`, 2026-10-05)

Not re-read item by item. Navigation over `lance-graph-quack` and
`lance-graph-mask-risc` found no `collapsed_terminals`, no `lane_guard`, no
aperture cap symbol, no `Range` `_under` twin and no timed DuckDB column, and
one `dead_tile` hit in an example (`fold_join_probe.rs`). Read as still open,
in the capstone's own order: collapse rewriter (1), guardrail on the build (2),
tile skip (3), `Pred::Range` as a `u16` span with an `_under` twin (4), timed
DuckDB column (5), single-tile probe (6), K knee as a refusal (8), scattered
extract (9), aperture cap (10). Point 7 is a citation rule, not work.

### 3d. New open items (from #1727–#1784)

| id | item | source | fruit | tier |
|---|---|---|---|---|
| **DRF-ZERO-COPY-EXCEPTION** | D-DRF-1 records a DECISION (2026-10-24): the DR64³ foreign-key cache is *"a scoped exception to the zero-copy law"*. `CLAUDE.md` says *"There are no boundaries"* and has no pointer to the exception | `entries/2026-10-24-d-drf-1-focus-lease-foreign-cache.md:6` | H | O (where it is written), then H5 spec-editor |
| **DRF-NO-CI** | `drf-focus-probe` has no CI job; its tests are TEST-PINNED by hand only | W8 entry, Status line | L | S (workflow file) |
| **DRF-EVICTION** | LRU is the worst policy under a cyclic working set one larger than the budget (30 builds in 30 loads); real thought workloads are not measured | W8 entry, OPEN | H | S (probe) |
| **CAPSTONE-COMPETITION** | two ONNX-RELALG capstones (#1744, #1750) with #1745 as a HOLD, plus D-NAM-0 as a third capstone for the same membrane | #1744, #1745, #1750, #1778 | — | O |
| **HOP-DOC** | `mask-risc/src/lib.rs:126` says the hop *"is PR5's and is declared when it lands"*; `:24-27` in the same file says it LANDED | read | L | H5 spec-editor |
| **ONNX-ALIAS-GAP** | ndarray #372, #373, #375 (saturating narrows, TFLite requantization) have no alias in `lance-graph-onnx` yet; D-NAM-2 plans exactly those leaves | C75, C76, `d-nam-2-onnx-tflite-leaf-dto-v1.md` | L–H | S |

## 4. Connecting the dots

- **K26 — D-DRF already implements two of the lane-fold capstone's open
  points, under different names.** Capstone point 10 asks for a cap on
  distinct apertures (*"a million masks at 8 KB are 8 GB"*); D-DRF-W8's
  `FocusManager::evict_to_budget` is a resident-bytes cap over gathered,
  facet-keyed foci, with disables. Point 9 asks to extract `u16`s only when
  the dead-word fraction is low; ndarray #376 makes `masked_sum_i32` sum
  dense words branch-free and walk sparse ones, and D-DRF's fold calls
  exactly that kernel (P1 entry line 8). Neither document cites the other.
  Linking them turns two open points into "adopt W8's budget as the aperture
  cap" and "measure the #376 crossover on lane workloads".
- **K27 — admission has a fifth instance.** Revision (e)'s K22 listed four
  layers asking "what may be built before execution, under what witness".
  D-DRF-1 adds a fifth and is the first to call itself an *exception*: a
  gathered column (16 KiB per focus) built under a lease over a sealed map,
  with dirty rows written back through `CycleWriter`. That is the strongest
  case yet for one admission vocabulary (witness, lease, promotion refused),
  because otherwise every lab that needs a cache will name its own exception.
- **K28 — the one cross-repo seam in the window that is wired end to end is
  ndarray numerics → `lance-graph-onnx`.** It is pure re-export (an alias
  where the contract is identical, an adapter where it is not, a refusal
  otherwise), so a numeric fix in ndarray reaches the ONNX membrane with no
  code in lance-graph. #375 and #372/#373 are the next aliases (ONNX-ALIAS-GAP).
- **K29 — the hop was never missing; multi-hop is the RECURSION row.** S2's
  "missing leg" is now three implementations, one quarantined. The
  edge-table re-anchor in `cypher-quack` reads each edge row once and needs
  no foreign value. Every refusal beyond one hop names the same missing
  thing (a foreign-value sum), which is revision (e)'s RECURSION driver.
  Item S2 and RECURSION are therefore one item.
- **K30 — the queue grew while planning continued.** Revision (e)'s K19 said
  the six rulings needed execution, not decisions. In the following 61
  commits none was executed, and about 22 plan files were added. Execution,
  not planning, is still the cheapest high-signal work in the repository.

## 5. Fruits

### Low-hanging (Haiku-led where mechanical; verified by the orchestrator)

1. **HOP-DOC** — `spec-editor`, one anchored replacement at `mask-risc/src/lib.rs:126`.
2. **S7** — `spec-editor` adds a host / run-count clause after `three-prefix-fold-carriers.md:131`; the wording comes from the orchestrator.
3. **FLOAT-REFS consumer audit** (carried) — `harvest-scout`: every `lance_graph::fold` import in the workspace and sibling repos.
4. **GATHER-CODE quotes** (carried) — `quote-harvester` over `quack/src/lib.rs:2077, 2092, 2296, 2390`, `mask-risc/src/exec.rs` (the `MaskOp::Gather` arm), `reference.rs:437, 822, 922`.
5. **ROWS-MISSING** (carried, grown) — `census-counter`: every D-id in `INTEGRATION_PLANS.md` and `.claude/plans/` with no STATUS row, now including D-NAM-*, D-ONNX-*, D-DRF-*, D-DRAV-*.
6. **DRF-NO-CI** — one workflow file (Sonnet).

### High-signal (one Sonnet slice each, test first, decision already made)

1. **ACTION-GATE** — flip the pinning test to expect denial; add the declaration; route or deprecate `commit`.
2. **WIRE-50 + ISA-LEGACY** in one PR (K20).
3. **FLOAT-REFS**, after fruit L3.
4. **SUP-CAPACITY → SUP-DRAIN → SUP-DISPATCH**.
5. **BF16 polyfill** in ndarray: transpose B once outside the loops; 256 allocations per call become zero.
6. **Capstone point 1, the collapse rewriter** with its 1,000-sum test, then point 3 (tile skip). The capstone's own order of work.
7. **K26 link**: adopt W8's budget as capstone point 10's aperture cap.
8. **ONNX-ALIAS-GAP**: alias #375 / #372 / #373 in `lance-graph-onnx`.
9. **D-CGW-4** promotion gate.

### Heavy (needs a placement or scope decision first)

1. **K21**: `DurableRef` → `SourceRef` → commit-witness anchor, at D-TGB-7.
2. **RECURSION / multi-hop** under ADMISSION B (K29).
3. **One admission vocabulary** (K22 + K27).
4. **CERT-COMPILER**, **PAL256-G5**, **THOUGHT-GIT-END** (carried).

## 6. Questions for the operator

Q1–Q6 of revision (e) are unanswered and carried. New:

| # | question | why it blocks |
|---|---|---|
| Q7 | S1: which arity is right for a recipe op, 2-for-`Datapath` (`nars.rs`) or 1-for-all (`recipe_vocab.rs`)? | one of the two `impl Vocabulary` must delegate to the other; the change spans OGAR and lance-graph |
| Q8 | Where is the D-DRF-1 zero-copy exception written, beside "There are no boundaries" in `CLAUDE.md` or only in its entry? | DRF-ZERO-COPY-EXCEPTION; K27 |
| Q9 | Which capstone governs the ONNX / native-method membrane: D-ONNX-RELALG-CAP (#1744 or #1750) or D-NAM-0? | CAPSTONE-COMPETITION |

## 7. Agent ledger (Haiku harvest tier)

| card | brief | result | checked by the orchestrator | wrong | missed |
|---|---|---|---|---|---|
| harvest-scout | duplicate recipe vocabulary | DONE | `nars.rs:128-175`, `recipe_vocab.rs:134-185` read | 0 | 0 known |
| harvest-scout | quack hop surface | DONE | `mask-risc/src/lib.rs:18-30, 108-150`, `cypher-quack/src/lib.rs:545-600` read | 0 | did not search `lance-graph-sap` / dir-sim (outside its space, stated) |
| quote-harvester | four doc corrections | DONE | quotes used as given; `LATEST_STATE.md` read only at `:1005-1024` (file too large) | 0 | 0 |
| harvest-scout | 64k / differential / Mississippi | PARTIAL | STATUS rows 1156, 1185, 1199 cited from its quotes | 0 | 36 STATUS lines truncated by the tool; plan Status lines not extracted |
| harvest-scout | five doors | DONE | `facet.rs:300-340`, ndarray `simd_masking_ops.rs:4676-4700` read | 0 | zero-hit surfaces reported as zero hits, not read |
| harvest-scout | alpha / belief / Cypher plan | PARTIAL | plan Status lines used as given | 0 | `belief-abi-restoration-v1` Status not found; four ISSUES lines truncated |
| harvest-scout | revision (e) re-check | DONE | `action.rs:262-272, 514-520`, `lance-graph/Cargo.toml:55-59`, `supervisor.rs:203-212` read | 0 | 0 known |
