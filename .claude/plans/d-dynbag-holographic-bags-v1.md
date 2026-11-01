# D-DYNBAG — DR{dynamic bag}: DuckDB's selection vector made explicit, holographic bags as a polyfill, and the PostgreSQL endgame

**Status:** PROPOSED, plan only. No code, crate, ABI or test lands with this plan. Each claim below is labelled:
- **VERIFIED-IN-CODE**, with its location;
- **MEASURED**, with its command;
- **WORKING-MODEL** (the design this plan proposes);
- **HYPOTHESIS** (unproven; carries a falsifier);
- **OPEN** (undecided; carries an owner).

**Date:** 2026-10-31. **Parent work:** D-XLOG-DR64 (#1857), D-XLOG-NATIVE (#1866, #1884), D-DT64-6 Amendments A.5 addenda 7–12 (#1869, #1871, #1883), D-SOC-FENCE (#1884, plan only), the ONNX classid correction (#1870).

**Directives this plan carries (operator, 2026-10-31):**
1. Inside lance-graph and ndarray, a tuple is a description only, the way `WHERE (x, y)` is in Quack.
2. Use the DuckDB / Quack selection idea to make a selection global → local, and run bag operations over DR256 / DR64.
3. "DR{dynamic bag}" is that, plus treating DR as a holographic representation, plus a bag-op polyfill.
4. The endgame is to beat PostgreSQL with Quack + holographic bags + the polyfill. A related, earlier endgame statement: beat Iceberg at its own game.
5. The external Quack surface may need an Abacus-pattern membrane. That idea lives on EXO-50 as a reference only and is not designed here (§12).

---

## 1. Thesis

A query is a **description**: a selection, a join body, a projection, an aggregate. It is stated over global identities and run over local positions.

**Running a query (WORKING-MODEL):**
1. **Select.** The selection is evaluated as masks.
2. **Relabel.** Its survivors get dense local positions through one pointer table. That is DuckDB's selection vector, made explicit as DR's `LocalMap`.
3. **Tile.** The local positions form a tile whose width is chosen from the survivor count (DR64, DR256 or several tiles): **DR{dynamic}**.
4. **Read in a semiring.** Every relation on that tile is read in a semiring chosen per operation: Boolean for presence and `DISTINCT`, natural numbers for bag multiplicity, later provenance or probability. The address stays one; the values have several readings. That is the **holographic** reading.
5. **Name the operation.** Each operation (select, union, intersect, except, project, join, distinct, compose) is one name with one implementation per semiring. That is the **bag-op polyfill**, in the sense that `ndarray::simd` is a polyfill: one name, the realization chosen elsewhere.

No row is ever built after intake.

**The endgame:** a PostgreSQL-compatible surface (SQL in, rows out at the edge only) that answers with that machinery. It must match PostgreSQL's semantics and beat its executor on the queries where folds over addresses beat tuple-at-a-time execution.

---

## 2. Vocabulary (precise, used consistently below)

| Term | Meaning here | Never means |
|---|---|---|
| **tuple** | a *description*: named positions over lanes, read at a local position (lance-graph P0, "a tuple is a zero-copy fold template, never a value") | an owned row, `Vec<(u32, u32)>`, a struct per row |
| **intake** | the one named place where external row-shaped input becomes lanes (columns). Same status as `RowStore::importRows` in lance-graph-java: visible, named, counted | a hidden conversion inside an operator |
| **global id** | a value's identity outside the tile (an XLOG constant, a SQL key, a `NodeGuid`) | a position |
| **local position** | the dense index a survivor gets in its tile, `0..W` | an identity |
| **pointer table** | the local → global table of one tile; the only thing a selection materializes. VERIFIED-IN-CODE: `lance_graph_dt64::dr::LocalMap` (`globals[local]`, plus a sorted global → local index; lookups counted) | a copy of the rows |
| **mask** | one bit per row or per local position, in the `ndarray::simd` word convention (bit `i % 64` of word `i / 64`, tail bits zero) | an index list |
| **selection vector** | DuckDB's term: the indices of the rows that survived a filter. Here it has two forms: the **mask** (bit form, for folds) and the **pointer table** (index form, for relabelling) | a materialized subset of rows |
| **DR64² tile** | one binary relation over two 64-wide local axes: `[u64; 64]`, one bit per cell (row = left key, column = right key). VERIFIED-IN-CODE: `dr_math_binding_probe::relcomp::Rows` | a list of pairs |
| **DR256²** | the same at width 256 | — |
| **DR64³** | the path space of a two-atom chain `(x, y, z)` across two DR64² tiles sharing the y axis. Addressed, never filled: densely it would be 64³ bits = 32 KiB (VERIFIED-IN-CODE: `Extent::dense_bytes`, addendum 10) | a stored cube |
| **DR{dynamic}** | a tile whose width is chosen from the survivor count of its selection (§4) | a fixed 64 |
| **witness** | one full body binding of a rule or join (e.g. `(x, y, z)` for `r(X,Y), s(Y,Z)`) | a stored tuple |
| **support** | the set of answers with at least one witness (Boolean semiring) | a count |
| **multiplicity** | the number of witnesses per answer (natural-number semiring) | a certification or a truth value (§9.4) |
| **semiring K** | the algebra a relation's cells take values in (§5) | a data type |
| **holographic** | one DR address, read in several semirings at once: a bit plane for B, a count lane for ℕ, later annotation lanes | a superposition or bundle (no VSA is involved) |
| **bag-op polyfill** | the set of relational operation names, each with one realization per semiring, chosen by the description's K (§6) | a new evaluator |
| **membrane** | the outer boundary where row-shaped external formats meet descriptions (§12, reference only) | an inner layer |

---

## 3. DuckDB's selection vector, made explicit

### 3.1 What DuckDB does (background)

- **Vectorized execution.** DuckDB runs over `DataChunk`s of up to 2048 rows (`STANDARD_VECTOR_SIZE`).
- **Filters build selection vectors.** A filter does not copy the surviving rows. It writes a `SelectionVector`: an array of row indices into the chunk.
- **Downstream reads through it.** Operators read the chunk through that selection, so a filter costs one index array and no row movement.
- **Joins and aggregates copy.** They read through the selection, then build hash tables or write output chunks, and that is where copies happen.

### 3.2 The same idea, explicit, in this stack (WORKING-MODEL)

| Step | DuckDB | Here | Owner |
|---|---|---|---|
| filter | predicate → `SelectionVector` | Quack `Filter` → mask-risc `Pred` / `MaskOp` → **mask** | Quack describes, mask-risc evaluates |
| survivors as indices | the selection vector itself | **pointer table**: `LocalMap::prepare(width, globals)` over the survivors' global ids | DT64 `dr` |
| downstream operator | reads the chunk through the selection | reads lanes at local positions; every bag operation runs on the **local tile** | mask-risc, `ndarray::simd` |
| join | hash table build and probe (copies keys and payload) | DR²: a tile per relation; DR³: path space through shared axes. Joins are folds or compose words over tiles (§6, §7) | polyfill |
| aggregate | hash aggregate (a table of groups) | `GroupReduce` over local keys: an O(groups) sink, no hash table | mask-risc |
| output | materialized chunks | rows exist only at the edge, through the pointer table, by name | intake / membrane |

### 3.3 What is explicit that DuckDB keeps implicit

1. **Both forms of the selection are first class.** The mask serves folds; the pointer table serves relabelling. DuckDB has only the index form and rebuilds masks when it needs them.
2. **The global → local relabelling is a named object with a cost.** `LocalMap` counts its lookups (VERIFIED-IN-CODE: `LocalMap::lookups`), so a test can show a projection never calls it.
3. **The tile width is a decision, not a constant.** DuckDB's 2048 is a vector size; here the width follows the survivor count (§4).
4. **The relabelling is versioned.** A frame records which coordinate assignment it was sealed under: `LocalMap::digest` → `FrameRevisions::coordinates` (VERIFIED-IN-CODE, `dt64::dr`). A stale tile is refused (`FrameRefusal::CoordinatesChanged`), never silently reused.

### 3.4 Rules

- The pointer table is the only thing a selection materializes. Lanes are read in place.
- A mask never crosses a program boundary as an O(N) intermediate when an O(groups) sink answers the question (the same rule Quack's `lower_group_by_semantic` states: "No O(N) kept-mask crosses a program boundary", VERIFIED-IN-CODE).
- A global id the pointer table does not hold is refused (`MapRefusal::Unmapped`), never dropped.

---

## 4. DR{dynamic}: the tile width chosen from the selection

### 4.1 Width rule (WORKING-MODEL)

| Survivors on the axis | Tile | Basis |
|---|---|---|
| ≤ 64 | one DR64 tile | `Width::W64` (VERIFIED-IN-CODE, `dt64::dr`) |
| 65 – 256 | one DR256 tile | `Width::W256` |
| > 256 | ⌈n / 256⌉ DR256 tiles, under a thin-provisioned reservation | `Axis { tile, tiles }`, `Extent::DR256_GLOBAL_X64` reserves 256 × 256 = 65,536 identity positions; only routed tiles are claimed (`BindStore::claimed`, addendum 11) |

- **Choosing 64 vs 256 is a cost decision.** A DR64 register is 512 B, a DR256 one 8 KiB. A tile that fits in 64 must not pay for 256.
- **Which width wins for a given density is MEASURED, not assumed** (§10).

### 4.2 What exists and what is missing

| Piece | Status |
|---|---|
| widths 64 and 256; one `LocalMap` per tile | VERIFIED-IN-CODE (`dt64::dr::{Width, LocalMap}`; `LocalMap::prepare` refuses more ids than the tile holds) |
| reserving many tiles, claiming few | VERIFIED-IN-CODE (`Axis`, `Extent`, `Claim`, addendum 11) |
| **a global → (tile, local) map across tiles** | **OPEN, gap.** `LocalMap` maps into one tile. A selection of 7,620 survivors (Jina, addendum 11) needs a tile index plus a local position, and the cross-tile operations of §6 need to address both. Owner: DT64 `dr`. |
| choosing the width from the survivor count | **gap**, small: a function from `n` to `Axis` |
| cross-tile folds and compose | **gap** (§7.4) |

### 4.3 Dynamic over time

- **A frame pins the tile layout.** When the population moves, the frame is advanced and re-sealed, and the old tile layout stays replayable at its route version.
- **No in-place resize.** A resize is a new frame, never a mutation of a sealed one (D5 immutability, addendum 12's frame witness).

---

## 5. The holographic reading: one address, several semirings

### 5.1 K-relations (background)

- **The formalism.** Green, Karvounarakis and Tannen, "Provenance Semirings" (PODS 2007): a relation is a function from tuples to a commutative semiring K.
- **The operators are fixed by K.**
  - union adds;
  - join multiplies;
  - projection sums over the projected-away columns;
  - selection multiplies by 0 or 1.
- **The semiring decides what is answered:**
  - K = Boolean gives set semantics;
  - K = ℕ gives bag semantics, PostgreSQL's default;
  - K = polynomials gives provenance;
  - K = [0, 1] gives probabilities.
- **The same formalism runs through probabilistic Datalog.** Scallop (provenance-semiring Datalog) is built on it; XLOG's README reports accuracy parity with Scallop on MNIST-addition.

### 5.2 The readings of one DR address

| K | Value at a cell | Carrier | Holds | Status |
|---|---|---|---|---|
| **B** (∨, ∧) | 0 or 1 | bit plane: a DR64² / DR256² register, masks | presence, `DISTINCT`, support | VERIFIED-IN-CODE (masks everywhere; the DR64 register in `relcomp`) |
| **ℕ** (+, ×) | multiplicity | **count lane**: one integer per local position (1-D), or a count tile (2-D, §7) | bag semantics, `COUNT(*)`, XLOG's `count` | 1-D: VERIFIED-IN-CODE (`GroupReduce{Count}` sinks; the probe's bag count). 2-D: HYPOTHESIS (§7) |
| ℕ[X] (provenance) | which witnesses | annotation lane | why an answer holds | OPEN; not in scope until ℕ is proven |
| [0, 1] (probability) | weight | annotation lane | ProbLog / Scallop style | OPEN; XLOG's probabilistic programs stay refused (D-XLOG-DR64 Gate G) |

**Projection between readings (WORKING-MODEL):**
- **ℕ → B** is `count > 0`: a compare-to-mask.
- **B → ℕ** is 0/1, through the signed-mask idiom.
- **The holographic property:** for every operation, the ℕ result projected to B equals the B result. That is the first falsifier of everything in §6 and §7.

### 5.3 What "holographic" is not

- **Not VSA.** No superposition, no bundling, no `Vsa16kF32`. The readings are separate lanes at one address, not a bundle.
- **Not a CE64 field.** A multiplicity is never written into CausalEdge64's frequency/confidence bits or its Epi5 bits (§9.4).

---

## 6. The bag-op polyfill

**One operation name, one realization per semiring.** The description carries K; the realization is chosen by K, never by the caller. Every row marks its word VERIFIED-IN-CODE (a mask-risc or `ndarray::simd` word that exists) or names the gap.

| Operation | Over B | Word (B) | Over ℕ | Word (ℕ) |
|---|---|---|---|---|
| selection `σ` | mask ∧ predicate | mask-risc `Pred`, `MaskOp::And` | count × 0/1 | the B mask applied to the count lane; **gap**: a masked-select over an i64 lane (`blend` is i32) |
| union | ∨ | `mask_or` | + (UNION ALL) | `checked_binary_i64(I64Op::Add)` (VERIFIED-IN-CODE, `ndarray::simd`, #407) |
| intersect | ∧ | `mask_and` | min | **gap**: elementwise i64 min (mask-risc has `GroupFold::MinI32` only as a group fold) |
| except | ∧ ¬ | `mask_andnot` | truncated subtraction `max(a − b, 0)` | **gap** |
| projection onto a key | scatter ∨ | `Terminal::ScatterOrU32` | scatter + | `Terminal::GroupReduce { key: Lane, fold: Count }` (VERIFIED-IN-CODE). **Correction:** `Terminal::ScatterCountU32` is `COUNT(DISTINCT …)`, held and not a lowering target; it is NOT the bag projection (an earlier answer said it was) |
| join on y, projected to y | forward ∧ backward | `ScatterOrU32` per side + `mask_and` | deg_R(y) × deg_S(y) | `GroupReduce{Count}` per side + `checked_binary_i64(Mul)` (VERIFIED-IN-CODE, the probe's `bag_count`, Gate B differential) |
| **join keeping both ends (x, z)** | Boolean matrix product | lab only: `relcomp::compose` | integer matrix product | **HYPOTHESIS:** `ndarray::simd::matmul_i8_to_i32` (§7) |
| distinct | identity | — | `count > 0` | compare-to-mask (`*_to_mask` family over i64: **gap**; i32 exists) |
| count of a population | popcount | `popcount_batch_u64`, `Terminal::Count` | sum of the lane | `masked_sum_i32` (i32) / **gap** for i64 |
| group aggregate | — | — | `GroupReduce { Count / SumSymI32 / MinI32 / MaxI32 }` | VERIFIED-IN-CODE (mask-risc) |

**Rules (WORKING-MODEL):**
- **Choosing a realization is the polyfill's job, as in `ndarray::simd`.** A caller names an operation and a K; it never names a kernel.
- **Unsupported (operation, K) pairs are typed refusals**, never a fallback to B. Falling back to B is exactly the support-for-bag error the XLOG fixture exposed (support 1, true count 4).
- **New words land backend-first:** in mask-risc or `ndarray::simd`, owned there, never inline in a frontend (D-SOC-FENCE `fold-evaluator`).

---

## 7. ℕ-compose: the many-to-many join as integer matrix multiplication

### 7.1 The mathematics (WORKING-MODEL)

- **Inputs.** R ⊆ X × Y and S ⊆ Y × Z on local tiles. As 0/1 matrices, A is |X| × |Y| and B is |Y| × |Z|.
- **The product counts witnesses.** (A · B)[x][z] = Σ_y A[x][y] · B[y][z] is exactly the number of y with r(x, y) ∧ s(y, z): the ℕ answer of `p(X, Z) :- r(X, Y), s(Y, Z)`, both ends kept.
- **B is its projection.** The Boolean answer (`relcomp::compose`) is `(A · B) > 0`.

### 7.2 The word that may already exist

- VERIFIED-IN-CODE: `ndarray::simd` re-exports `matmul_i8_to_i32` (`src/simd.rs` lines 845 and 863, from `hpc::amx_matmul`) and `gemm_u8_i8`.
- `matmul_i8_to_i32(lhs: ArrayView2<i8>, rhs: ArrayView2<i8>, out: ArrayViewMut2<i32>)` takes an AMX `TDPBUSD` path when `amx_available() && m % 16 == 0 && n % 16 == 0 && k % 64 == 0`, and is otherwise a bit-identical scalar i32 reference.

### 7.3 Exactness bounds

| Case | Bound | Exact? |
|---|---|---|
| one hop, 0/1 inputs, tile width W | each cell ≤ W (64 or 256) | yes, fits i32 |
| one hop on a DR64 tile | m = n = k = 64: meets the AMX alignment | yes |
| a chain of hops (counts feed the next product) | inputs exceed i8 after one hop | **no.** Needs a wider integer product or an overflow refusal per hop. Typed refusal until a word exists |
| bags with input multiplicities > 1 (base relations are sets, §9.2, so only derived inputs) | as for chains | refused, as above |

### 7.4 Zero-copy and tiles

- **The current signature copies.** `matmul_i8_to_i32` packs its inputs (`pack_contig`) and allocates `c = vec![0i32; m * n]` per call (VERIFIED-IN-CODE).
- **A DR64² tile is bit-packed**, so feeding it in means unpacking bits to i8: a copy per tile, not per row.
- **Status: HYPOTHESIS, to measure.** Whether that per-tile cost is acceptable, or whether a bit-packed i1 × i1 → i32 word is needed (popcount of row ∧ column, the classic bit-matrix product), is decided by measurement. If a new word is needed, it lands in `ndarray::simd` (backend-first), never here.
- **Cross-tile compose is a gap.** Tile (i, k) × tile (k, j), summed over k, as in blocked GEMM, needs the multi-tile map of §4.2.

### 7.5 First falsifier (D-DYNBAG-1)

- **Equality:** on the XLOG multiplicity fixture and the 300-case Gate B differential, `matmul_i8_to_i32` over two DR64² tiles equals scalar enumeration of witnesses per (x, z) (oracle 1) and a bag walk of XLOG's RIR (oracle 2).
- **Projection:** `count > 0` equals `relcomp::compose`, bit for bit (the holographic property, §5.2).
- **Disable runs:**
  - swapping the operands fails the test;
  - reading the result as B before counting fails the test.

---

## 8. Layering: one place per concern

Consistent with D-SOC-FENCE §2. Nothing in this plan adds a layer.

| Layer | Owner | Role in this plan | Never |
|---|---|---|---|
| frontend | XLOG compiler, SQL parser, Cypher parser (`lance-graph-cypher-quack`), Mistaber | produces the plan / AST | evaluates |
| intake | one named function per frontend | external rows → lanes, once; base relations as sets | hides inside an operator |
| describe | `lance-graph-relalg` (`Rel` IR, `join_body::is_alpha_acyclic`, proven rewrites, lowering into one mask-risc program per fold) | the tuple description, gaining a semiring parameter K | evaluates |
| surface | `lance-graph-quack` (`Filter`, `GroupBy`, `Filter::sql_where`, `Semijoin`) | what a caller writes, DuckDB-shaped | evaluates ("may build a Program and must never evaluate one") |
| address | `lance_graph_dt64::dr` (`LocalMap`, `Extent`, `FrameStore`) | global → local, tile width, frame witness | computes |
| bind | `lance-graph-dt64` (`BindStore`) | which method a class's step names (polyfill words by position) | computes |
| evaluate | `lance-graph-mask-risc` | the one fold evaluator | is bypassed |
| compute | `ndarray::simd` | the kernels, including matmul for ℕ-compose | is called from a frontend directly for folds |
| edge | membrane (§12) | rows out | lives inside |

---

## 9. Frontends

### 9.1 XLOG (first; already in the lab)

- **Today (VERIFIED-IN-CODE, `crates/xlog-relcomp-probe`, PR #1884):** the admitted two-atom chain's support and bag count run in `src/native.rs` by calling `ndarray::simd` directly. That is a second evaluator, D-SOC-FENCE's first named violation. Facts land as hand-rolled `Vec<(u32, u32)>`.
- **Target (WORKING-MODEL):**
  - intake XLOG facts into lanes, as sets (XLOG's own `load_grouped_facts_into_store` skips repeated rows, VERIFIED-IN-CODE at the pinned revision);
  - express the chain and `admit_count` as relalg/Quack descriptions with K = B or ℕ, and evaluate through mask-risc;
  - ℕ-compose (§7) for `p(X, Z)`;
  - the register path stays as the B differential;
  - the oracles compare by folds, not expanded rows.
- **Effort estimate:** about 1.5–2 days in the probe. Blast radius: the probe only; no workspace crate changes.

### 9.2 Base relations are sets

- The line drawn on PR #1884 for the XLOG case (VERIFIED-IN-CODE, `facts()` dedup, test `a_repeated_base_fact_is_not_a_second_witness`).
- Generalized:

| Case | Rule |
|---|---|
| the same tuple twice in one base relation | one tuple: the relation is a set at intake |
| the same values in two relations | two facts, never merged |
| two different witnesses of one answer | counted: this is ℕ multiplicity |
| one fact in two representations (lanes and a DR² bit plane) | one fact: the bit plane is a reading, never added |
| provenance of a duplicate (two sources asserting one fact) | not a count; belongs to the witness/evidence path (Epi5 contract), in its own lane if ever needed |

- **SQL is the exception.** A SQL base table is a bag (no implicit `DISTINCT`), so SQL intake keeps multiplicity where XLOG intake drops it. The frontend states which; the polyfill never guesses.

### 9.3 SQL / PostgreSQL

**Template (VERIFIED-IN-CODE):** `lance-graph-cypher-quack`. Its own description: "Cypher over tabular inputs lowered to lance-graph-quack … a demand classifier choosing the carrier, and typed refusals for every other shape. No DataFusion on its call path."

**Proposed (WORKING-MODEL):** `sql-quack-probe`, a lab crate with the same shape:
- an upstream SQL parser, read-only;
- a demand classifier: Distinct, Bag or count;
- lowering to relalg/Quack with K chosen from SQL semantics;
- typed refusals.

| SQL | K | Lowering | Status |
|---|---|---|---|
| `SELECT DISTINCT y FROM r JOIN s ON r.y = s.y` | B | §6 join row (B) | words exist |
| `SELECT y FROM r JOIN s …` | ℕ | count lane, rows emitted at the edge with multiplicity | count exists; edge emission is the membrane |
| `SELECT y, COUNT(*) … GROUP BY y` | ℕ | deg × deg | exists (probe) |
| `SELECT r.x, s.z … JOIN …` | ℕ | ℕ-compose | HYPOTHESIS (§7) |
| `WHERE` with NULL | B, three-valued | `Filter::sql_where` | VERIFIED-IN-CODE (Quack) |
| `UNION ALL` / `UNION` | ℕ / B | + / ∨ | words exist (i64 add, `mask_or`) |
| `INTERSECT ALL` / `EXCEPT ALL` | ℕ | min / truncated minus | gaps (§6) |
| `LEFT` / `FULL JOIN` | — | — | refused (needs NULL-extension in the description) |
| `ORDER BY`, `LIMIT` | — | — | **OPEN:** lance-graph's rule is that order belongs to coordinates, not decoded rows. Either a key-ordered projection serves it, or it is a named materialization at the edge |
| `WITH RECURSIVE` | — | semi-naive deltas | OPEN (D-XLOG-DR64-2 shape) |
| `SUM/MIN/MAX` on i32 | ℕ / group folds | `GroupReduce` | exists |
| `SUM` on i64 / numeric, floats | — | — | gaps / refused |

**Constraints:**
- **Not DataFusion.** It is in its grace period (2026-09-05). `lance-graph-callcenter` stays as the existing Supabase-shaped membrane and is not extended for this.
- **The SQL parser is a dependency decision.** The AdaWorldAPI-fork rule applies; whether a fork exists is OPEN (§13).
- **Reference oracles.** Scalar enumeration, plus a real PostgreSQL if one can run in the lab, and DuckDB (which reads the same data and is the shape Quack follows).

### 9.4 Epi5 and presence: what bags never touch

From PR #1884, `tests/epi5_reading.rs`:
- **Multiplicity is not epistemic.** A bag count is never written into CE64 frequency/confidence (bits 24..39) or Epi5 (bits 59..63).
- **If a result is ever written as a CE64 edge:** a base fact is `Direct × Open`, a composed edge with its intermediate retained is `IndirectKnown × Open`, and a join certifies nothing.
- **Absent is not zero.** `raw5 == 0` reads as `Direct × Open`. "Absent" is the class's `FieldMask` presence bit, never a sentinel inside a lane. The same holds for count lanes: an absent cell is not a cell with count 0 unless the description says the population is total.

### 9.5 Cypher, Mistaber, ONNX

- **Cypher.** `lance-graph-cypher-quack` exists (experimental, routed nowhere). Its pattern matches are DR² / DR³ bodies, so ℕ-compose serves path counts.
- **Mistaber.** Its Kripke holdings are a least fixpoint over `relcomp::compose` (D-MIST-NATIVE). It is blocked on a production compose, the same gap as §7.
- **ONNX.** Elementwise (`lance-graph-onnx` aliases `ndarray::simd`), so it is not a bag consumer. It shares the polyfill's B words.

---

## 10. The endgame: beat PostgreSQL (and keep Iceberg on the scoreboard)

### 10.1 What PostgreSQL does, and what we put against it

| PostgreSQL | How it works | Here | Expected outcome (HYPOTHESIS until §10.3 measures it) |
|---|---|---|---|
| heap tuples, row-oriented pages | a row is read whole | lanes; a query reads only the lanes it names | win on narrow selections over wide tables |
| volcano executor (`next()` per tuple), JIT for expressions | per-tuple dispatch | folds over 64-row words, one evaluator | win on scans, filters and aggregates |
| hash join, merge join, nested loop | build a hash table and probe; copies keys and payload | many-to-one: gather, semijoin. Many-to-many: ℕ-compose on tiles | win when tiles are dense enough for matmul; **may lose** on very sparse many-to-many (to measure) |
| HashAggregate / GroupAggregate | hash table of groups | `GroupReduce`, an O(groups) sink | win |
| bag semantics by default | a row per witness | ℕ count lane; rows only at the edge | win when the answer is a count; even when rows are demanded, emission is at the edge only |
| three-valued NULL logic | per-tuple | `Filter::sql_where`, validity planes | parity required |
| B-tree / BRIN indexes | separate structures | the address prefix (classid / HHTL) and DR coordinates are the index | win on point and prefix lookups; to measure |
| MVCC snapshots | xmin/xmax per tuple | Lance versions, plus DT64 route versions | parity required (snapshot reads) |
| concurrent writers, transactions | row locks, WAL | one writer per mailbox, Lance commits | **different model; state it, don't hide it.** Not a win claimed here |
| extensions, ecosystem, wire protocol | mature | membrane (§12) | **PostgreSQL wins today** |

### 10.2 Iceberg (the table-format scoreboard)

- **Pruning.** Iceberg prunes by manifest statistics (min/max per file), which only exclude files that cannot match. The DR address prunes exactly.
- **Deletes.** Iceberg writes delete files; here a delete is a mask (`mask_andnot`).
- **Filtered counts.** Iceberg answers `COUNT(*)` from metadata only without a filter; here filtered and grouped counts are folds.
- **Interop.** Iceberg wins on engine neutrality. The membrane (§12) is the only answer to that.

### 10.3 Scoreboard protocol (gates, not claims)

- **Units.** Per operation in ns; batches at most in µs. Cold costs (parse, compile, intake) are reported in µs and are not gates. A millisecond result is a failed gate, never a number to report (lance-graph P0).
- **Referees.** DuckDB on the same data (it also reads Iceberg). A real PostgreSQL where the lab can run one; otherwise stated as missing, never approximated.
- **Query classes:**

  | Class | Example |
  |---|---|
  | selective filter | point and range |
  | filtered + grouped count | — |
  | two-table bag join count | XLOG/SQL fixture |
  | many-to-many join keeping both ends | ℕ-compose |
  | `UNION ALL` / `DISTINCT` | — |
  | delete then count | — |
  | snapshot read at an older version | — |

- **Loss cases are mandatory.** Every class runs a dense and a sparse variant. A wide-row, full-scan case is included where PostgreSQL is expected to do well. Results are reported even when they go against us.
- **Correctness first.** Every timed query asserts its answer against the referee before its time counts (`query-stage-profiler` rule).

---

## 11. Waves, D-ids and gates

| Wave | D-id | Work | Gate (falsifier) | Depends on | Tier |
|---|---|---|---|---|---|
| 0 | D-DYNBAG-0 | (§17) Fix PR #1884's layering: XLOG intake as lanes (sets), support and bag count through Quack → mask-risc, oracles by folds, cost probe gains native and Quack columns | all existing Gate B / litho tests green; register differential unchanged; disable runs red; `native.rs` no longer calls `ndarray::simd` fold words | operator go-ahead on #1884 | Opus orchestrates; Sonnet implements from spec |
| 1 | D-DYNBAG-1 | ℕ-compose probe: `matmul_i8_to_i32` over two DR64² tiles for `p(X, Z)` | §7.5: equality with both oracles, B projection equals `relcomp::compose`, disable runs red; timed vs `relcomp::compose` in ns | D-DYNBAG-0 | Sonnet; Opus reviews |
| 2 | D-DYNBAG-2 | Semiring parameter in relalg's `Rel` (B, ℕ) and Quack's surface; typed refusal for unsupported (op, K) | B/ℕ projection property holds for every admitted op; an unsupported pair is refused, never a fallback to B | D-DYNBAG-1 | relalg / Quack owners |
| 3 | D-DYNBAG-3 | Missing ℕ words, backend-first: masked select over i64, elementwise min, truncated minus, i64 compare-to-mask, i64 masked sum; and, if §7.4 measures it necessary, a bit-packed i1 × i1 → i32 product | each word with can-fire and can-stay-silent tests in its owner crate | D-DYNBAG-2 | mask-risc / ndarray owners |
| 4 | D-DYNBAG-4 | DR{dynamic}: width from survivor count, multi-tile global → (tile, local) map, cross-tile fold and compose | answers identical across widths 64 / 256 / multi-tile on the same data; frame refusal on a moved layout | D-DYNBAG-1, DT64 owner | DT64 `dr` owner |
| 5 | D-DYNBAG-5 | `sql-quack-probe` (lab), modelled on `lance-graph-cypher-quack`: the §9.3 table, supported rows plus typed refusals | each supported query equals scalar enumeration and the referee; each refused shape names its refusal | D-DYNBAG-2, §13 parser decision | Sonnet; Opus reviews |
| 6 | D-DYNBAG-6 | Scoreboard (§10.3) against DuckDB and, where possible, PostgreSQL | every class has dense and sparse results, correctness asserted, ns units; losses reported | D-DYNBAG-5 | Opus |
| 7 | D-DYNBAG-7 | Second Datalog frontend through the same path (Scallop: provenance semirings), as the test that the polyfill is frontend-neutral | Scallop programs in the admitted shape answer identically to Scallop | D-DYNBAG-2 | later |

**Ordering rules:**
- Nothing past Wave 1 starts before the ℕ-compose falsifier has run. It decides whether the many-to-many join has a home.
- Wave 3 words land in their owner crates, never in a probe.

---

## 12. The membrane (reference only)

On EXO-50, posted 2026-10-31, as an idea to keep in view and not designed here:
- **The question.** Should the external Quack surface get an Abacus-pattern membrane, amortizing at the outer boundary the way `lance-graph-callcenter` does for its Supabase-shaped API?
- **Its open half.** Whether the conversion cost is per call or per frame.

This plan only notes where it would sit: the "edge" row of §8, and the PostgreSQL wire protocol and row emission of §10.1. It does not design it.

---

## 13. Open decisions

| # | Decision | Owner | Blocks |
|---|---|---|---|
| 1 | Go-ahead to fix PR #1884's layering (D-DYNBAG-0) | operator | everything |
| 2 | Is `matmul_i8_to_i32`'s per-tile unpack and allocation acceptable, or is a bit-packed word needed? | ndarray owner, after D-DYNBAG-1 measures | D-DYNBAG-3 |
| 3 | Where K lives in relalg's IR: on the relation, on the operation, or both | relalg owner | D-DYNBAG-2 |
| 4 | The multi-tile map's shape (tile index × local position) and its frame witness | DT64 `dr` owner | D-DYNBAG-4 |
| 5 | SQL parser: AdaWorldAPI fork or none | operator | D-DYNBAG-5 |
| 6 | `ORDER BY` / `LIMIT`: key-ordered projection or named edge materialization | operator | D-DYNBAG-5 |
| 7 | Method identity (`MethodSignature` / `MethodRef` / `MethodKey`) | EXO-50 | binding the polyfill words to classes in production |
| 8 | Whether a real PostgreSQL can run as a referee in the lab | operator | D-DYNBAG-6 |

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| A polyfill layer that starts evaluating (a fourth evaluator) | D-SOC-FENCE `fold-evaluator` and `query-surface` concerns; description layers may build programs, never run them |
| Falling back to B when ℕ is unsupported (support-for-bag) | typed refusal per (op, K); the XLOG fixture as a standing test |
| ℕ-compose loses on sparse tiles | the scoreboard runs sparse variants; a sparse many-to-many may keep a gather-based path; decided by measurement |
| Count overflow in chains | per-hop overflow refusal; i32 bound stated per tile width |
| Claiming a PostgreSQL win from a toy fixture | §10.3: referee, correctness first, dense and sparse, loss cases reported |
| Copy creep at intake (rows kept "for convenience") | one named intake per frontend; D-SOC-FENCE row for hand-rolled tuples |
| Dynamic width turning into in-place resize | resize is a new frame (§4.3) |

---

## 15. What this plan refuses

- No row engine, no volcano iterator, no hash table for joins or groups.
- No DataFusion on the new path.
- No hand-rolled tuple after intake, in production, probes or oracles.
- No direct `ndarray::simd` fold call from a frontend.
- No multiplicity written into CE64 truth or Epi5 bits.
- No VSA in the holographic reading.
- No PostgreSQL win claimed without the §10.3 protocol.

---

## 16. Cross-references

- `.claude/plans/d-xlog-native-v1.md`: XLOG crosswalk, Amendments A (binding, Epi5) and B (execution).
- `.claude/plans/d-soc-fence-v1.md`: one place per concern, checked in CI.
- `.claude/plans/d-dt64-6-routing-bindstore-behavior-v1.md`: A.5 addenda 7–12 (polyfill lens, extent, thin provisioning, the `dr` module).
- `crates/xlog-relcomp-probe` (PR #1884): native path, binding, Epi5 tests.
- `crates/lance-graph-cypher-quack`: the SQL probe's template.
- EXO-50: the membrane reference note (2026-10-31).

---

## 17. D-DYNBAG-0: adopted architecture, actual vs target (2026-10-31, before coding)

**DECISION (operator, 2026-10-31).** XLOG consumes tuples as a logical polyfill over DT64-addressed relational projections. Quack/RelAlg describe the relation; mask-risc evaluates the bound operations; `ndarray::simd` computes only under mask-risc. A tuple is a coordinate binding, never a stored row. XLOG owns no tuple evaluator and builds no intermediate tuple collection. No new architecture, planner, runtime, tensor store or row engine is introduced: the work is wiring what exists, and labelling what does not.

### 17.1 Ownership

| concern | owner | XLOG's part |
|---|---|---|
| description | XLOG RIR (frontend), then Quack `Query` / `GroupBy` | admits a shape; never executes |
| addressing | `lance_graph_dt64::dr`: `LocalMap`, `FrameStore`, `Guards`; one map per axis | none |
| binding | `lance_graph_dt64`: `BindStore`, `MethodBinding`, `MethodRef`, `AlgebraRef`, `ClassView` | names its lens |
| semiring | the bound method (B support vs ℕ multiplicity) | never the frame |
| evaluation | `lance-graph-mask-risc::execute_into` | none |
| compute | `ndarray::simd`, called by mask-risc | none |

### 17.2 Actual data flow (PR #1884, as merged)

```
XLOG compile ─► RIR ─► admit / admit_count (ok)
                         │
facts() ─► Vec<(u32,u32)> (dedup'd)          ← owned tuples after intake
                         │
LocalMap::prepare per call                   ← rebuilt per query
                         │
native.rs ─► ndarray::simd directly          ← D-SOC-FENCE fold-evaluator violation
   eq_u32_to_mask, mask_scatter_or_u32, mask_and,
   masked_group_count_u32, checked_binary_i64
                         │
ids() / nonzero() ─► addressed result

BindStore + FrameStore: only in tests/dto_binding.rs, never on the path above.
```

The admission and the computation are separate. A binding is resolved in a test and then ignored; the counts are computed whether or not it admits. Replacing the ndarray calls one-for-one would keep that separation, so D-DYNBAG-0 closes the chain rather than swapping calls.

### 17.3 Target data flow

```
XLOG compile ─► RIR ─► admit / admit_count
   │
   ├─► intake (one named function): facts → u32 lanes, once; sets
   │
   ├─► dr: LocalMap for Y prepared once, digest sealed into the frame;
   │        FrameStore::check / Guards::check before every reuse
   │
   ├─► BindStore::resolve → admit(MethodRef) → MethodBinding
   │        (Population, AlgebraRef words: GROUP_COUNT, MUL_I64)
   │
   ├─► Quack description, per side (left r(X,Y), right s(Y,Z)):
   │        Filter: Cmp(free, Eq x) | Plane(live)
   │        Agg::GroupReduce { key: GroupAddr::Local(y_local), agg: GroupAgg::Count }
   │        lower(&Query) → Program
   │
   ├─► mask-risc execute_into(Program, Planes{lanes}, Foreign::none, Scratch, Out::I64(n_y))
   │        → deg_l[y], deg_r[y]        (each is the demanded result of its program)
   │
   ├─► combine (MISSING-3 below): ℕ = deg_l × deg_r ; B = (deg_l>0) ∧ (deg_r>0)
   │
   └─► addressed result: LocalMap::global over nonzero slots, at the boundary only
```

### 17.4 Existing APIs used (read 2026-10-31)

- **Quack** (`lance-graph-quack/src/lib.rs`): `Filter::{Cmp, Plane, EqU32Via, Semijoin, And, Or, Not}`; `Query { filter, agg }`; `Agg::GroupReduce { key: GroupAddr::Local(Col), agg: GroupAgg::Count }` (one program, `Out::I64` sized to the group universe, key past it dropped); `Agg::ScatterOrU32 { fk, out_rows }` (legal only as the demanded result); `lower`, `lower_group_by_semantic`.
- **Quack bind** (`bind.rs`): `trait Binder { table, live, field, code }`, `Draft::bind` (single table). Implementations exist in `lance-graph-glove-parity` (`CanonicalBinder`), `lance-graph-sap`, `lance-graph-dir-sim`, `lance-graph-report` tests; none is backed by `BindStore`.
- **mask-risc** (`ir.rs`, `exec.rs`): `execute_into(&Program, &Planes, &Foreign, &mut Scratch, Out)`, `Planes { n_rows, masks, lanes }`, `Terminal::GroupReduce` / `GroupFold::Count`, `Terminal::ScatterOrU32`.
- **DT64** (`lib.rs`): `BindStore::{route, resolve, resolve_algebra, admit}`, `MethodBinding`, `MethodRef`, `AlgebraRef { lens, word }`.
- **DR** (`dr.rs`): `LocalMap::{prepare, local, global, len, lookups, digest}` (lookups counted; digest is the frame's coordinates revision), `FrameStore::{seal, advance, check}`, `Guards::check`.
- **relalg**: `Rel` / `Measure` / `Query` are single-table; `join_body::is_alpha_acyclic` is the only join-body tool.

### 17.5 Missing integrations (labelled; none worked around)

| id | missing | owner | D-DYNBAG-0 behaviour |
|---|---|---|---|
| MISSING-1 | `AlgebraRef` word → Quack `Agg`/`GroupAgg` mapping (the polyfill lens's words have no Quack reading) | DT64 + Quack | a probe-local table over the two words in use, refusing any other word; filed for the owner |
| MISSING-2 | a `Binder` backed by `BindStore` + `ClassView` (existing binders are catalogue-specific) | Quack bind + DT64 | reconcile, not add: decide whether `CanonicalBinder`'s shape is the template. No new catalogue in the probe |
| MISSING-3 | a post-terminal combine of two group sinks: i64 elementwise multiply and i64 compare-to-mask (both listed in Wave 3) | mask-risc / ndarray | `checked_binary_i64` stays the one direct call (D-SOC-FENCE §6), TRACKED; B is derived from the two degree sinks, never from a scatter mask fed to a second program |
| MISSING-4 | two-relation support as ONE description: relalg has no join IR; `Filter::Semijoin` needs a resident `ForeignPlane`, never a lowered output | relalg / Quack | not built; the per-side degree route above is exact for the admitted chain only |
| MISSING-5 | a reuse cache for prepared `LocalMap`s keyed by frame | DT64 `dr` | the probe holds one map and checks the frame per query; a test asserts `lookups` does not grow with query count beyond intake |
| MISSING-6 | `p(X, Z)` and n-ary witnesses (ℕ-compose) | Wave 1 | refused with a typed refusal; `q(Y)` support is never offered as proof of `p(X, Z)` |
| MISSING-7 | multi-tile global → (tile, local) map | Wave 4 | width 64 or 256; past it, `MapRefusal` |

### 17.6 Acceptance gates for D-DYNBAG-0

1. One connected chain in a test: compile → admit → resolve/bind → Quack `Query` → `lower` → `execute_into` → addressed result. Unbinding the method (retract the route) refuses the computation, not just the test.
2. No owned tuples after intake: `facts()` output is consumed by the intake function only.
3. No unguarded global → local rebuild per query (MISSING-5 test).
4. Support and multiplicity distinct, with XLOG's exact `count` semantics; base relations as sets.
5. Gate B, bag-count, stale-frame and disable tests stay meaningful: each disable run (drop the frame check, drop the admission, swap Count for a scatter) goes red.
6. `q(Y)` is never reported for `p(X, Z)` (MISSING-6 refusal tested).
7. `native.rs` names no `ndarray::simd` fold word; only `checked_binary_i64` remains, listed as TRACKED.

**REVISIT WHEN:** MISSING-3 or MISSING-4 lands in its owner crate; the TRACKED call is then deleted.

### 17.7 Status (2026-11-01, branch `ccr-85acaff0-oszq40-dynbag0`, on #1905)

| gate | state | where |
|---|---|---|
| 1 compile → admit → bind → Quack → `lower` → `execute_into` → addressed result | met; a retracted route yields no `MethodBinding`, so nothing runs | `tests/dto_binding.rs` `the_admitted_method_drives_the_count`, `a_retracted_route_leaves_nothing_to_run` |
| 2 no owned tuples after intake | met: `native::Intake` holds `u32` lanes and a live plane; queries read only those | `src/native.rs` |
| 3 no global → local lookup per query (MISSING-5) | met via `dr::MapTable` (#1905): the intake looks Y up once under a sealed frame; each query re-checks the frame | `queries_after_intake_do_not_look_y_up_again`, `a_stale_map_frame_is_refused_before_anything_runs` |
| 4 support and multiplicity distinct | met: support = `ScatterOrU32` ×2 + `mask_and`; bag = `GroupReduce Count` ×2 + checked product; a `Distinct` method is refused as a bag count | `gate_b`, `litho`, `a_distinct_method_is_not_run_as_a_bag_count` |
| 5 disable runs | all red: Count swapped (litho 3 red), backward dropped (gate_b 5 red), population check, frame check, unread word, intake frame re-check, lookup per query (1 red each) | — |
| 6 `q(Y)` never answers `p(X, Z)` (MISSING-6) | met: `admit` and `admit_count` both refuse a `p(X, Z)` body (behaviour that already held, now pinned) | `p_x_z_is_refused_not_answered_by_q_y` |
| 7 no `ndarray::simd` fold word in `native.rs` | met: only `mask_and` (elementwise alias) and `checked_binary_i64` (TRACKED, MISSING-3) | `src/native.rs` module doc |

MISSING-1 stays probe-local: the caller reads each `AlgebraRef` word as a `native::Word`, and any word without a reading is refused (`BoundRefusal::UnknownWord`).

---

## 18. Cross-frontend projection crosswalk (2026-10-31, before coding)

**DECISION (operator).** XLOG is the first consumer of the projection capability, not its owner. A consumer requests a typed, versioned projection over existing DR/DT64 coordinates and method bindings. A tuple is a coordinate binding; a method is a typed behavioural binding. No frontend owns a projection runtime. Quack (relational description) and R2IL (`FoldDialect` procedures) are both legitimate surfaces, both converging on mask-risc. BindStore resolves and validates; it never executes.

Evidence: four read-only harvests at `d3c2a29`/`f3b2cfe`, each claim cited file:line and spot-checked by the orchestrator. Classes: VERIFIED (production + test), LAB, PROPOSED (plan only), MISSING (nothing found in the searched space).

### 18.1 The five projections are distinct, and "project" means three things

| projection | existing carrier | class |
|---|---|---|
| coordinate (global → local, revision) | `dt64::dr::{LocalMap, FrameStore, Guards}`; `LocalMap::digest` as the frame's coordinates revision | VERIFIED |
| field (meaning × mask → labels/lanes) | `ClassView::project(class, FieldMask)` (`class_view.rs:1079`, **64-bit `FieldMask` only**); `dt64::Resolved::field_view(&WideFieldMask)` (labels, no values) | VERIFIED; no lane borrowing from either |
| method (MethodRef → ON/WHEN/DO, algebra words) | `dt64::MethodBinding { on, when, reads, effects, population, truth, demand }` (`lib.rs:197`); `admit` (`:796`) checks id, `SigRev`, guard witness | VERIFIED as description; **no executor of a `MethodBinding` exists anywhere**; `admit`/`resolve` have no production caller |
| relational (select/join/fold/group/witness) | Quack `Query`/`GroupBy`/`Agg`; relalg `Rel` (single-table, `Measure::Keep`); R2IL `FoldDialect` + `procs` | VERIFIED; no join IR (relalg), semijoin needs resident foreign planes (Quack) |
| consumer (XLOG bindings, SQL results, codegraph relations, R2IL procedures, MLIR exprs) | per frontend, below | mostly LAB / PROPOSED |

"Project" is not one operation: XLOG `RirNode::Project` selects/computes **columns** (and runs on GPU in xlog-runtime); relalg `Measure::Keep` and Quack `Agg::Rows` keep **rows** as a mask; `ClassView::project` and `field_view` select **fields as labels**. No meaning selects methods (that is `Resolved::methods_on`). Do not unify them under a "Project" opcode.

### 18.2 Crosswalk

| frontend / source IR | projection entrypoint | identity & revision witness | method resolution | lowering | execution owner | output demand | present → gap |
|---|---|---|---|---|---|---|---|
| **XLOG** (`RirNode`) | `admit` / `admit_count` (structural match) | `LocalMap` digest (LAB); BindStore route log in a test only | `BindStore::admit(MethodRef::of(XLOG, …))` in `dto_binding.rs` — **asserted, never drives execution** | none: `native.rs` calls `ndarray::simd` | ndarray direct (D-SOC-FENCE violation) | support mask, per-Y count | LAB → §17: binding must select the Quack program that runs |
| **ONNX / Tract** (protobuf + Tract facts) | `plan::resolve` → `Plan::run` | `FrameRevisions{source sha256, relation_map=registry.revision(), coordinates}`, checked every run (`plan.rs:57-68,171`) | own `Registry` (ON × WHEN(K3) × DO), selected binding **is** the one dispatched (`plan.rs:116-128,204`) | cold shape fold; one native fn per node | `lance-graph-onnx` aliases of `ndarray::simd`/`hpc` | every value by name | LAB precedent for binding-drives-execution; gap: own registry not DT64 `BindStore`, lab lens `0x0F00` in Geo domain |
| **SQL / PostgreSQL** | Quack `lower`/`lower_fused`, `Filter::sql_where` (K3); `lance_graph_r2il::procs` | MISSING in code (D-QSP proposes a version-attested codebook witness) | `procs` are Rust functions; ClassView method ordinal PROPOSED | R2IL → `MaskOp` → mask-risc `Program` | mask-risc `execute_into` (`r2il lib.rs:716`) | scalar per fold, `GROUP_SUM`, `KEEP` | VERIFIED fold path; gaps: executable `PROC_DEF` (OPEN in OGAR), bag/join carrier, nullable arithmetic, frontends not yet routed through procs |
| **egglog-MLIR / DialEgg** | `egglog-r2il-probe::lower_and_run` (scalar) | egglog proof self-check + independent i128 oracle | `RuleId` = law × `Arith` domain (refuses reassociation for Checked/Ieee) | `INT_MULT` over empty planes | FoldDialect scalar arm | one scalar | LAB; external `mlir-egglog` admits rewrites with no evidence (float reassociation unconditional) and runs via LLVM — not a model to adopt |
| **Codegraph** (rust-analyzer HIR + Charon LLBC) | `sag-ra-probe` → `llbc_calls.py` → `span_join.py` → `project_rows.py` (TSV) | moniker (candidate only) + FNV body digest + `RESOLVE` move witness → `sid-N`; revision caller-recorded | rust-analyzer impls; Charon `Fun`/`Clause`/`Dyn`/`TraitImpl` | none | none (R2IL `FoldDialect` not connected) | `ProposedMethod` (PROPOSED, no type) | LAB; WITNESSED/POSSIBLE/UNKNOWN preserved in rows; gaps listed in 18.4 |
| **Scallop** (Datalog + provenance tags) | none in lance-graph | none | `Provenance` impls (unit, boolean, natural, top-k proofs) | none | Scallop's interpreter | tagged relations | external VERIFIED; natural's `saturated` is always `true`, so recursive bag counts stop early |

### 18.3 What two consumers already share (and what they do not)

- **Shared and real:** DR coordinates + frame witness (XLOG uses `LocalMap`; Tract uses its own `FrameStore` revisions) and mask-risc as the evaluator (SQL via R2IL; XLOG once §17 lands).
- **Not shared:** method binding. Tract executes through its own registry; DT64 `BindStore` is wired to nothing that executes. The integration proof the operator asks for is therefore not yet possible with existing code: no two consumers obtain their executed method through the same `BindStore`.

### 18.4 Gaps for owning crates (recorded, not built in D-DYNBAG-0)

| id | gap | owner |
|---|---|---|
| X1 | an executor seam: something that takes an admitted `MethodBinding` and selects the existing Quack/R2IL description that runs (Tract's `plan.rs:204` is the precedent shape) | DT64 + Quack/R2IL; decision needed |
| X2 | `admit` checks id, `SigRev`, guard only; it cannot refuse on an effect or on POSSIBLE/UNKNOWN evidence, so an effectful codegraph method would be admitted if registered | DT64 |
| X3 | `ClassView::project` takes `FieldMask` (64); DT64 uses `WideFieldMask` | contract |
| X4 | no Quack `Binder` backed by `BindStore`/`ClassView` (8 impls, none) | Quack bind + DT64 |
| X5 | method identity: `MethodId` is registrant-chosen; codegraph `sid-N` → `MethodId` mint does not exist and needs owner approval; impl/closure identity open | DT64 + OGAR (O2 in the logic-compiler RFC) |
| X6 | codegraph revision witness inside the tools (workspace digest, features, target) | sag-ra-probe |
| X7 | executable `PROC_DEF`/`PROC_CALL` | OGAR `ogar-loco` |
| X8 | Tract registry → DT64 `BindStore`; lab lens `0x0F00` minted in OGAR | tract probe + OGAR |
| X9 | bag/support carrier across procedures (SQL D-QSP-2, XLOG ℕ) | relalg/Quack/mask-risc (Wave 2-3) |

### 18.5 Integration proof, restated against what exists

- **A (XLOG):** buildable inside D-DYNBAG-0 once the executed Quack program is selected by the admitted binding (§17.6 gate 1). Oracles: scalar enumeration and the DR64 register.
- **B (Codegraph):** buildable today only as a hand-fed fixture: rows from the D-SAG-6-2 board entry, pinned `sag_fixture` functions, `tests/law.rs` as the behaviour oracle. Admitting the pure `count_below` and refusing `count_with_tally` / `count_div` / dyn requires X2 (effect/evidence refusal), which does not exist. Wiring to live rust-analyzer/Charon output needs X5, X6.
- **SQL bag fixture:** duplicated base tuples must keep multiplicity; XLOG's intake dedups by design (sets). Needs X9; not in D-DYNBAG-0.
- **Not claimed:** that A and B share one binding contract today. They do only after X1 and X2.

### 18.6 Scope of D-DYNBAG-0 (unchanged, bounded to XLOG)

§17 stands. One addition: the binding that selects the executed program is DT64's `MethodBinding` (its `effects` name the polyfill words), and the selection mechanism follows Tract's precedent (resolve once, dispatch through the resolved entry, frame-checked per run) without copying Tract's registry. If X1 needs a type in DT64, D-DYNBAG-0 stops and asks rather than adding it in the probe.

### 7.6 Result (2026-11-01, branch `ccr-85acaff0-oszq40-dynbag1`)

- **Equality: met.** `xlog-relcomp-probe` `ncompose::pair_counts` (two DR64² i8 tiles, `matmul_i8_to_i32`) equals scalar enumeration per (x, z) and a bag walk of XLOG's RIR for `p`, on the multiplicity fixture and on a 300-case differential (ids 0..6 per axis; ℕ ≠ B in over 100 cases).
- **Projection: met.** `count > 0` equals `relcomp::compose` on the DR64 register, as (x, z) id sets, on every case.
- **Admission.** `ncompose::admit_pair` takes the join in both orders XLOG's optimizer emits, and a `ChainJoin` only when its own relations match its fallback. Bounds are refused. `q(Y)` is still never admitted as `p(X, Z)`.
- **Disable runs: red.** Operands swapped: the differential fails; the hand fixture alone passes, so the differential is what catches it. Result read as B before counting: both tests fail.
- **Cost (MEASURED, `examples/ncompose_cost.rs`, host with AVX-512F, no VNNI, no AMX, so the scalar reference tier):**

  | case | `compose` (B) | `matmul_i8_to_i32` (ℕ, tiles prepared) | `pair_counts` (ℕ, from tuples) |
  |---|---|---|---|
  | sparse, 4 × 4 tuples | 50 ns | 20,849 ns | 21,420 ns |
  | dense 64 × 64 | 2,402 ns | 20,829 ns | 158,637 ns |

- **Answer to open decision 2 (on this host):** a dense i8 GEMM over a whole 64² tile costs about 21 µs whatever the density. It is exact, but it is the wrong carrier for sparse tiles here: 400× slower than B on sparse data and 9× on dense. Untested: the AMX and VNNI tiers, and a bit-packed i1 × i1 → i32 word (popcount of R[x] ∧ Sᵀ[z], which skips empty rows). Per §7.4 that word lands in `ndarray::simd`, backend-first, and is the D-DYNBAG-3 candidate. Building the tiles from tuples (about 17 ns per tuple lookup, dense) also dominates end-to-end time.

---

## 19. Polyfills are composed, not hardcoded (2026-10-31, operator decision before coding)

**Invariant.** Class patterns and bound methods are reusable. Polyfills are compositions of existing algebra atoms, never hardcoded sequences. A `WideFieldMask` selects behaviour words inside a declared basis; DR projects their versioned coordinate surface; DT64 gives each consumer its semantic alias and method recipe; Quack or R2IL lowers the admitted description; mask-risc executes. A frontend never owns a fixed execution sequence that bypasses `MethodBinding`. A backend-owned mapping from an admitted atom to an existing operator is legitimate.

### 19.1 What already provides the composition (read 2026-10-31)

| need | existing API (`lance-graph-dt64/src/lib.rs`) |
|---|---|
| declare a lens of reusable, classid-free words with arities | `BindStore::declare_algebra` (`:628`), `arity` (`:654`) |
| read a lens's words through its `ClassView` | `BindStore::resolve_algebra` (`:665`) → `Resolved::field_view` |
| which words a consumer class uses, as a mask over the lens | `BindStore::words_used` (`:683`) |
| which classes reuse a word | `BindStore::users_of` (`:700`) |
| the recipe: ordered DO steps, each a word with ordered operands | `MethodBinding::effects` (`:197`), `Effect { target, algebra, args }`, `Arg::{Field, Const}` |
| select behaviour by field | `Resolved::methods_on` (`:451`), `methods_touching` (`:520`) |

### 19.2 The backend-owned atom → operator mapping already exists, under R2IL

- `ogar_loco::FnIndex` is a `u8` (`OGAR crates/ogar-loco/src/lib.rs:380`), the same width as `AlgebraRef.word`.
- The R2IL fold reading `CONCEPT_R2IL_FOLD = 0xC401` (`ogar-r2il/src/lib.rs:545`, PROVISIONAL, not in `CODEBOOK`) is a lens whose word positions are R2IL opcodes.
- `lance_graph_r2il::FoldDialect::call` maps each opcode to a mask-risc program (`lance-graph-r2il/src/lib.rs:770`), and `ogar_loco`'s interpreter runs call sequences.

So `AlgebraRef { lens: compose(0xC401, 0), word: FnIndex.0 }` names an atom the backend already knows how to run. No name-to-operator table exists in mask-risc, Quack or relalg (searched `src/` of all three for `from_name`/`by_name`/`label`/`parse`/`from_str`; none), so the R2IL lens is the only existing one.

### 19.3 D-DYNBAG-0 stops here: three missing backend capabilities

| id | missing | owner | why essential |
|---|---|---|---|
| R1 | lowering a `MethodBinding`'s ordered `effects` (`AlgebraRef` under the fold lens, `Arg`s) onto `FoldDialect` calls (this is X1, the executor seam) | `lance-graph-r2il` (the reading's crate; DT64 stays non-executing) | without it the executed path cannot follow the binding; any probe-side sequence would be the frontend-owned recipe this section forbids |
| R2 | a keyed-count fold word (`GROUP_COUNT` → mask-risc `Terminal::GroupReduce { fold: GroupFold::Count }`, which exists) | OGAR `ogar-r2il` opcode table + `lance-graph-r2il` arm | the fold band is `VIA, RANGE, SUM, MIN, MAX, GROUP_SUM, KEY_RUNS, ANY, ALL, KEEP, SCATTER_OR, BLEND` (`ogar-r2il/src/lib.rs:499-525`); no keyed count. `GROUP_SUM` over a fabricated all-ones lane would be a per-row materialization |
| R3 | combining two group sinks: checked i64 multiply (ℕ) and nonzero-to-mask (B) | mask-risc word + R2IL fold word (D-DYNBAG-3) | XLOG's chain is two relations over two tables; the per-Y degrees come from two programs, and support/multiplicity are their combination |

Also: `0xC401` is provisional; a lab use is fine, a persisted GUID is not until OGAR mints it.

### 19.4 Wave 0 resumes when

R1 exists in `lance-graph-r2il` (testable with existing words such as `SUM`, `GROUP_SUM`, `POP_COUNT`, `KEEP`) and R2 lands. Then the XLOG binding's effects are `[GROUP_COUNT(left), GROUP_COUNT(right), combine]`, and the Wave 0 gate is: changing or refusing that binding changes or refuses the executed path. Until R3, the combine step stays the tracked `checked_binary_i64` call. Disable runs, register and scalar oracles, and stale-frame tests are kept unchanged.

### 19.5 F### (hypothesis, unchanged status)

`F###` is a candidate compact address space for reusable atoms and compositions, not a spelling of `MethodId`. §19.2 shows atoms already have a compact address (`FnIndex` under a reading's lens) and compositions have one (`MethodBinding::effects` under a consumer classid). Whether a separate region is needed is undemonstrated. No `0xFFFF_F###` classid is minted and the CanonHigh split is unchanged.

---

## 20. Correction: the codegraph is the inspectable representation behind execution, not its toll gate (2026-10-31, operator)

**Invariant.** R2IL / the codegraph is never a mandatory execution intermediary. A consumer resolves and binds once (cold) and then runs its own prepared native plan (Quack, mask-risc, `ndarray::simd`) with no codegraph lookup in the hot path. Tract is the precedent: `plan.rs` resolves once and `Plan::run` dispatches the bound entry after a frame check (`tract-harvest-probe/src/plan.rs:116-128,171,204`). No mandatory graph construction per invocation, no mandatory R2IL interpretation of native code, no second runtime.

Four things stay distinct, and none proves another:

| layer | answers | today |
|---|---|---|
| source / typed codegraph | which operations and dependencies the program has | r2ssa graphs, the Tract `TypedModel`, D-SAG-6 rows (LAB); the universal addressed codegraph (`0xC402`) is PROPOSED |
| optimized plan + rewrite witnesses | which rules turned it into the executable form | `egglog-r2il-probe` law-gated rewrites (LAB, scalar) |
| runtime execution events | what actually ran, under which frame | R1 `Receipt` (bounded by step count) |
| logical provenance | which rules and facts derived an answer | not built; Soufflé-style lazy reconstruction is the model |

Keep source operation ids, method signatures, typed operand edges, plan ids, rule ids and version witnesses, so a receipt can be related back to the codegraph. No per-row or per-op hot-path audit; bounded receipts, and reconstruction on demand.

**Consequences:**
- **R1 (#1892)** is an optional reading: `prepare` (cold) / `run` (hot) / `Receipt`. It is not a dependency of D-DYNBAG-0.
- **D-DYNBAG-0's seam** is binding → prepared native plan. Its missing piece is a backend-owned mapping from a bound word to a Quack aggregate, built at prepare time, not R1.
- **R2 (an R2IL keyed-count word) is dropped for D-DYNBAG-0:** Quack already has `Agg::GroupReduce { Local, Count }`.
- **R3 (combining two group sinks) still stands.**
- **Future:** codegraph relations (`Defines`, `Uses`, `Calls`, `DependsOn`, `Rewrites`, `Emits`, `Executed`, `Derived`) are projections over codegraph addresses for XLOG / Egglog / DialEgg analysis, never a second graph store.
