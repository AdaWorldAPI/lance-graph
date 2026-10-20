# D-RST: compiled relational geometry and its amortization — RelationshipMapStore × RelAlg/Datalog × Quack

(File name kept for existing links; the plan was first written as "RelationshipMapStore × Quack × petgraph". See the correction below.)

**Status: LAB EXPLORATION, 2026-10-20 (operator). Nothing here is admitted to production, and no production code may be rewired to it.** The relationship store tested here is one option among many, meant for certain data shapes. It is unfit to promote until further testing: the measured preparation is slow (300–400 ns per edge, §7), and nothing has been compared with the alternatives on real workloads. Every "critical path", "PR" or "production" below describes a lab sequence that could only become a production proposal after its own review. Documentation only. No crate, type, opcode, ABI, dependency or ruling change lands with this plan. Every capability below is labelled IMPLEMENTED, PROBE, PROPOSAL or RULING from its own source.

**Decision taken (operator, 2026-10-20), recorded as provenance, not as validation:** option B. `RelationshipMapStore` owns relationship persistence, identities, index layouts, coordinate-domain seals and invalidation. Quack owns compilation and numeric query semantics. mask-risc on `ndarray::simd` stays the only population executor. This keeps the D-RMAP-1A §1 ownership table and the 2026-10-05 Quack storage-portability ruling as they are. Neither is superseded here.

**Correction (operator, 2026-10-20, later the same day), recorded as provenance, not as validation: the center of gravity is reusable relational geometry, not a graph API.**
1. **Petgraph is an optional consumer, an optional intermediate and a usability reference**, on the same footing as the Gremlin adapter. It does not decide the storage format, coordinate model, algebra, optimizer, execution ABI or relationship lifecycle. The first version of this plan made it a pillar ("petgraph makes it usable") and brought D-RST-3 forward. Both are withdrawn: D-RST-3 is OPTIONAL / DEFERRED / MEASUREMENT-GATED (§7), and petgraph parity is an optional interoperability falsifier (F-PG), not an acceptance criterion.
2. **Tile GEMM consumes prepared relationships; it does not obtain them.** It speeds up arithmetic once the operands are in a compatible dense layout. It does not resolve identities, scatter, gather or traverse sparse data (§5b).

**Goal.** Write relationships once. Normalize irregular access into reusable relational geometry: identities, coordinate domains, index permutations, prepared layouts and version dependencies. Amortize that preparation across joins, projections, recursion and traversal, whichever front end asks (SQL-shaped Quack, Cypher, Datalog, RelAlg, Gremlin, petgraph) and across compatible thought versions. **The optimization target is the minimum repeated work per relational reasoning step, not maximum arithmetic throughput.**

**Success criterion.** Datalog, RelAlg and Quack workloads run efficiently over retained relational layouts without importing petgraph or Gremlin. Petgraph-shaped iterators or selected petgraph algorithms may later be exposed over the same layouts, which would show usability, not decide the architecture.

**Scope.** Relations are n-ary: Datalog predicates, variable permutations, join domains, derived relations, recursion and provenance. Graph traversal is one special case of relational composition (a binary relation composed with itself). Neither a graph API nor a GEMM kernel may change the representation chosen for a relation.

Read with:
- [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md) §§1, 3, 6–8: ownership, borrowed views, falsifiers, versioning, focus probe.
- [Schnellverschluss](mork-pathmap-schnellverschluss-v1.md) §7b: the native join over a versioned relationship table.
- [D-CPG](causal-pickaxe-resident-git-lance13-temporal-v1.md) §1: `DurableRef`, `ThoughtCommitId`.
- [D-TGA](thought-git-api-v1.md).
- [H14c](d-holo-h14c-thought-first-v1.md) §1: the address is the register.
- Board entry `2026-10-05-quack-storage-portability.md`.
- `.claude/knowledge/fold-execution-laws.md`.

**Short form:** the store remembers the geometry, BIND resolves once, RelAlg/Datalog define the bindings, Quack compiles, mask-risc executes. Graph APIs are optional views over the same layouts.

**The amortization principle (operator, 2026-10-20): write a relationship as a relationship.** A relation is written once, as relation facts, and the seal fixes its identity, its version and the base indexes its declaration names (a sorted permutation, and CSR/CSC where declared). Specialized execution geometry (bitmap blocks, focus tiles, a co-addressed join domain on one shared variable) is prepared on the first demand that needs it, once, and retained under a dependency seal on that relation version. Nothing is rebuilt per query or per front end, and nothing is built for a query that never comes: a relation with several joinable attributes has many candidate join domains, and only the demanded ones exist. The seal establishes validity; preparation establishes geometry, and F7 measures whether it paid for itself. RelAlg's semijoins, joins and projections then read prepared relationship layouts instead of translating identities on every evaluation. That is where RelAlg gets faster: the translation is paid once per sealed version, the evaluation is a fold. §5a checks what the shipped BlasGraph CSR contributes to this.

## 0. Findings that shape this plan

1. **Multi-hop under the no-gather ruling is blocked by scatter as much as by gather.** One hop from a frontier mask has three spellings:
   - **Push:** for each set source `s`, write `next[t]` for every `t` in `out(s)`. This writes at positions taken from values: scatter.
   - **Pull:** `next[t] = OR over s in in(t) of frontier[s]`. This reads at positions taken from values: gather.
   - **Tile:** frontier broadcast `∧` a pre-transposed `(target, source)` bit plane, then any per target. Every read and write is at its own coordinate. It costs `|focus|²` bits, so it is bounded by the focus.

   So the only admitted multi-hop shape today is the tile. Over a domain larger than a focus, multi-hop needs an admission decision (§8, A2). It is not an implementation gap a store can fill. Gremlin's `NonFunctionalChain` (`lance-graph-quack/tests/gremlin_parity.rs`, the `Refusal` enum: "needs a barrier and a workspace inside one computation. Gap: no typed phase handoff exists") is two problems. The handoff is the store's to give, as a stored intermediate. The hop is the ruling's to admit.
2. **The existing petgraph integration rests on the quarantined gather.** `AdaWorldAPI/petgraph` `experiments/lance-fold-hop` (fork PR #3) consumes `lance_graph_mask_risc::OutgoingHop`. That is two `Gather` ops plus `ScatterOrU32`, pinned to lance-graph `66636da`, from before #1616's quarantine. Its README already says it is one demanded hop, and that iterative BFS "needs a reviewed phase-handoff / continuation primitive". ISSUES.md records that bumping the pin must enable `quarantined-gather-hop` explicitly. This plan does not build on it.
3. **(Optional consumer; evidence for D-RST-3.) petgraph's traversal needs only two traits.** `Bfs::new` / `Dfs::new` need `GraphRef + Visitable`, and `next` needs `IntoNeighbors` (`crates/petgraph/src/visit/traversal.rs`). The visit map is a `FixedBitSet` keyed by `IndexType::index()` (`visit/mod.rs`, `impl VisitMap<Ix> for FixedBitSet`), so a view whose `NodeId` is a dense `u32` slot gets petgraph's own visit map. `Csr` (`crates/petgraph/src/csr.rs`) shows the shape: `IntoNeighbors for &Csr` returns a slice iterator over `neighbors_slice`. It owns its `Vec`s, so the view must borrow instead.
4. **The Registrar seam is real, minimal and unowned.** `lance-graph-quack/src/bind.rs`: `trait Registrar { fn register(&mut self, name: &str, width: u32) -> Result<TableId, RegisterError>; }`. The doc says "No crate owns that today; this trait names the seam, and only tests implement it". The only implementor is a `#[cfg(test)] Mem`. Three `Binder`s exist in library code: `lance-graph-sap` `CatsBinder`, `lance-graph-dir-sim` `UserBinder`, `lance-graph-glove-parity` `CanonicalBinder`.
5. **A coordinate domain is itself a relation.** The join probe needs a seal for its shared domain (Schnellverschluss §7b: a new `y` re-addresses every side). Stored as a relation (the dictionary of global ids, in slot order), the domain gets a `DurableRef` like any other table. No counter is added. A domain change is a new version, and versions are not reused, so domain ABA cannot happen.

## 1. Current capability inventory

| Capability | Where | Grade |
|---|---|---|
| Quack query surface: scan/filter/project/aggregate lowered to one `Program`; "may build a `Program` and must never evaluate one" | `lance-graph-quack/src/lib.rs` module doc | IMPLEMENTED |
| `Binder` (catalog + codebook, resolved once), `Registrar` (catalog + allocation seam), `create_table` → `ResolvedTable` | `lance-graph-quack/src/bind.rs` | IMPLEMENTED (trait); Registrar has no production implementor |
| Gremlin parity adapter: `V, Has, Out, In, Where, Dedup, Values, Count, Sum, GroupCountBy, Emit`; refusals incl. `NonFunctionalChain`, `ComposedFunctionalHop`, `Repeat`, `PathMultiplicity`, `BagOfElementsNotAMask` | `lance-graph-quack/tests/gremlin_parity.rs` | IMPLEMENTED as test code ("adds no operator to Quack and no op to mask-risc") |
| Cypher → Quack: single-label aggregates, one-hop walk count, `count(DISTINCT b)` under `dst` order; refuses a second hop, variable length ("a hop chain over a computed frontier"), WHERE with a hop | `lance-graph-cypher-quack/src/lib.rs` (`Refusal::Gap` doc) | IMPLEMENTED |
| RelAlg: set algebra over one row universe (`Rel`, `Measure`), lowered via R2IL `FoldDialect`; "recursion (`μ`) and the hop (preimage) are not built" | `lance-graph-relalg/src/{ir,lower}.rs`, `lib.rs` | IMPLEMENTED (prototype for D-CMR) |
| α-acyclicity test (GYO) for a join body; routes acyclic → semijoin chain, cyclic → intersection | `lance-graph-relalg/src/join_body.rs` | IMPLEMENTED (test only; no execution) |
| R2IL fold dialect: `VIA, RANGE, SUM, MIN, MAX, GROUP_SUM, KEEP` (+ refused `KEY_RUNS, ANY, ALL, SCATTER_OR, BLEND`) | `lance-graph-r2il`, opcodes from `ogar-r2il` | IMPLEMENTED; no JOIN or HOP op |
| mask-risc: `MaskOp::{Pred, And, Or, Xor, AndNot, Not, Ternlog, Gather}`; terminals incl. `Keep`, `Count`, `GroupReduce`, `ScatterOrU32` | `lance-graph-mask-risc/src/ir.rs` | IMPLEMENTED; no frontier-continuation primitive |
| `OutgoingHop` (two `Gather` + `ScatterOrU32`) | `lance-graph-mask-risc/src/graph_hop.rs`, feature `quarantined-gather-hop` (off) | QUARANTINED (`ISS-GATHER-HOP-1616`) |
| Focus-bound RelationshipMap: sorted run, PathMap trie, slot-space CSR, 256×256 tile via mask-risc, incremental rebind, thought deltas | `mork-pathmap-probe/src/rmap.rs` | PROBE (D-RMAP-1A §8) |
| Versioned relationship table: shared-variable domain, per-side generations, `And` intersection, bag answers, star joins, filters per thought | `mork-pathmap-probe/src/join.rs` | PROBE (Schnellverschluss §7b, D-MPS-8) |
| `DurableRef{dataset, branch, version}`, `ThoughtCommitId` | D-CPG §1 | PROPOSAL; no type in `crates/` |
| Durable write boundary: cycle seal → `LanceCycleWriter`, one `DatasetVersion` per cycle | `cycle_sink.rs`; storage-portability entry | IMPLEMENTED |
| Ownership: store owns facts/orders/maps/tiles; Quack "must not own semantic maps"; RelAlg never owns maps | D-RMAP-1A §1 | PROPOSAL (controlling contract) |
| Quack owns read semantics, the backend supplies capabilities, the seal owns durable writes; "no universal `Storage` trait" | storage-portability entry, 2026-10-05 | RULING |
| No gather; "any address translation is explicit, counted, and outside the direct-position path" | H14c §1 rule 6; 2026-10-16 ruling | RULING |
| A pre-fold global→local binding is a candidate membrane that needs its own review; until then, a typed refusal | D-RMAP-1 §6 | PROPOSAL (refusal stands) |
| petgraph visit traits, `Bfs`/`Dfs`/`Walker`, `Csr` | `AdaWorldAPI/petgraph` `crates/petgraph/src/{visit,csr.rs}` (0.8.3, MSRV 1.85, dev toolchain 1.99) | IMPLEMENTED upstream |
| BlasGraph sparse storage: `CsrStorage`, `CscStorage`, `HyperCsrStorage`, COO→CSR, `GrBMatrix` (CSR + optional CSC), `mxv`/`vxm`/`mxm` over a `dyn Semiring` incl. `HdrSemiring::Boolean`; `StorageFormat::Bitmap` reserved | `lance-graph/src/graph/blasgraph/{sparse,matrix,semiring}.rs`; planner `blasgraph_planner.rs` (Cypher `Expand` → `mxm`) | IMPLEMENTED; a second copy in `holograph/src/graphblas` |
| ndarray dense mask algebra: `mask_and`, `mask_or`, `mask_andnot`, `mask_ternlog`, popcount/any; dense BLAS levels 1–3 | `ndarray/src/simd_masking_ops.rs`, `src/hpc/blas_level{1,2,3}.rs` | IMPLEMENTED; no sparse/CSR primitive under `src/hpc` or `src/simd*` (searched `csr`, `spmv`, `sparse`) |
| Pattern CSR + CSC substrate with edge ids, one store per relationship type ("Kuzu pattern"); `adjacent`, `adjacent_incoming`, `edge_ids`, degrees, `from_edges`; a borrowed `AdjacencyBatch<'a>` (no allocation; deliberately not `Clone`/`Copy`, operator ruling 2026-07-29 "copies are forbidden, borrows are only for the same mailbox"); `intersect` returns an owned `IntersectionResult` | `lance-graph-planner/src/adjacency/{csr,batch,propagate,distance}.rs` | IMPLEMENTED; `u64` offsets/targets/ids, owned `Vec`s, no version pin, edge id = arrival position in the input (`from_edges`), so not a stable fact identity; binary only; used only inside its own module |
| `CsrIndex` (offsets + neighbours, `u64`), built from an Arrow `RecordBatch` | `lance-graph/src/csr_index.rs` | IMPLEMENTED; owned, out-edges only |
| Borrowed, version-pinned pattern-CSR view traversed by petgraph's unmodified `Bfs`/`Dfs`; Quack one-hop over the same buffers | `AdaWorldAPI/petgraph#4`, `experiments/relstore-visit` | PROBE (D-RST-3, now deferred); its view is `Copy`, which the 2026-07-29 ruling forbids (§3.3) |
| BF16 16 × 16 tile GEMM vs bit-tile composition, polyfill tier only | `mork-pathmap-probe/src/tile_gemm.rs` (#1623, D-RMAP-2) | PROBE |
| `RelationshipMapStore`, `RelationCatalog` | — | not found under `crates/` (searched `*.rs`, `*.md`) |

**Prior art to reuse before adding a type.** Three owned CSR families already exist: the planner's `AdjacencyStore` (CSR **and** CSC, pattern-only, with a borrowed batch view that already follows the no-`Copy` ruling), core `CsrIndex`, and BlasGraph's valued `CsrStorage`/`CscStorage` (plus the `holograph` copy). The native borrowed view of §3.1 starts from `AdjacencyStore`/`AdjacencyBatch`: it already has the shape (pattern CSR + CSC + ids, borrowed batch). Its gaps are `u32` slots, a stable fact identity instead of the arrival position, a version pin, and n-ary relations as sorted permutations. No fourth owned CSR type is added; the probe CSRs (`rmap.rs`, `relstore-visit`) stay probes. Masked adjacency is already a presence plane in mask-risc (`Planes.masks`) and the probe tile (`rmap.rs`, 256 × 256); no new tile type is needed for the Boolean path.

## 2. Ownership and seams

```text
   Quack query   Cypher   Datalog / RelAlg               optional consumers
         \          |          /                          (petgraph, Gremlin adapter)
          \______ PARSE ______/                                     :
                    |                                               :
        Quack COMPILE (V4: what to compute)                         :
                    |                                               :
   BIND: identities, coordinate domain, authorization, version,     :
         eligible prepared layouts — resolved once                  :
                    |                                               :
   BUNDLE: choose a prepared layout (may not change the answer)     :
                    |                                               :
   mask-risc / ndarray::simd EXECUTE over borrowed planes           :
                    |                                               :
   RelationshipMapStore: catalog · facts · derived relations ·      :
   permutations · layouts · domains · seals · invalidation  ◄········ borrowed views of the
                    |                                                 same layouts (no copy,
       cycle seal → Lance (DurableRef)    ThoughtCommitId             no second graph)
                    |
       SoA: canonical identity-addressed state (separate admission)
```

| Owner | Owns | Must not own |
|---|---|---|
| RelationshipMapStore | relation catalog (arity, argument sorts, set/bag, functional constraints); facts; derived relations with multiplicity, provenance and dependencies; sorted permutations; physical layouts (CSR, tiles, join tables); coordinate domains; layout seals; invalidation | Datalog/Gremlin syntax, a query evaluator, a version counter of its own, CE64 fields |
| BIND | resolving names, identities, domains, authorization and versions into a witness; attesting which layouts are eligible | layouts, storage, execution |
| Quack | compiling relational and graph operations into `Program`s; numeric query semantics; the `Registrar`/`Binder` seams | storage, catalog state, evaluation |
| RelAlg / Datalog | variable bindings, join bodies, set vs bag, recursion and fixpoint meaning | map tables, index builds |
| mask-risc / `ndarray::simd` | population execution | anything else |
| petgraph, Gremlin adapter (optional consumers) | their algorithm and iterator vocabulary, where a measured workload adopts it | storage format, coordinate model, algebra, optimizer, execution ABI, relationship lifecycle, a second graph copy |
| SoA | canonical identity-addressed state; admission of stable semantic relationships | derived relations (admission is optional and separate) |
| cycle seal / Lance | durable writes, `DurableRef` | relational semantics |

### 2.1 Seams (proposed shapes, not code)

```rust
// The store satisfies Quack's existing seam for its tables. No new catalog trait.
impl Registrar for RelationshipMapStore { /* register(name, width) -> TableId */ }

// Relations carry more than a width. Declared beside, not inside, Registrar,
// so the Quack trait does not change in the first PRs.
pub struct RelationDecl { name, arity, sorts: [SortId; N], semantics: SetOrBag, functional: Option<Column> }
pub struct RelationId(u32);           // resolved once; never a version
pub struct LayoutId(u32);             // one physical layout of one relation version

/// Every read is pinned to this. Nothing here is a counter the store mints.
pub struct RelationVersion {
    relation: RelationId,
    at: Revision,                     // Durable(DurableRef) | Speculative(ThoughtCommitId)
}
pub struct LayoutSeal {
    layout: LayoutId,
    built_from: RelationVersion,
    domain: RelationVersion,          // the coordinate domain is a relation too (finding 5)
    addresses: Option<DurableRef>,    // row-address map, dropped with stable row ids (D-RMAP-1A §7.3)
}

/// What BIND hands Quack and mask-risc. Borrowed, ephemeral, never stored.
pub struct BoundRelation<'s> { version: RelationVersion, layout: LayoutSeal, planes: Planes<'s>, scope: AuthorizationReceipt }

// Derived relations are relations.
store.materialize(compiled_result, deps: &[RelationVersion]) -> RelationVersion
```

Durable versions come from the seal (`DurableRef`); speculative ones from a thought commit (`ThoughtCommitId`). The store records which versions a layout or derived relation was built from and refuses a read when any of them differs. It persists no counter of its own (D-RMAP-1A §7.7, option B).

## 3. Relational access: borrowed views and compiled cursors

### 3.1 Native borrowed relation views (critical path)

A relation version is read through a **view**: a borrow of the store's prepared buffers, pinned to the `RelationVersion` and `LayoutSeal` they were built from, refused for any other. A view owns nothing and copies nothing. Following the 2026-07-29 ruling, a view is **not** `Clone` or `Copy` (the shape `AdjacencyBatch` already has); it is passed by reference, so its reach is bounded by the borrow it came from.

- **n-ary first.** A relation's facts are columns, one per argument, plus a stable fact identity. Its base indexes are sorted permutations in the variable orders its declaration names (for `r(x, y, z)`: `(x, y, z)`, `(y, x, z)`, …), read as ranges.
- **CSR and CSC are the binary specialization**: offsets over a sorted permutation. CSR serves push and neighbour iteration, CSC serves pull (§5a); both may be declared.
- **The fact identity is the edge id.** A physical row offset is an address that compaction moves (D-RMAP-1A §7.3); a view reports it separately and never as an identity.
- **Start from what exists.** The planner's `AdjacencyStore`/`AdjacencyBatch` already has CSR + CSC + ids and a borrowed batch view (§1). The gaps are `u32` slots, a stable fact identity, a version pin and n-ary permutations, plus BlasGraph's pattern-only reading (§5a). Whether that becomes a change to `AdjacencyStore` or a view beside it is decided by D-RST-9, not here.

### 3.2 Compiled relational cursors

The same `RelationVersion` and `LayoutSeal`, consumed as whole spans and planes, compiled by Quack:
- index ranges over sorted permutations (a range read needs the seal to attest the order; §8 A6);
- CSR/CSC spans per slot range;
- semijoins, joins and projections over a co-addressed join domain (join-table presence planes, intersection = `And`; Schnellverschluss §7b);
- tile planes and frontier masks (the only admitted multi-hop shape, finding 1).

**Typed traversal vocabulary.** Arbitrary `Iterator::filter` / `flat_map` closures cannot be inspected and are not compiled or fused. The plan instead proposes a small typed vocabulary that compiles (terminal `.compile(&quack)`) and can also run as an ordinary iterator (terminal `.iter()`):

| Typed step | Meaning | Compiles to |
|---|---|---|
| `from(mask \| slots)` | start population | resident plane |
| `out(rel)` / `in_(rel)` | one hop | tile hop (in focus); refused over a large domain until A2 |
| `has(field, Cmp)` | filter | `Pred` under the population |
| `dedup()` | set semantics | mask |
| `count()` / `group_count(field)` | terminals | `Count` / `GroupReduce` |
| `repeat_bounded(k)` / `until_fixpoint()` | iteration | k tile hops / hop until the plane stops changing, within focus |
| `emit()` | materialize | explicit terminal; the only exit producing ids |

Each step carries set/bag/path semantics in its type. Mixing them without an explicit `dedup` or `paths` step is a compile error, never a silent collapse (Gremlin's `BagOfElementsNotAMask` and `PathMultiplicity` stay real refusals). A chain that cannot compile refuses at `.compile`. It never falls back to `.iter()` on its own (no silent fallback). The traversal is a value that is compiled, not an object stepped per traverser, so no traverser VM is needed.

### 3.3 Optional: petgraph over the same views (deferred, measurement-gated)

Petgraph's `Bfs`/`Dfs` need `GraphRef + Visitable` and `IntoNeighbors` (finding 3). Over a §3.1 view they can run unmodified, with the iterator a slice of the stored layout and `EdgeId` the fact identity. AdaWorldAPI/petgraph#4 showed this works (§7, D-RST-3 result). It is not on the critical path, and nothing in §§3.1–3.2 depends on it.

Two constraints for any later adoption:
- **`GraphRef: Copy` conflicts with the no-`Copy` view.** petgraph#4's view is `Copy`, which the 2026-07-29 ruling forbids. The compatible shape implements the visit traits for `&View` (petgraph's blanket `impl GraphRef for &G`), so the copyable thing is a shared reference and the view itself stays unique.
- **`IntoNeighborsDirected`** only where the store holds CSC; never computed on demand.

Admission A3 (whether host iteration sits outside the no-gather ruling) only needs an answer if petgraph or another scalar iterator is adopted.

## 4. Execution flow

1. **Declare / store.** `RelationDecl` → store catalog. Facts arrive through the cycle seal. The store records the resulting `DurableRef` against the relation.
2. **Seal, then prepare on demand.** At seal time the store builds only the base indexes the declaration names (sorted permutation; CSR/CSC where declared) and the coordinate domain as a relation. A specialized layout (bitmap blocks, a focus tile, a join table for one shared variable) is built when a compiled demand first needs it and is retained under its `LayoutSeal`, so the next demand on the same relation version reuses it. Each preparation is timed and counted (normalization, index build, translation), and an unused layout is never built.
3. **Bind.** BIND resolves names → ids, picks the domain, checks authorization, pins versions, and lists the layouts whose seals match. The global→local translation runs here or in step 2, never inside a fold (H14c rule 6), and it is subject to admission A1.
4. **Compile.** Quack turns the query (Quack query, Cypher, Gremlin, RelAlg/Datalog, typed traversal) into a demand plus a `Program` over the eligible layouts' planes.
5. **Bundle.** Choose a layout using the capability table (§5). By fold law 3, the choice may not change the answer; F2 is the test.
6. **Execute.** mask-risc over borrowed planes. Results come back as masks, counts or group reductions; tuples only through an explicit `emit`/`materialize`.
7. **Remember.** A result that later steps reuse is materialized as a derived relation: its rows plus multiplicity, provenance (source fact rows or binding tuples) and dependencies. It can be queried, laid out and bound like any relation. A bare mask is never stored as a relation; it carries neither bindings nor multiplicity.

## 5. Layout reuse and invalidation

**Measured selection inputs (probes, warm, one machine; to be re-measured cold in F7):**

| Layout | Fits | Measured | Source |
|---|---|---|---|
| sorted permutation | sparse, any domain; seeks | build 29–483 ms for 0.4–2.9 M facts | D-RMAP-1A §8.3 |
| CSR (slot space) | sparse in focus | wins at degree 2 (3.3 vs 5.1 µs) | §8.3 |
| 256×256 tile | dense in focus; the only no-gather multi-hop | wins dense (4.5 vs 12.4 µs) | §8.3 |
| join table on a shared variable | star / acyclic joins on one variable | prepare 2.7 ms vs PathMap 14.7 ms (n = 50 000); count 36 µs | Schnellverschluss §7b |
| PathMap trie | prefix seek on sparse | faster than the run on sparse, 4–6× slower to build | §8.6 |

**Invalidation rule:** a layout or derived relation is valid only while every `RelationVersion` in its seal or dependency list equals the current one. Otherwise it is refused, never silently reused.
- **Maintenance:** incremental where measured to win (one side of a join table, or k slots of a focus at small k); a rebuild otherwise. The probe crossovers are the starting policy, to be re-measured.
- **Deletions:** a derived relation over a deleted fact is retracted before re-derivation (DRed, D-TGA Rule 2; D-RMAP-1A X9).
- **Thoughts:** a thought reads layouts at its pinned versions and adds sparse deltas. Layouts are not copied per thought (D-RMAP-1A §8.5 measured 32 B per thought as sparse edits).

## 5a. BlasGraph CSR and ndarray: what to reuse (read in source, 2026-10-20)

BlasGraph already ships the relationship-matrix vocabulary this plan needs: CSR, CSC, hypersparse CSR, COO→CSR, a Boolean semiring, and a planner that lowers Cypher `Expand` to `mxm`. The store should reuse that vocabulary for its CSR/CSC layouts instead of carrying the probe's CSR into production. As shipped it cannot serve as a relationship layout, for five reasons, each read in source:

1. **Every entry is a 2 KiB value.** `CsrStorage { row_ptrs: Vec<usize>, col_indices: Vec<usize>, values: Vec<BitVec> }`, and `BitVec` is `[u64; 256]` (16 384 bits). A structural relationship has no value, so 400 000 edges would cost about 800 MB of values. The store needs a pattern-only CSR/CSC (values absent, or a `u32` canonical fact row for provenance).
2. **Indices are `usize`.** Eight bytes per index, where a sealed domain needs `u32` slots (and a focus `u8`).
3. **`mxv` copies the matrix per call.** `let a = if desc.transpose_input0() { self.transpose() } else { self.clone() };`. Even the "zero-copy" `transpose` with a CSC present clones `col_ptrs`, `row_indices` and `values`. A stored layout must be borrowed by every evaluation.
4. **`mxv` is a pull with an indexed read.** For each row `i`, for each `(j, a_val)` in the row: `vec.get(j)`. That reads the input vector at positions taken from stored column indices, the gather shape, run as scalar host code with a virtual `Semiring` call and a `BitVec` clone per product.
5. **ndarray is not on this path.** BlasGraph routes Hamming and popcount through `ndarray_bridge::dispatch_hamming`, but CSR construction and `mxv` use plain loops. ndarray has dense mask algebra and dense BLAS, and no sparse primitive.

**The BF16 16 × 16 tile GEMM is measured separately** (D-RMAP-2 probe, #1623). On a host without AMX or AVX512-BF16 only ndarray's F32x16 polyfill tier runs; there the bit arm wins every cell, and that result says nothing about tile-hardware performance. The GEMM stays a candidate execution path for tile hosts and weighted counts until one is measured; §5b sets the terms on which it can win.

**What follows.**
- **Structural and valued relations are different things, and both stay.** BlasGraph's `HdrSemiring::Boolean` is bitwise AND/OR over 16 384-bit hyperdimensional values, not edge existence. A structural relation's value is the edge's existence (plus an optional fact identity), with no 2 KiB payload. A valued relation keeps BlasGraph's hyperdimensional vectors, or another typed edge value, read only when the computation needs it. The lean CSR is a pattern-only interpretation inside the same sparse-storage family; it does not replace BlasGraph's hyperdimensional algebra.
- **Reuse means a representation change, not a constructor.** `CsrStorage` owns its `Vec`s, so borrowing buffers and dropping values is a new representation. The first step is a borrowed, version-pinned pattern-CSR **view** defined alongside the existing storage types; whether to refactor `CsrStorage` itself is decided after it has been measured. The probe's CSR is retired in favour of that view; no third owned CSR type is added.
- **Dense rows go to ndarray, and the direction decides whether that is gather-free.** BlasGraph's reserved `StorageFormat::Bitmap` is the bridge: a bitmap row is an ndarray mask. But the representation does not settle the question; the operation does. With `A[s,t]` an edge from source `s` to target `t` and `F` the frontier:

  | Representation | Operation | Indexed access |
  |---|---|---|
  | lean CSR | iterate neighbours | row-offset lookup, then a contiguous scan |
  | sparse bitmap blocks | intersect frontier words | block lookup (an indexed read per 64 columns), then a word-wise AND |
  | dense **predecessor** bitmap (per target, a row over sources) | `F'[t] = any(pred(t) ∧ F)` | none: co-addressed AND plus a reduction |
  | dense **successor** bitmap (per source, a row over targets) | push: OR the rows of the selected sources | source-row selection is indexed by the frontier's set bits, then OR accumulation |

  Only the predecessor form is the admitted mask-risc case, and it is the tile of D-RMAP-1A §8 generalized from 256 slots to any row block. Expanding a whole frontier with it still needs one AND-and-reduce per target, so it costs `|targets| × |sources| / 64` words whatever the frontier's size; it is chosen per row block by density (BUNDLE), never globally. The compiler therefore picks the relation's direction to match the algebra (CSC for pull, CSR for push and neighbour iteration), which is a reason to keep both CSR and CSC rather than one universal layout.
- **Sparse rows still need an indexed read.** A sparse row lists column indices; any hop over it reads the frontier at those indices. Hypersparse bitmap blocks (a list of `(block, 64-bit word)`) make that read word-granular instead of bit-granular: one indexed read per 64 columns instead of one per column, and every read counted. That is the concrete content of admission A2 option (a)/(b); this plan does not admit it.
- **Duplication:** `holograph/src/graphblas` carries a second GraphBLAS. The store binds to one (the `lance-graph` BlasGraph types) and records the other as debt, not a second consumer.

## 5b. Tile GEMM consumes prepared geometry (operator correction, 2026-10-20)

```text
global relationship facts      sparse · irregular identities · CSR / CSC / trie
          │
   gather / resolve / pack     select coordinates, build dense tiles, align shared domains
          │                    ← possibly expensive preparation
   tile GEMM / mask-risc       repeated, regular arithmetic on prepared operands
          │
   relational results          counts · existence · frontiers · derived relations
```

A GEMM or AMX kernel loads its operands from memory, so they need not be in registers already, but they must already be laid out as tiles. Every gather, pack or coordinate normalization happens before the kernel, and it is paid there. If a traversal gathers and packs each time it runs, even a very fast tile kernel can lose to an ordinary CSR iterator. That is the case for retaining prepared tiles under a versioned seal in the store (§4 step 2).

**Three costs, measured separately for every tile accelerator:**
1. **Preparation**: identity resolution, coordinate alignment, index traversal, gathering, packing, tile construction, sealing.
2. **Execution**: mask-risc bit operations, SIMD, or BF16/AMX GEMM on prepared operands.
3. **Maintenance**: version checks, invalidation, incremental updates, repacking, rebuilds.

For `N` uses and `M` invalidations of one prepared layout:

```text
T_tile(N, M)   = T_prepare + N · T_exec + M · T_maintain
T_sparse(N)    = N · T_traverse            (CSR/CSC or a native iterator, no tile)
```

The tile wins only where `T_exec < T_traverse`, and then only beyond the reuse count where the two lines cross. Break-even counts are reported from measured `T_prepare`, `T_exec`, `T_maintain` and `T_traverse`, never from an assumed kernel speedup. A warm tile kernel compared with a cold sparse traversal is not an end-to-end result. A faster isolated GEMM kernel does not establish a faster graph operation.

**Representations keep their roles:**

| Technique | Best role |
|---|---|
| borrowed pattern CSR/CSC, sorted permutations, native cursors | sparse, irregular navigation and range reads |
| sparse bitmap blocks | indexed word-level access, every indexed read counted |
| dense predecessor bitmaps (CSC direction) | gather-free, co-addressed frontier tests |
| mask-risc / ndarray bit tiles | dense Boolean relationship composition |
| BF16 / AMX tile GEMM | dense arithmetic composition, where density, reuse and precision justify it |
| RelationshipMapStore | retaining prepared layouts so their cost is paid across reuse |

Sparse data is not forced into dense tiles in order to use GEMM. Structural Boolean adjacency stays distinct from BlasGraph's hyperdimensional `BitVec` values (§5a).

**Boolean and arithmetic are different questions.** Reachability and existence need only `And`/`Or`; a BF16 product computes path multiplicities. Thresholding a count to existence is an explicit derivation step, never an implicit reading. BF16 operands carry an 8-bit significand, so they represent integers exactly only up to 256. The F32 accumulator is exact up to 2^24. Any arithmetic path through BF16 checks its counts against those bounds, and any count it cannot represent exactly is refused. Polyfill (F32x16) measurements and native AMX/AVX512-BF16 measurements are reported apart; this host has neither instruction set (#1623).

Evidence so far: on the polyfill tier the bit arm wins every D-RMAP-2 cell (#1623). D-RST-3 measured a borrowed CSR walk at about 2 ns per edge against a preparation of 300–400 ns per edge (§7), which is exactly the cost the store exists to amortize.

## 6. Falsifiers (exact; each guard gets a disable run)

| # | Test | Pass condition | Disable that must turn it red |
|---|---|---|---|
| F1 | native borrowed views (§3.1), no petgraph: range reads over sorted permutations of an n-ary relation, CSR and CSC spans of its binary projection | equal a nested-loop oracle built from the facts; zero copies (each returned slice lies inside the layout buffer); building a view allocates nothing; fact ids survive compaction while rows move; a stale version and an out-of-domain id are refused | build the view from a copied `Vec`; drop a row; use the row offset as the id; skip the version check |
| F2 | one stored relation read by a Quack query, Cypher, the Gremlin adapter and a RelAlg/Datalog rule | identical ids and exact counts in set and bag semantics, against a nested-loop oracle; same `RelationVersion` cited by all four | change one layout so it answers set where bag was asked |
| F2b | hypothesis: a Quack query, a Gremlin `out()` expansion and a Datalog recursive step over the same relation compile to compatible demands and reuse one prepared layout, without sharing an execution model | the second and third front ends find the layout prepared by the first (prepare count stays 1, asserted), answers equal their oracles | give each front end its own layout key (prepare count goes to 3) |
| F3 | `out(r).in(r)` and `repeat_bounded(3)` within a focus, over a stored intermediate | equals the oracle; no `Gather`, no `ScatterOrU32` in any `Program` (asserted on the op list); `NonFunctionalChain` lifted only for the tiled case; over a large domain it still refuses | pass the hop-1 result as a raw mask instead of a stored intermediate with its seal (version check must refuse) |
| F4 | reseal; domain ABA (same ids, new domain version); thought slot reuse; compaction without stable row ids; authorization revoked; dirty speculative state; deletion | each is a typed refusal naming the stale component, never a plausible answer (D-RMAP-1A X1–X9, §6a) | key any check on a bare integer or on content |
| F5 | distinct neighbours vs parallel edges; repeated traversers (bag); path multiplicity; recursive reachability | set, bag and path answers stay distinct where the oracle says they differ; a bag→mask step refuses | collapse bag to set inside a step |
| F6 | triangle `edge(X,Y), edge(Y,Z), edge(Z,X)`: existence, count, enumeration; sparse, dense, skewed (hub) graphs | equals an independent oracle; MORK/PathMap where supported (set only); within focus via tiles; larger domains report the refusal instead of an answer | enumerate from the existence mask alone (must miss multiplicity) |
| F7 | cold and warm; 1/8/64 thoughts; domain churn; incremental reseal; shipped BlasGraph `mxv` vs pattern-only borrowed CSR vs bitmap rows through mask-risc, on one relation | measured crossover points, each reported with its command and repetition count; nothing derived by arithmetic is labelled measured | n/a (measurement), but every timed path asserts its answer first |
| F8 | **amortization**, one fixed relation, four paths: (A) native iterator over borrowed CSR; (B) compiled sparse traversal over CSR/CSC; (C) prepared Boolean bit tiles through mask-risc; (D) prepared arithmetic tiles through GEMM. Each cold and warm, then over repeated queries and 1/8/64 compatible thought versions; sparse, medium, dense and skewed (hub) distributions | reported per path: preparation time and bytes allocated, execution latency, index reads and copies, layout reuse count, invalidation/rebuild cost, the end-to-end crossover from §5b's two lines; exact semantic parity across A–D where the semantics coincide (existence for A–C, counts for D checked against an exact integer oracle and the BF16 bounds); BF16 polyfill and hardware numbers in separate columns | n/a (measurement); every timed path asserts its answer first, and a path that skips its preparation in the cold run is a defect |
| F-PG | **optional interoperability, only if petgraph is adopted (§3.3)**: unmodified `Bfs`, `Dfs` over a §3.1 view | visited order and set equal petgraph over an independently built `Graph`; zero copies; fact ids as edge ids; the view is not `Copy` | as F1 |

**F6 is not a gate for F1–F5, F7 or F8. F-PG gates nothing.** The store and traversal capabilities stand on their own.

## 7. Sequence: independently mergeable PRs

| D-id | PR | Content | Depends on |
|---|---|---|---|
| D-RST-0 | this plan | documentation; board rows | — |
| D-RST-1 | identity types | `DurableRef`, `ThoughtCommitId` as contract types (D-CPG CPG-00 owns them; coordinate, don't duplicate) with a never-reused rule test | D-CPG agreement (B1) |
| D-RST-2 | store skeleton | `RelationshipMapStore` crate: catalog, `RelationDecl`, facts through the cycle seal, sorted permutation layout, `LayoutSeal`, coordinate domain as a relation; `impl Registrar`; F4 version and domain refusals | D-RST-1 |
| D-RST-9 | native borrowed relation views (critical path, first) | §3.1 views without petgraph: n-ary sorted permutations read as ranges, pattern CSR/CSC as the binary case, stable fact ids, version pin, not `Copy`; starting from the planner's `AdjacencyStore`/`AdjacencyBatch` and BlasGraph's pattern reading, not a new owned CSR; F1 | nothing (an isolated probe can run before D-RST-1/2) |
| D-RST-3 | **OPTIONAL / DEFERRED / MEASUREMENT-GATED**: petgraph over a borrowed view | built as an experiment before this correction (AdaWorldAPI/petgraph#4); kept as evidence, not extended; adoption only after D-RST-7 measures direct relational methods against petgraph-style iteration, and only for the workloads where it wins (F-PG) | D-RST-9, D-RST-7 |
| D-RST-4 | layouts | pattern-only `u32` CSR/CSC from D-RST-9 (§§3.1, 5a), bitmap row blocks through mask-risc (`StorageFormat::Bitmap`), and the join table from the probe, behind the store's capability table; base indexes at seal time, specialized layouts prepared on demand and retained under their seal; F2, F2b | D-RST-2, D-RST-9 |
| D-RST-5 | derived relations | `materialize` with multiplicity, provenance, dependencies; invalidation; DRed retraction; F4 deletion cases, F5 | D-RST-4 |
| D-RST-6 | compiled relational cursors | semijoins, joins and projections over retained layouts (§3.2); the typed step vocabulary; tile multi-hop through stored intermediates; Gremlin adapter `out().in()` within focus; F3 | D-RST-5, A1 |
| D-RST-7 | triangle and amortization | F6, F7 and F8, cold and warm; F8's path D only through the existing D-RMAP-2 probe kernels, with no new production kernel | D-RST-6 |
| D-RST-8 | large-domain multi-hop | only after admission A2 | A2 |

**Lab sequence (not a production path):** D-RST-9 → D-RST-1 → D-RST-2 → D-RST-4 → D-RST-5 → D-RST-6 → D-RST-7. **Off it:** D-RST-3 (optional petgraph), D-RST-8 (waits on A2), F-PG, and F8's path D (a measurement through existing probe kernels, never a reason to add one).

**D-RST-3 result (2026-10-20, AdaWorldAPI/petgraph#4, `experiments/relstore-visit`), recorded as evidence for a deferred optional item.** The view met what was then F1 (now F-PG): petgraph's unmodified `Bfs`/`Dfs` match a `DiGraph` oracle built from the facts on fixtures covering parallel edges, self-loops, isolated nodes and empty relations. The edge id is the fact id, and it survives compaction while rows move. The view borrows the prepared buffers, and building it or walking every neighbour list allocates nothing. The F2b-shaped falsifier passes: a Quack one-hop reads the same `dst` buffer at the same version, and the store prepares once.

Measured at 1M edges: the neighbour walk and BFS cost the same as petgraph `Csr` (2.0 vs 1.8–1.9 ns per edge). Prepare takes 308–396 ns per edge and is not optimized. The Quack one-hop is O(E) per query (14–18 ms for 64 hops vs 0.3 µs through the view), because Quack cannot know that the `src` lane is sorted; that question belongs to the critical path and is §8 A6. The experiment's view is `Copy`, which the 2026-07-29 ruling forbids; §3.3 names the compatible shape. It is not extended further.

Each step stays a probe or isolated experiment (`mork-pathmap-probe`, the petgraph fork's `experiments/`). None of it is production code, none is wired into an existing crate, and none may be promoted without a separate review after further testing.

## 8. Open admission decisions and blockers

- **A1, the global→local binding.** Translating global ids to slots during preparation, by dictionary search, is the "candidate membrane" of D-RMAP-1 §6. Both probes use it. It needs an explicit admission: either accepted as the counted translation H14c rule 6 permits outside the fold, or refused. Until then D-RST-4's layouts are buildable but not admitted for production reads.
- **A2, large-domain multi-hop.** Push is scatter and pull is gather (finding 1). Options:
  - (a) blocked tiles: `|domain|²` bits in 256×256 blocks, with only non-empty blocks stored, i.e. a sparse block matrix;
  - (b) admit a counted translation step between hops (a phase handoff through a stored intermediate);
  - (c) stay refused.

  This decides whether `NonFunctionalChain` can be lifted beyond a focus. §5a narrows the choice: dense row blocks in the predecessor (pull) direction need no indexed read (`any(pred(t) ∧ F)` through mask-risc); the push direction and sparse rows do, and hypersparse bitmap blocks bring that read down to one per 64 columns.
- **A3, scalar host iteration (only if petgraph or another scalar iterator is adopted).** Whether host iteration over a borrowed layout sits outside the no-gather ruling, which governs mask-risc folds. This plan reads it that way and asks for confirmation when D-RST-3 is taken up.
- **A6, attested order.** A sorted permutation turns a one-key read into a range only if Quack may rely on the order. D-RST-3 measured the cost of not relying on it (a full-lane scan per query). Whether the `LayoutSeal` carries an ordering witness that Quack's existing `Cmp::Range` lowering (D-DIAMOND) can validate is open.
- **A4, store backend and home.** Which crate, and whether facts live in their own Lance datasets written through the cycle seal. This is consistent with "the seal owns durable writes"; the storage-portability ruling forbids a universal `Storage` trait, so the store is one owner, not an abstraction.
- **A5, Registrar shape.** `register(name, width)` cannot describe a relation's arity or sorts. Either the store keeps `RelationDecl` beside it (this plan) or Quack's seam grows. The latter changes a Quack trait and needs its own review.
- **B1, identity types.** `DurableRef` and `ThoughtCommitId` do not exist in code, and `ThoughtCommitId`'s never-reused rule is unspecified (D-RMAP-1A §7.6 gap 1). D-RST-2 cannot ship real seals before D-RST-1.
- **B2, petgraph fork pin (optional work only).** `lance-fold-hop` pins a pre-quarantine rev of lance-graph; `relstore-visit` pins a current rev and does not enable `quarantined-gather-hop`.
- **B3, cyclic joins at scale.** The triangle over a large domain has no admitted shape (F6 reports a refusal there). Not a blocker for D-RST-1..6.

**Stop rule:** any step that copies canonical facts per thought, adds a second evaluator or traverser VM, persists a version counter beside Lance and Thought Git, treats a pre-fold translation as already approved, adds a production interface or dependency only to satisfy petgraph compatibility, or forces a representation in order to use GEMM has chosen the wrong boundary.
