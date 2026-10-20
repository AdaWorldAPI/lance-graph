# D-RST: RelationshipMapStore × Quack × petgraph — stored relations, compiled traversal

**Status: PROPOSAL, 2026-10-20. Documentation only.** No crate, type, opcode, ABI, dependency or ruling change lands with this plan. Every capability below is labelled IMPLEMENTED, PROBE, PROPOSAL or RULING from its own source.

**Decision taken (operator, 2026-10-20), recorded as provenance, not as validation:** option B. `RelationshipMapStore` owns relationship persistence, identities, index layouts, coordinate-domain seals and invalidation. Quack owns compilation and numeric query semantics. mask-risc on `ndarray::simd` stays the only population executor. This keeps the D-RMAP-1A §1 ownership table and the 2026-10-05 Quack storage-portability ruling as they are. Neither is superseded here.

Read with:
- [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md) §§1, 3, 6–8: ownership, borrowed views, falsifiers, versioning, focus probe.
- [Schnellverschluss](mork-pathmap-schnellverschluss-v1.md) §7b: the native join over a versioned relationship table.
- [D-CPG](causal-pickaxe-resident-git-lance13-temporal-v1.md) §1: `DurableRef`, `ThoughtCommitId`.
- [D-TGA](thought-git-api-v1.md).
- [H14c](d-holo-h14c-thought-first-v1.md) §1: the address is the register.
- Board entry `2026-10-05-quack-storage-portability.md`.
- `.claude/knowledge/fold-execution-laws.md`.

**Short form:** the store remembers, BIND resolves once, Quack compiles, mask-risc executes, petgraph makes it usable.

**The amortization principle (operator, 2026-10-20): write a relationship as a relationship.** A relation is written once, as relation facts, and its relationship-shaped layouts (adjacency by subject and by object, join tables on a declared shared variable) are built at seal time from those facts. They are not rebuilt per query or per front end. RelAlg's semijoins, joins and projections then read prepared relationship layouts instead of translating identities on every evaluation. That is where RelAlg gets faster: the translation is paid once per sealed version, the evaluation is a fold. §5a checks what the shipped BlasGraph CSR contributes to this.

## 0. Findings that shape this plan

1. **Multi-hop under the no-gather ruling is blocked by scatter as much as by gather.** One hop from a frontier mask has three spellings:
   - **Push:** for each set source `s`, write `next[t]` for every `t` in `out(s)`. This writes at positions taken from values: scatter.
   - **Pull:** `next[t] = OR over s in in(t) of frontier[s]`. This reads at positions taken from values: gather.
   - **Tile:** frontier broadcast `∧` a pre-transposed `(target, source)` bit plane, then any per target. Every read and write is at its own coordinate. It costs `|focus|²` bits, so it is bounded by the focus.

   So the only admitted multi-hop shape today is the tile. Over a domain larger than a focus, multi-hop needs an admission decision (§8, A2). It is not an implementation gap a store can fill. Gremlin's `NonFunctionalChain` (`lance-graph-quack/tests/gremlin_parity.rs`, the `Refusal` enum: "needs a barrier and a workspace inside one computation. Gap: no typed phase handoff exists") is two problems. The handoff is the store's to give, as a stored intermediate. The hop is the ruling's to admit.
2. **The existing petgraph integration rests on the quarantined gather.** `AdaWorldAPI/petgraph` `experiments/lance-fold-hop` (fork PR #3) consumes `lance_graph_mask_risc::OutgoingHop`. That is two `Gather` ops plus `ScatterOrU32`, pinned to lance-graph `66636da`, from before #1616's quarantine. Its README already says it is one demanded hop, and that iterative BFS "needs a reviewed phase-handoff / continuation primitive". ISSUES.md records that bumping the pin must enable `quarantined-gather-hop` explicitly. This plan does not build on it.
3. **petgraph's traversal needs only two traits.** `Bfs::new` / `Dfs::new` need `GraphRef + Visitable`, and `next` needs `IntoNeighbors` (`crates/petgraph/src/visit/traversal.rs`). The visit map is a `FixedBitSet` keyed by `IndexType::index()` (`visit/mod.rs`, `impl VisitMap<Ix> for FixedBitSet`), so a view whose `NodeId` is a dense `u32` slot gets petgraph's own visit map. `Csr` (`crates/petgraph/src/csr.rs`) shows the shape: `IntoNeighbors for &Csr` returns a slice iterator over `neighbors_slice`. It owns its `Vec`s, so the view must borrow instead.
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
| `RelationshipMapStore`, `RelationCatalog` | — | not found under `crates/` (searched `*.rs`, `*.md`) |

## 2. Ownership and seams

```text
   SQL-shaped quack::Query   Cypher   Gremlin   Datalog/RelAlg   petgraph algorithms
              \                 |        |            /              |
               \______________ PARSE ___/____________/              |
                                 |                                  |
                     Quack COMPILE (V4: what to compute)            |
                                 |                                  |
     BIND: identities, coordinate domain, authorization, version,  |
           layout capabilities — resolved once ─────────────────────┤
                                 |                                  |
               BUNDLE: choose layout (may not change the answer)    |
                                 |                                  |
          mask-risc / ndarray::simd EXECUTE over borrowed planes    |
                                 |                                  |
   RelationshipMapStore: catalog · facts · derived relations ·      |
   orders · layouts · domains · seals · invalidation  ◄── borrowed views ─┘
                                 |
              cycle seal → Lance (DurableRef)    ThoughtCommitId (speculative)
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
| petgraph | algorithm and iterator vocabulary | storage, a second graph copy |
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

## 3. The petgraph contract: one identity, two consumption shapes

**Shape 1, scalar, unmodified petgraph.** A borrowed view type over one sealed CSR layout:

```rust
#[derive(Clone, Copy)]
pub struct RelView<'s> { seal: &'s LayoutSeal, off: &'s [u32], dst: &'s [u32], n: usize }

impl GraphBase for RelView<'_> { type NodeId = Slot; type EdgeId = EdgeRow; }   // Slot: IndexType over u32
impl<'s> IntoNeighbors for RelView<'s> { type Neighbors = SlotIter<'s>; /* slice of dst[off[a]..off[a+1]] */ }
impl Visitable for RelView<'_> { type Map = FixedBitSet; /* capacity n */ }
impl NodeIndexable / NodeCount / NodeCompactIndexable  // the slot domain is dense
impl IntoEdgeReferences / IntoEdges / Data            // edge id = canonical row: provenance survives
impl IntoNeighborsDirected                            // only when the store holds the reverse permutation;
                                                      // otherwise unimplemented, never computed on demand
```

- Petgraph's `Bfs`, `Dfs`, `DfsPostOrder` and `Topo` then run unmodified. Nothing is copied: the iterator is a slice of the stored layout.
- `EdgeId` is the canonical fact row, so a path can be traced back to facts.
- Out-of-focus or out-of-domain neighbours are a property of the layout. A view over a full-domain CSR has none; a view over a focus layout must report escapes and refuse `global()` reads (D-RMAP-1A §8).

**Classification (admission A3).** Shape 1 is ordinary host iteration inside petgraph's algorithm. It is not population execution and does not go through mask-risc, so the no-gather ruling, which governs fold execution, does not decide it. This plan records that reading as an open admission item rather than assuming it.

**Shape 2, batched relational cursors.** The same `RelationVersion` and `LayoutSeal`, consumed as whole spans and planes:
- CSR spans per slot range;
- tile planes and frontier masks (the only admitted multi-hop shape, finding 1);
- join-table presence planes (intersection = `And`);
- index intersections over sorted permutations.

These are Quack inputs, compiled to `Program`s.

**Compiling iterator chains.** Arbitrary `Iterator::filter` / `flat_map` closures cannot be inspected and will not be compiled or fused. The plan instead proposes a small typed traversal vocabulary that is both an ordinary `Iterator` (terminal `.iter()`) and compilable (terminal `.compile(&quack)`):

| Typed step | Meaning | Compiles to |
|---|---|---|
| `from(mask \| slots)` | start population | resident plane |
| `out(rel)` / `in_(rel)` | one hop | tile hop (in focus); refused over a large domain until A2 |
| `has(field, Cmp)` | filter | `Pred` under the population |
| `dedup()` | set semantics | mask |
| `count()` / `group_count(field)` | terminals | `Count` / `GroupReduce` |
| `repeat_bounded(k)` / `until_fixpoint()` | iteration | k tile hops / hop until the plane stops changing, within focus |
| `emit()` | materialize | explicit terminal; the only exit producing ids |

Each step carries set/bag/path semantics in its type. Mixing them without an explicit `dedup` or `paths` step is a compile error, never a silent collapse (Gremlin's `BagOfElementsNotAMask` and `PathMultiplicity` stay real refusals). A chain that cannot compile refuses at `.compile`. It never falls back to `.iter()` on its own (no silent fallback). This keeps Gremlin's step vocabulary without a traverser VM: the traversal is a value that is compiled, not an object stepped per traverser.

## 4. Execution flow

1. **Declare / store.** `RelationDecl` → store catalog. Facts arrive through the cycle seal. The store records the resulting `DurableRef` against the relation.
2. **Prepare (normalize once).** The store builds layouts from a sealed version: a sorted permutation, CSR, tiles for a declared focus, join tables for a declared shared variable, and the coordinate domain as a relation. Each layout gets a `LayoutSeal`, and each preparation is timed and counted (normalization, index build, translation).
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

**What follows.**
- **Reuse:** the store's CSR/CSC layouts are BlasGraph's storage types, extended with a pattern-only, `u32`-indexed variant that borrows its buffers. No third CSR type is added beside BlasGraph's and the probe's; the probe's CSR is retired in favour of it.
- **Dense rows go to ndarray.** BlasGraph's reserved `StorageFormat::Bitmap` is the bridge: a row stored as a bitmap over its column domain is exactly an ndarray mask, and a Boolean hop over bitmap rows is `mask_and` plus any, through mask-risc, with no indexed read. That is the tile of D-RMAP-1A §8, generalized from 256 slots to any bitmap row. It costs `|rows| × |cols|` bits, so it is chosen per row block by density (BUNDLE), never globally.
- **Sparse rows still need an indexed read.** A sparse row lists column indices; any hop over it reads the frontier at those indices. Hypersparse bitmap blocks (a list of `(block, 64-bit word)`) make that read word-granular instead of bit-granular: one indexed read per 64 columns instead of one per column, and every read counted. That is the concrete content of admission A2 option (a)/(b); this plan does not admit it.
- **Duplication:** `holograph/src/graphblas` carries a second GraphBLAS. The store binds to one (the `lance-graph` BlasGraph types) and records the other as debt, not a second consumer.

## 6. Falsifiers (exact; each guard gets a disable run)

| # | Test | Pass condition | Disable that must turn it red |
|---|---|---|---|
| F1 | unmodified `Bfs`, `Dfs`, `DfsPostOrder` over `RelView` on one sealed CSR | visited order and set equal petgraph over an independently built `Graph` (test-only oracle); zero edge copies, checked by asserting the iterator's slice pointer lies inside the layout buffer | build the view from a copied `Vec` (the pointer check fails); drop a neighbour |
| F2 | one stored relation read by a Quack query, Cypher, the Gremlin adapter, a RelAlg/Datalog rule and petgraph | identical ids and exact counts in set and bag semantics, against a nested-loop oracle; same `RelationVersion` cited by all five | change one layout so it answers set where bag was asked |
| F3 | `out(r).in(r)` and `repeat_bounded(3)` within a focus, over a stored intermediate | equals the oracle; no `Gather`, no `ScatterOrU32` in any `Program` (asserted on the op list); `NonFunctionalChain` lifted only for the tiled case; over a large domain it still refuses | pass the hop-1 result as a raw mask instead of a stored intermediate with its seal (version check must refuse) |
| F4 | reseal; domain ABA (same ids, new domain version); thought slot reuse; compaction without stable row ids; authorization revoked; dirty speculative state; deletion | each is a typed refusal naming the stale component, never a plausible answer (D-RMAP-1A X1–X9, §6a) | key any check on a bare integer or on content |
| F5 | distinct neighbours vs parallel edges; repeated traversers (bag); path multiplicity; recursive reachability | set, bag and path answers stay distinct where the oracle says they differ; a bag→mask step refuses | collapse bag to set inside a step |
| F6 | triangle `edge(X,Y), edge(Y,Z), edge(Z,X)`: existence, count, enumeration; sparse, dense, skewed (hub) graphs | equals an independent oracle; MORK/PathMap where supported (set only); within focus via tiles; larger domains report the refusal instead of an answer | enumerate from the existence mask alone (must miss multiplicity) |
| F7 | cold and warm; 1/8/64 thoughts; domain churn; incremental reseal; scalar `RelView` BFS vs batched tile frontier; shipped BlasGraph `mxv` vs pattern-only borrowed CSR vs bitmap rows through mask-risc, on one relation | measured crossover points, each reported with its command and repetition count; nothing derived by arithmetic is labelled measured | n/a (measurement), but every timed path asserts its answer first |

**F6 is not a gate for F1–F5 or F7.** The store and traversal capabilities stand on their own.

## 7. Sequence: independently mergeable PRs

| D-id | PR | Content | Depends on |
|---|---|---|---|
| D-RST-0 | this plan | documentation; board rows | — |
| D-RST-1 | identity types | `DurableRef`, `ThoughtCommitId` as contract types (D-CPG CPG-00 owns them; coordinate, don't duplicate) with a never-reused rule test | D-CPG agreement (B1) |
| D-RST-2 | store skeleton | `RelationshipMapStore` crate: catalog, `RelationDecl`, facts through the cycle seal, sorted permutation layout, `LayoutSeal`, coordinate domain as a relation; `impl Registrar`; F4 version and domain refusals | D-RST-1 |
| D-RST-3 | petgraph shape 1 | `RelView` in a new fork experiment (`experiments/relstore-visit`, isolated workspace, pinned rev), replacing nothing in `lance-fold-hop`; F1 | D-RST-2 |
| D-RST-4 | layouts | pattern-only `u32` CSR/CSC on BlasGraph's storage types (§5a), bitmap row blocks through mask-risc (`StorageFormat::Bitmap`), and the join table from the probe, behind the store's capability table; layouts built at seal time; F2 (Quack query, Cypher, petgraph first) | D-RST-2 |
| D-RST-5 | derived relations | `materialize` with multiplicity, provenance, dependencies; invalidation; DRed retraction; F4 deletion cases, F5 | D-RST-4 |
| D-RST-6 | typed traversal + multi-hop in focus | the typed step vocabulary; tile multi-hop through stored intermediates; Gremlin adapter `out().in()` within focus; F3 | D-RST-5, A1 |
| D-RST-7 | triangle and amortization | F6 and F7, cold and warm | D-RST-6 |
| D-RST-8 | large-domain multi-hop | only after admission A2 | A2 |

Each PR keeps the probes in `mork-pathmap-probe` as probes. Production code is new code in the store crate with its own tests; it does not import the probe crate.

## 8. Open admission decisions and blockers

- **A1, the global→local binding.** Translating global ids to slots during preparation, by dictionary search, is the "candidate membrane" of D-RMAP-1 §6. Both probes use it. It needs an explicit admission: either accepted as the counted translation H14c rule 6 permits outside the fold, or refused. Until then D-RST-4's layouts are buildable but not admitted for production reads.
- **A2, large-domain multi-hop.** Push is scatter and pull is gather (finding 1). Options:
  - (a) blocked tiles: `|domain|²` bits in 256×256 blocks, with only non-empty blocks stored, i.e. a sparse block matrix;
  - (b) admit a counted translation step between hops (a phase handoff through a stored intermediate);
  - (c) stay refused.

  This decides whether `NonFunctionalChain` can be lifted beyond a focus. §5a narrows the choice: dense row blocks need no indexed read at all (bitmap rows through mask-risc); only sparse rows do, and hypersparse bitmap blocks bring that read down to one per 64 columns.
- **A3, scalar petgraph iteration.** Whether shape-1 host iteration over a borrowed layout sits outside the no-gather ruling, which governs mask-risc folds. This plan reads it that way and asks for confirmation.
- **A4, store backend and home.** Which crate, and whether facts live in their own Lance datasets written through the cycle seal. This is consistent with "the seal owns durable writes"; the storage-portability ruling forbids a universal `Storage` trait, so the store is one owner, not an abstraction.
- **A5, Registrar shape.** `register(name, width)` cannot describe a relation's arity or sorts. Either the store keeps `RelationDecl` beside it (this plan) or Quack's seam grows. The latter changes a Quack trait and needs its own review.
- **B1, identity types.** `DurableRef` and `ThoughtCommitId` do not exist in code, and `ThoughtCommitId`'s never-reused rule is unspecified (D-RMAP-1A §7.6 gap 1). D-RST-2 cannot ship real seals before D-RST-1.
- **B2, petgraph fork pin.** The existing experiment pins a pre-quarantine rev of lance-graph. The new experiment pins a current rev and does not enable `quarantined-gather-hop`.
- **B3, cyclic joins at scale.** The triangle over a large domain has no admitted shape (F6 reports a refusal there). Not a blocker for D-RST-1..6.

**Stop rule:** any step that copies canonical facts per thought, adds a second evaluator or traverser VM, persists a version counter beside Lance and Thought Git, or treats a pre-fold translation as already approved has chosen the wrong boundary.
