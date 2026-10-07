# D-TINKER-JANUS-FOLD-0 — what TinkerPop and JanusGraph teach BIND/BUNDLE/FOLD

> **READ BY:** `fold-carrier-scientist`, `cypher-lowering-warden`,
> `isa-anti-lasagne-warden`, anyone proposing bulking, path liveness, a
> fold barrier, a sorted access path, or parameter caching.
> **Status:** 2026-10-07. Primary source read (no secondary descriptions):
> Apache TinkerPop master `82242e7f` (4.0.0-SNAPSHOT); JanusGraph master
> `dddfdcd1`. Paths below are relative to `gremlin-core/.../gremlin/` and
> `janusgraph-core/.../org/janusgraph/`. Local measurements are cited by probe.
> Grades: [G] read in code / measured, [H] reasoned from code, [S] speculative.

## 1. TinkerPop execution model

A traversal is a step list; traversers (object + state) flow between steps.
The ONLY place traversers merge is `TraverserSet.add` (a LinkedHashMap: an
`equals()` match is folded with `existing.merge()`), reached in OLTP through a
`NoOpBarrierStep`. Strategies rewrite the step list before execution, in fixed
categories (decoration → optimization → provider → finalization → verification).

## 2. Traverser / bulk semantics — the merge key IS the distinction set

| state | must match to merge? | merges by | source |
|---|---|---|---|
| current object | yes | — | AbstractTraverser:181-187 |
| next step id (`future`) | yes | — | B_O_Traverser:70-71 |
| tags | yes (in equals, not hash) | union | O_Traverser:66-79 |
| bulk | no | sum (`ONE_BULK`: stays 1) | B_O:50-53 |
| sack | no if a sackMerger exists (folded); never merges without one | merger | B_O_S_SE_SL:125-161 |
| loops / loopName / stepLabel | yes | — | B_O_S_SE_SL:151-155 |
| labelled path | yes (labelled elements only) | — | B_LP_O_S_SE_SL:117-118 |
| full path | yes (whole history) | — | B_LP_O_P_S_SE_SL:91-92 |
| side effects | no (shared) | — | B_O_S_SE_SL:34 |

**Mapping to our carriers [G]:** `BULK` = per-node walk count (**K** in
`.claude/tools/carrier_sufficiency.py`); `ONE_BULK` (`withBulk(false)`) does
NOT stop merging — it merges and discards multiplicity, i.e. the support mask
(**S**). Path / labels in the key = full bindings (**B**). So TinkerPop's own
semantics already embody "multiplicity is a distinction you may keep or drop".

## 3. LazyBarrierStrategy — scheduling, not semantics; F6 applies

Inserts `NoOpBarrierStep` after flat-map steps (skipping the first, skipping
edge-returning steps), only while no label has been seen, never under PATH,
Drop or Element steps, cap 2500 distinct merged traversers
(LazyBarrierStrategy:74-134). It changes no answer; it is a hash-merge point.

**Transfer: the mechanism does not (F6 — it compensates for one-object-per-
traverser execution; a mask step is already a full barrier). Its LEGALITY
GUARDS do:** a merge point is illegal while PATH, a live label, or an
identity-sensitive operation (drop of a specific element) is downstream.

## 4. PathRetractionStrategy — liveness analysis, the strongest transfer

A backward scan computes, per step, the labels any later step references;
PathProcessor steps drop the rest and a NoOpBarrier is inserted after each, so
retraction widens the merge class (PathRetractionStrategy:91-143). Disabled
wholesale by any lambda or PATH requirement; a Repeat in the lineage keeps
everything (221-245).

**Transfer [G by analogy, H as design]:** this is exactly what `demand::classify`
does for Cypher, coarser: it decides which variables are live at the consumer.
The analysis belongs in the **semantic plan**, before lowering.

## 5. Incident → adjacent

`outE().inV()` (unlabelled, adjacent steps) → `out()`; blocked by Path, Tree,
lambda anywhere in the root traversal; `bothE().bothV()` is never rewritten
(it doubles multiplicity) (IncidentToAdjacentStrategy:79-135). The inverse,
`AdjacentToIncidentStrategy`, turns `out().count()` into `outE().count()`.

**Transfer [G]:** our one-hop `count(*)` already IS AdjacentToIncident — it
re-anchors on the edge table (one edge row = one walk). Incident→adjacent
(skip the edge population, semijoin endpoint directly) is legal exactly when
edge identity and multiplicity are dead: support demand (`TerminalSet`), no
edge property, no path. Illegal: edge property read, edge returned, path,
`count(*)` over parallel edges (our parallel-edge fixture: 4 walks, 2 targets).

## 6. Parameters — GValue pinning is the bind/bundle invalidation rule

Bytecode `Binding` is gone in 4.x. `GValue` = named slot carrying its current
value. `GValueManager` records **pinned** names: a parameter whose value some
rewrite READ; `updateVariable` refuses to change a pinned one
(`step/GValue.java:42-56`, `HasContainerHolder.java:45-52`). The only cache is
a parse cache keyed by source + exact string (GremlinLangScriptEngine:139-178);
strategies re-run every execution. Two core strategies over-pin (they read only
a length).

**Transfer [G+H]:** this is the precise form of our prepared-program rule. A
parameter slot is execute-only **iff no BIND/BUNDLE decision read its value**;
a slot whose value was read (e.g. to fold `v ≥ 2^31` to empty, or to pick a
slice bound) is pinned and invalidates the bundle on change. Our measured case
(Quack Q5 patch, 1004/1004) is the unpinned case. Lesson from the over-pinning:
pin on the value read, never on a structural read.

## 7. JanusGraph storage model

One sorted row per vertex; every incident relation is a column. Column prefix
= relation type id + direction (property 0 / out 2 / in 3), so columns sort by
visibility, kind, type, direction (IDHandler:51-55,130-134,185-205). Edge
column = sort-key bytes, other vertex id, relation id (EdgeSerializer:288-381);
edge properties are copied inline into every copy.

## 8. Bounded adjacency slice — the address range folds

`BasicVertexCentricQueryBuilder` turns label + direction + sort-key constraints
into `SliceQuery`s by the **prefix rule**: equality on k1..k(i-1), at most one
range on k_i, nothing after (EdgeSerializer.getQuery 436-545); EQ/LT/LTE/GT/GTE
and OR-of-EQ only; NOT_EQUAL and the rest are residual in-memory filters;
`isFitted` requires every constraint consumed (BVCQB:740-746).

**Local measurement (D-TJ-3, `bounded_slice_probe.rs`) [G]:** timestamp-ordered
adjacency, `ts ∈ [a,b)`: binary search → `Cmp::Range{lo,hi}` vs a `Ge∧Lt` sweep,
both asserted equal. 3.2–3.4× at 50 % selectivity, **30–38× at 0.1–1 %**. The
floor (~20 µs at 1M) is the population-wide popcount; running over the extent
would remove it.

**Already in tree [G]:** `ordered_lane::OrderedLaneWitness` — storage attests a
lane's order; `Filter::prefix_facet` lowers to `Cmp::Range` only under a valid
witness, otherwise sweeps and records why (`PrefixLowering::{Bound,
SweepNoWitness, SweepInvalidWitness}`). This is JanusGraph's slice, minus one
thing: **the witness exists only for facet-key lanes; an ordered scalar
property (timestamp) has no witness type.** And cypher-quack's
`EdgeTable::dst_ordered: bool` is an unattested second spelling of the same
fact — a defect to replace with the witness.

## 9. Direction and indexes

- Every directed edge is written twice (OUT column in source row, IN in target
  row), plus once per enabled vertex-centric index: ~2 × (1 + indexes) full
  copies (StandardJanusGraph:996-1038). Unidirected labels store OUT only.
- Vertex-centric index = another sorted copy with its own sort key; the reader
  scores candidates (+5/|points| per point key, +1 range, +3 order) (BVCQB:640-688).
- Composite index = equality-only posting lists; mixed index = external engine;
  greedy set cover over candidates, no statistics (a TODO admits it).
- `multiQuery` batching and prefetch exist "if there is a non-trivial latency
  to the backend" (GraphDatabaseConfiguration:330-336) — **F7: kill.**

**Transfer:** on a local SoA store the right shape is one payload plus ordered
projections (a dst-sorted and a src-sorted permutation lane, CSR + CSC), never
full copies. Which projections exist is **BIND metadata**; choosing scan vs
slice vs projection is **BUNDLE**. The JanusGraph scoring rule is cheap,
deterministic and worth copying for projection choice.

## 10. Semantic carrier matrix (from `carrier_sufficiency.py`, exhaustive over small graphs)

See `D-TINKER-JANUS-CARRIER-MATRIX.md`. Headline: **WeightedSupport is K**, the
per-node walk count Quack already produces with `GroupReduce{Local(dst),
Count}`. It answers `count(*)` and `GROUP BY terminal count(*)` where support
cannot (DAG: 4 vs 3), and fails exactly where TinkerPop refuses to bulk: any
question about an earlier variable (`CountBy(n0)`, `Support(n0)` → B) and
TRAIL. **F1 (WeightedSupport) survives — but as an existing carrier, not a new
type.** The missing piece is chaining K into the next hop (foreign-value sum).

## 11. BIND/BUNDLE ownership (derived, not copied)

| concern | owner | evidence |
|---|---|---|
| which distinctions are live (path, labels, multiplicity) | **semantic plan** | `demand::classify` runs on the bound plan; TinkerPop's PathRetraction runs on the step list |
| semantic carrier choice (S / K / W / B) | **semantic plan → BIND** | carrier_sufficiency; refusal when no carrier is sufficient |
| ordered projections available, direction availability | **BIND metadata** | `OrderedLaneWitness`; JanusGraph unidirected labels |
| parameter slot; pinned-or-not | **BIND** (slot) / **BUNDLE** (pin = value read by a decision) | GValue pinning |
| scan vs bounded slice vs projection | **BUNDLE** | `PrefixLowering`; JanusGraph scoring |
| fold barrier / merge point placement | **BUNDLE**, legal only where liveness permits | LazyBarrier guards |
| survivor gating, bit vs word schedule | **BUNDLE** | D-GATED-GATHER-0 |
| SIMD width, kernels | **backend** | — |

## 12. ISA sprinkle candidates

None earns an opcode. Every harvested mechanism is a carrier choice (S/K/B), BIND
metadata (ordered projection, direction availability), a BUNDLE strategy (slice,
gating, merge placement), or a backend kernel. The candidate "BARRIER" op is
rejected: in mask execution every op is already a barrier.

## 13. Rejected harvests

- LazyBarrier as a mechanism (F6). - JanusGraph multiQuery/prefetch (F7).
- FilterRanking's static rank table (ANDed masks commute; the gated lowering's
  order matters only for `Pred` and is measured, not ranked).
- Full-copy edge duplication (local projection lanes are cheaper).
- A plan cache that re-runs strategies per execution (TinkerPop's only cache is
  a parse cache; ours already skips re-planning).

## 14. Experiments

| | status |
|---|---|
| D-TJ-0 weighted multiplicity frontier | DONE by exhaustive sufficiency check (§10) |
| D-TJ-1 legal fold barrier | DONE as sufficiency: path/earlier-var consumers force B (illegal merge); terminal consumers allow K (legal). Not yet as executed code |
| D-TJ-2 incident→adjacent | PARTIAL: parallel-edge fixture pins 4 walks vs 2 targets; the fused (edge-dead) route is the `TerminalSet` path |
| D-TJ-3 bounded slice | DONE (30–38×) |
| D-TJ-4 prepared binding | DONE earlier (patch 10.4 µs vs DataFusion 1117 µs); pinning rule from §6 is new |

## 15. Final architecture ruling

| Law | Verdict | Witness |
|---|---|---|
| 1. Folding is the elimination of distinctions no future consumer can observe | **HOLDS** | carrier_sufficiency: K suffices for terminal counts, B is forced for earlier-variable questions; TinkerPop's merge key is exactly the live state |
| 2. BIND determines which distinctions must remain representable | **PARTIAL** | the semantic plan decides liveness (`demand::classify`); BIND only adds layout legality (`dst_ordered` / witness) |
| 3. BUNDLE determines how aggressively surviving distinctions are executed | **HOLDS** | bundle_probe 32/32 equal; gated gather; bounded slice |
| 4. V4 should express operations, not optimisation artifacts | **HOLDS** as a rule, **violated** by today's R2IL fold band (mirrors mask-risc) |
| 5. Storage access can be folded when binding proves regions cannot contribute | **HOLDS** | bounded_slice 30–38× under an ordering fact; `OrderedLaneWitness` → `Cmp::Range` |

KEEP
- Demand/liveness in the semantic plan (`demand::classify`); the carrier ladder S ⊂ K ⊂ W ⊂ B, each step down legal only when the sufficiency check says so.
- Edge-table re-anchoring for walk counts (TinkerPop's AdjacentToIncident, already ours) and `OrderedLaneWitness` → `Cmp::Range` for ordered access.
- Typed refusal where no sufficient carrier exists; no new V4 opcode for any harvested mechanism.

BUILD
- `Gather { under }` plus a branchless `mask_gather_u32` (D-GATED-GATHER-0, measured 3.8× from the branch, up to 1 000× from gating at low density).
- An ordering witness for scalar property lanes, and replace cypher-quack's `dst_ordered: bool` with it.
- Parameter pinning in prepared programs: record which slot values a BIND/BUNDLE decision read; only those invalidate.

KILL
- LazyBarrier as a mechanism, a BARRIER opcode, multiQuery/prefetch batching, full-copy edge duplication.
- A WeightedSupport type separate from `GroupReduce{Local(dst), Count}`.
- Plan caches that re-run optimisation per execution.
