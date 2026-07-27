# Trace: zero-copy discriminator violation census — BeliefArena reasoning cycle

Read in full (depth=full, no snippet reads): `nars/belief.rs` (476 ln),
`nars/tactics.rs` (821 ln), `nars/insight.rs` (416 ln), `nars/truth.rs` (154 ln,
pulled in because `TruthValue` arithmetic is the f32 surface every tactic
touches), `adjacency/mod.rs` (32 ln), `adjacency/csr.rs` (248 ln),
`adjacency/batch.rs` (134 ln), `adjacency/properties.rs` (82 ln),
`adjacency/distance.rs` (91 ln), `adjacency/propagate.rs` (128 ln),
`physical/accumulate.rs` (364 ln, pulled in because `adjacent_truth_propagate`
dispatches through `dyn Semiring` / `SemiringValue` — the actual add/multiply
arithmetic lives there, not in `propagate.rs`).

## Allocation inventory (reasoning path)

Scope: `BeliefArena` (belief.rs), the five tactics (tactics.rs), the S10
insight cycle (insight.rs), and `adjacent_truth_propagate` + its `Semiring`
dispatch (adjacency/propagate.rs + physical/accumulate.rs). "Canonical state"
= the population of `Belief`s living in `BeliefArena::entries` (and, for
adjacency, the `AdjacencyStore` CSR/CSC arrays once built).

| Site (file:line) | What it holds | Class | Lifetime |
|---|---|---|---|
| `belief.rs:130` `BeliefArena.entries: Vec<Belief>` | THE canonical population — every belief ever admitted | **CANONICAL STATE (not a copy — this is the arena itself)** | persistent (arena lifetime) |
| `belief.rs:131` `BeliefArena.index: HashMap<CStmt,u32>` | statement → arena-index lookup, mirrors `entries` | **CANONICAL STATE (index, not a copy of belief content)** | persistent |
| `belief.rs:100` `Belief.premises: Vec<u32>` | per-belief premise pointer list (0, 1, or 2 entries typically) | KERNEL-SCRATCH-shaped but PERSISTENT (stored per belief, not transient) — see note below | persistent (per-belief) |
| `belief.rs:250` `e.premises = premises.to_vec()` (`admit_derived`, existing-entry overwrite branch) | clones the caller's `&[u32]` premise slice into the belief's own `Vec` | **FORBIDDEN-COPY of caller-owned data ONLY IF `premises` is itself derived from canonical state elsewhere** — see AMBIGUOUS #1 below | per-call (alloc), persistent (stored) |
| `belief.rs:262` `premises: premises.to_vec()` (`admit_derived`, new-entry branch) | same clone, on the fresh-insert path | same as above | per-call / persistent |
| `belief.rs:285` `let mut by_sc: HashMap<(u16,Copula),Vec<u32>> = HashMap::new();` (`close_transitive`) | pivot table: `(subject,copula) → [arena indices]`, rebuilt from `self.entries` EVERY PASS | **FORBIDDEN-COPY — a full reconstructed index of canonical-state indices, rebuilt from scratch every `while` iteration** | per-pass (rebuilt each of up to `max_passes` loops) |
| `belief.rs:288-291` `.or_default().push(i as u32)` inside the by_sc loop | per-key `Vec<u32>` growth, N pushes total = N entries scanned | same bucket as above (part of the same reconstruction) | per-pass |
| `belief.rs:296` `let mut derived: HashMap<CStmt,(TruthValue,u32,[u32;2])> = HashMap::new();` | this pass's proposed-derivation table, keyed by derived statement, keeping only the max-expectation candidate | **OUTPUT-BUFFER** (new derived values, not a copy of anything that already exists in the arena — though some entries in `derived` duplicate statements ALREADY in `entries`, discovered only via `admit_derived`'s own lookup) | per-pass |
| `belief.rs:329` `for (stmt, (truth, rung, premises)) in derived` — the `premises` here is `[u32;2]` (Copy, stack) not heap | consumed directly by `admit_derived`, which re-clones it (`&premises` → `.to_vec()`, see above) | KERNEL-SCRATCH (Copy array, no separate alloc) | per-pass |
| `belief.rs:363` `TruthValue::new(0.9,0.9)` etc. (test-only, `#[cfg(test)]`) | test fixtures | N/A (test code) | test scope |

| `tactics.rs:161-168` `inh_predicate_indegree`: `let mut deg: HashMap<u16,usize> = HashMap::new();` | full re-scan of `arena.entries()`, one accumulator entry per distinct predicate | **FORBIDDEN-COPY — a derived index recomputed from canonical state on every call** (called once per `rcr_abduce` AND once per `cas_abstract` invocation — not cached, not incremental) | per-call |
| `tactics.rs:181-186` `rcr_abduce`: `let mut by_pred: HashMap<u16,Vec<u32>> = HashMap::new();` | predicate → arena-index list, full re-scan of `arena.entries()` | **FORBIDDEN-COPY — same shape as belief.rs's `by_sc`, rebuilt every call, not shared/cached across the RCR + CAS + closure call sites even though all three build near-identical `predicate → indices` maps** | per-call |
| `tactics.rs:191` `let mut preds: Vec<u16> = by_pred.keys().copied().collect();` | materializes the map's keys to sort them (determinism requirement, see rcr_floor_and_budget below) | KERNEL-SCRATCH (small, `Copy` u16 keys, needed for the sort) — arguably OUTPUT-BUFFER since it's a derived ordering, not a copy of belief CONTENT | per-call |
| `tactics.rs:337-344` `inh_by_subject`: `let mut m: HashMap<u16,Vec<(u32,u16)>> = HashMap::new();` | subject → `(arena idx, predicate)` list, full scan of `arena.entries()` | **FORBIDDEN-COPY — third independent full-arena re-index (deg / by_pred / by_subj), each call rebuilding what the previous already built** | per-call (once per `cas_abstract` call) |
| `tactics.rs:151-156` `Frontier { candidates: Vec<Candidate>, gaps: Vec<ReasoningGap> }` | the tactic's proposed-but-unadmitted output | **OUTPUT-BUFFER** — new derived candidates, genuinely not-yet-in-arena content | per-call, returned to caller |
| `tactics.rs:244-254`, `407-417`, `437-447`, `309-319` (Candidate pushes in rcr_abduce/cas_abstract/tr_diverge) | each `Candidate` is `Copy`-friendly (`CStmt`, `TruthValue`, `[u32;2]`, `u32`, enum) pushed into the `Vec<Candidate>` above | OUTPUT-BUFFER (same Vec as above) | per-call |
| `tactics.rs:613` (test) `let capped_stmts: Vec<CStmt> = capped.candidates.iter().map(\|c\| c.stmt).collect();` | test-only projection | N/A (test) | test scope |

| `insight.rs:137` `arena.entries().iter().fold((0usize,0usize), ...)` | scalar fold, NO Vec/Map allocation — pure aggregation over a borrowed slice | KERNEL-SCRATCH (two `usize` accumulators, stack) | per-call |
| `insight.rs:163` `coherence()`: `arena.entries().iter().filter(...).count()` | scalar count over borrowed slice, no allocation | KERNEL-SCRATCH | per-call |
| `insight.rs:174` `wonder()`: `.iter().map(...).sum::<f32>()` | scalar fold, no allocation | KERNEL-SCRATCH | per-call |
| `insight.rs:186` `confidence_entropy()`: `let mut hist = [0usize; BINS];` | **stack array**, `BINS=10`, not heap | KERNEL-SCRATCH (genuinely register/stack-local, not a copy of population) | per-call |
| `insight.rs:283/296` (tests) `VersionedSnapshot::new(...)` etc. | test scaffolding | N/A | test scope |
| `insight.rs:362-377` (test) `Vec<(u16,u16)> coherent`, `Vec<(u16,u16)> null_edges` | synthetic test fixtures for the null-falsifier | N/A (test) | test scope |

**Note on `insight.rs`: this file is clean.** Every non-test function
(`arena_graph_signals`, `coherence`, `wonder`, `confidence_entropy`, `ratio`,
`detect`, `flow_state`) operates by borrowing `arena.entries()` (a `&[Belief]`)
and folding into scalars or a fixed-size stack array. There is **no
Vec/HashMap reconstruction of canonical state anywhere in the non-test S10
code** — this is the one file of the four that already satisfies the
zero-copy discriminator as written.

| `propagate.rs:25-26` `let mut output: HashMap<u64, SemiringValue> = HashMap::new();` | per-target accumulation table for one `adjacent_truth_propagate` call | OUTPUT-BUFFER (new propagated values keyed by target node — not a copy of `AdjacencyStore`/`Belief` canonical state, though it DOES duplicate `TruthValue`s already stored as edge properties, converted through `SemiringValue`) | per-call |
| `propagate.rs:38-41` `SemiringValue::Truth { frequency: input_truth.frequency as f64, ... }` | **f32→f64 widening COPY of a truth value the caller already owns** (`input_truths: &[TruthValue]`) | FORBIDDEN-COPY-shaped but tiny/Copy-only (two f64 scalars, no heap) — flagged mainly for the f32/f64 boundary crossing, not for being a structural violation | per-edge-visit |
| `propagate.rs:45-52` `store.edge_properties.truth_value(*edge_id)` then rebuilds a second `SemiringValue::Truth{..}` | reads canonical edge-property columns (`EdgeProperties.float_columns["truth_f"/"truth_c"]`, see `properties.rs:54-58`) and re-wraps as a NEW enum value | FORBIDDEN-COPY of canonical edge-truth state, materialized fresh per (source,target) pair with **zero caching across the batch** — the same edge's truth is re-fetched and re-wrapped for every target it reaches | per-edge-visit |
| `propagate.rs:63-79` final `.into_iter().map(...).collect()` | converts the `HashMap<u64,SemiringValue>` output into `Vec<(u64,TruthValue)>`, narrowing f64→f32 | OUTPUT-BUFFER (this IS the function's return value — legitimately new derived data) | per-call, returned |
| `accumulate.rs:112/115` `XorBundleSemiring::zero/one`: `vec![0u64;256]` / `vec![u64::MAX;256]` | fresh 2KB fingerprint allocated on every `.zero()`/`.one()` call (not used by the TruthPropagating path `adjacent_truth_propagate` actually dispatches, but same `Semiring` trait object surface) | OUTPUT-BUFFER / KERNEL-SCRATCH depending on caller — **not exercised by the NARS reasoning cycle** (propagate.rs always constructs `TruthPropagatingSemiring`), flagged for completeness only | per-call |
| `accumulate.rs:121/131` `XorBundleSemiring::add/multiply`: `let result: Vec<u64> = a.iter().zip(b.iter()).map(...).collect();` | full 256-word fingerprint reconstruction per bind/bundle op | **FORBIDDEN-COPY in the XorBundle path** — irrelevant to the current NARS/Belief cycle (that cycle only exercises `TruthPropagatingSemiring`, which is scalar, no Vec) but is the SAME `dyn Semiring` interface, so a future caller routing NARS truth through this trait object would hit it | per-call (not on the live reasoning path today) |
| `accumulate.rs:24` `AccumulateOp.child: Box<dyn PhysicalOperator>` | boxed operator tree | KERNEL-SCRATCH/architecture (not reasoning-path allocation) | persistent (operator tree) |

| `csr.rs` `AdjacencyStore::batch_adjacent` | **FIXED 2026-07-27 (PR #855)** — was: three Vecs filled by `extend_from_slice` from `self.adjacent(src)`/`self.edge_ids(src)`, i.e. a NEW flat copy of a subset of the canonical, already-contiguous `csr_targets`/`csr_edge_ids`. Now returns `AdjacencyBatch::new(self, source_ids)` — a borrowed view, no allocation | **RESOLVED** — the row below records the shape that replaced it. Kept (not deleted) because this audit is append-only and the original classification is what licensed the fix | zero (allocation-free) |
| `csr.rs:127-208` `AdjacencyStore::from_edges`: `csr_offsets`/`csr_targets`/`csr_edge_ids`/`csc_offsets`/`csc_sources`/`csc_edge_ids` Vecs, plus the per-node `pairs: Vec<(u64,u64)>` sort buffers at csr.rs:152-159 and csr.rs:186-193 | ONE-TIME construction of the canonical CSR/CSC store from an input edge list | **CONSTRUCTION, not a copy of existing canonical state** (there is no pre-existing store yet) — the two `pairs` Vecs (one per node, for CSR sort and again for CSC sort) are transient sort scratch, freed at the end of each node's block | build-time only; `pairs` per-node-transient |
| `batch.rs:24-29` `AdjacencyBatch<'a> { store: &'a AdjacencyStore, source_ids: &'a [u64] }` | **FIXED 2026-07-27 (PR #855)** — the owned four-Vec struct was REPLACED (not supplemented) by a borrowed view; `targets_for(i)`/`edge_ids_for(i)` delegate to the store and return `&'a [u64]` straight out of the resident CSR arrays | **ZERO-COPY VIEW** — conforms to the universal zero-copy rule (primer §11/§15). `intersect()`'s result stays owned and legitimately so: it is a join product present in neither input | zero for the view; the join result is an OUTPUT-BUFFER |
| `batch.rs:47-49` `intersect()`: `matched_sources_left`/`matched_sources_right`/`matched_targets: Vec<u64>` | the join RESULT — new tuples not present anywhere before intersection | OUTPUT-BUFFER (legitimately new) | per-call, returned |
| `distance.rs:32` `adjacent_fingerprint_distance`: `let batch = store.batch_adjacent(source_ids);` | **FIXED transitively 2026-07-27 (PR #855)** — inherited the `batch_adjacent` copy; now inherits the borrowed view instead, unchanged at the call site | **RESOLVED** (no source change needed here — the fix was upstream in the constructor) | zero |
| `distance.rs:32` `let mut matches = Vec::new();` | scan-result accumulator | OUTPUT-BUFFER | per-call |
| `properties.rs:12-18` `EdgeProperties { float_columns: HashMap<String,Vec<f32>>, int_columns, string_columns, fingerprint_columns }` | canonical columnar edge-property storage itself | **CANONICAL STATE (not a copy)** | persistent |
| `properties.rs:48-49` `with_nars_truth`: `self.float_columns.insert("truth_f".into(), frequencies)` | takes ownership of caller-supplied `Vec<f32>` — a move, not a copy, UNLESS the caller already held a reference into some other canonical array | AMBIGUOUS #2 (depends on caller — see below) | persistent once inserted |

## AMBIGUOUS

- **#1 (`belief.rs:250,262` `premises.to_vec()`):** whether this is a
  FORBIDDEN-COPY depends on what the caller passes as `premises: &[u32]`.
  From the call sites actually exercised (`close_transitive` at
  `belief.rs:330` and every tactic's `admit_derived` call), the slice is
  always a freshly-built `[u32; 2]` (stack array, e.g. `tactics.rs:251`
  `premises: [r, o]`) — i.e. the SOURCE is already kernel-scratch, so
  `.to_vec()` here is a scratch→persistent promotion (legitimate: a belief's
  premise pointers must outlive the call that derived them), not a
  duplication of canonical population data. Verdict: **not a discriminator
  violation** in the paths actually exercised, but the API signature
  (`&[u32]`) does not prevent a future caller from passing a slice borrowed
  out of `entries` itself, which WOULD be a violation. Recommend the
  BeliefArena migration pin this at the type level (e.g. accept `[u32; 2]`
  by value, matching every real caller) rather than leave `&[u32]` open.

- **#2 (`properties.rs:48` `with_nars_truth`):** `EdgeProperties::new()` is
  always called fresh in `AdjacencyStore::new`/`from_edges` (csr.rs:55,205),
  and the one live call site that supplies real data is the test at
  `propagate.rs:91-94`, which constructs `vec![0.9,0.7]` / `vec![0.8,0.9]`
  literals — i.e. genuinely new content, not a copy of pre-existing canonical
  state. In production this method would be fed from whatever upstream
  ingested the edge truths; whether THAT is a copy depends on code outside
  the read scope of this trace (not found in belief/tactics/insight/
  adjacency). Flagged AMBIGUOUS because the method's shape (take-ownership-
  of-caller-Vec) is fine, but I cannot verify the caller's provenance beyond
  this scope.

- **#3 (`tactics.rs` triple re-indexing — `deg` / `by_pred` / `by_subj`):**
  these three full-arena reconstructions are each individually a
  FORBIDDEN-COPY by the discriminator's letter (each is a `HashMap` mirror of
  `arena.entries()`, rebuilt every call, never cached). Whether the
  BeliefArena migration should eliminate them by (a) maintaining these
  indices incrementally ON `BeliefArena` itself as canonical secondary
  indices (in which case they stop being copies and become part of state),
  or (b) leaving them as throwaway per-call scratch (arguably KERNEL-SCRATCH
  if narrowly read as "local computation, discarded after the call") is a
  design decision, not a fact I can resolve from the code alone. I classify
  them FORBIDDEN-COPY above because they duplicate exactly the same
  information three times across three functions with no sharing, which is
  the discriminator's stated failure mode ("caches of canonical state"), but
  flag the KERNEL-SCRATCH reading as a live alternative for the council.

## The rcr_floor_and budget test lock

Test: `rcr_floor_and_budget`, `tactics.rs:590-624`. Full verbatim quote of the
determinism-critical portion (`tactics.rs:611-618`):

```rust
        // DETERMINISM: predicate iteration is sorted + members are in arena
        // order, so the budget-capped set is a STABLE prefix, not hash-seeded.
        let capped_stmts: Vec<CStmt> = capped.candidates.iter().map(|c| c.stmt).collect();
        assert_eq!(
            capped_stmts,
            vec![inh(2, 1), inh(3, 1), inh(4, 1), inh(1, 2), inh(3, 2)],
            "budget keeps a deterministic prefix"
        );
```

The comment this depends on is at `tactics.rs:187-190`, in `rcr_abduce`,
verbatim:

```rust
    // DETERMINISM: `by_pred` is a HashMap (randomly-seeded iteration). Under a
    // finite budget the set of candidates KEPT depends on which predicates are
    // visited first, so the frontier must be reproducible — iterate predicates
    // in a stable (ascending) order. `members` are already in arena-index order.
```

**Exact property the test locks:** the assertion is on the LITERAL SEQUENCE
of `CStmt`s produced, not just the set or the count. Two nested orderings
must both be preserved for `capped_stmts` to equal that exact `vec![...]`:

1. **Outer order — which predicate `m` is visited first.** `rcr_abduce`
   (tactics.rs:191-192) collects `by_pred.keys()` into `preds: Vec<u16>` and
   `sort_unstable()`s them — this is explicit code, independent of `HashMap`
   iteration order, and is NOT touched by the arena-index question. This
   half survives any reordering of `BeliefArena.entries` as long as the
   predicate values (`u16`) themselves are unchanged.
2. **Inner order — which `(r, o)` pair is visited first/second within a
   given predicate's `members` list.** This is where "members are already
   in arena-index order" is load-bearing: `by_pred[&m]` (tactics.rs:184) is
   built by iterating `arena.entries().iter().enumerate()` — i.e. `members`
   is populated in `entries` admission order (index 0, 1, 2, ... as pushed),
   NOT re-sorted. The nested `for &r in members { for &o in members { ... } }`
   loop (tactics.rs:211-212) then visits pairs in that same admission-order
   sequence, and it is THIS sequence that determines which 5 candidates
   survive the `budget: 5` cap before `BudgetExhausted` fires
   (tactics.rs:232-239).

**What a migration must preserve for this test to stay green, verbatim:**
the members of `by_pred[&m]` (or whatever data structure replaces it) must
be visited in the SAME order they are today — i.e. **ascending arena-index /
admission order** — for a fixed predicate `m`. It is not merely "some
deterministic order"; the asserted vector's specific content
(`[inh(2,1), inh(3,1), inh(4,1), inh(1,2), inh(3,2)]`) was derived by hand
from admission order (subjects 1,2,3,4 observed in that sequence at
tactics.rs:592-598, predicate 9 for all of them, single predicate bucket, so
`members = [0,1,2,3]` in arena order; the nested loop over `r,o` pairs with
`r != o` produces `(1,0)→B=2,A=1→inh(2,1)`, `(2,0)→inh(3,1)`,
`(3,0)→inh(4,1)`, `(0,1)→inh(1,2)`, `(2,1)→inh(3,2)` as the first five in
iteration order — matching the asserted prefix). **If the BeliefArena
migration changes `entries` to be reordered by version, by a different
insertion discipline, or if `by_pred`'s equivalent structure is built by
iterating something other than admission-ordered arena indices (e.g. a
version-sorted BTree, a HashMap without the explicit `preds.sort_unstable()`
step, or a parallel/rayon iterator), this exact assertion breaks** — not
because the SET of 5 candidates changes, but because the PREFIX does.

**If order becomes version-order instead of admission-order:** as long as
"version order" is monotonic with "admission order" for observations (i.e.
each `observe()` call bumps both the arena index AND the version identically,
which is true today since there is no out-of-order insert path in
`BeliefArena`), version-order and admission-order coincide and the test
stays valid unchanged. The test would only need to change if the migration
introduces a scenario where version order and arena-index order can DIVERGE
(e.g. retroactive/out-of-order admission, or a versioned read that
re-orders visible entries by a criterion other than insertion sequence) — in
that case the assertion would need to become either (a) a set-equality
check (`capped_stmts.iter().collect::<HashSet<_>>()` against the same 5
statements, dropping the prefix-order claim), or (b) an explicit statement
of whatever the new canonical order is (e.g. sorted by `(subject, predicate)`
pairs) with the expected vector rewritten to match. I did NOT change the
test, per instructions.

## Adjacency input trace

**`AdjacencyStore::from_edges` signature** (csr.rs:123):
```rust
pub fn from_edges(rel_type: String, num_nodes: u64, edges: &[(u64, u64)]) -> Self
```
It consumes:
- `rel_type: String` — an owned relationship-type label (moved in, not
  borrowed).
- `num_nodes: u64` — a scalar.
- `edges: &[(u64, u64)]` — a **borrowed** slice of `(source, dest)` node-id
  pairs. This is the entire adjacency input; NO truth values, NO edge
  properties are supplied at construction time.

**Every caller found in the read scope** (csr.rs tests + propagate.rs test —
no non-test caller exists in `belief.rs`/`tactics.rs`/`insight.rs`/
`adjacency/*` or `physical/accumulate.rs`):
- `csr.rs:218` `AdjacencyStore::from_edges("KNOWS".into(), 3, &[(0,1),(1,2),(2,0)])`
  — literal edge list, test fixture.
- `csr.rs:230` `AdjacencyStore::from_edges("LINKS".into(), 4, &[(0,1),(0,2),(0,3),(1,3)])`
  — literal, test fixture.
- `csr.rs:239` `AdjacencyStore::from_edges("KNOWS".into(), 3, &[(0,2),(1,2)])`
  — literal, test fixture.
- `propagate.rs:90` `AdjacencyStore::from_edges("CAUSES".into(), 3, &[(0,1),(1,2)])`
  — literal, test fixture.
- `distance.rs:70` `AdjacencyStore::from_edges("KNOWS".into(), 4, &[(0,1),(0,2),(0,3)])`
  — literal, test fixture.

**No production/non-test caller of `from_edges` exists anywhere in the four
target files or their direct dependencies read for this trace.** I searched
`belief.rs`, `tactics.rs`, `insight.rs`, and every file under `adjacency/`;
`from_edges` is exercised ONLY from `#[cfg(test)] mod tests` blocks in
`csr.rs`, `propagate.rs`, and `distance.rs`. **UNDETERMINED beyond this
scope:** where a real caller (outside these files) would source `edges` from
— e.g. whether it is built from `BeliefArena.entries` (a `CStmt`'s `(s, p)`
pair reinterpreted as node ids), from a separate SPO/graph ingestion path, or
from Lance-persisted columnar data. That caller was not found in the files I
was scoped to read; naming it would require reading outside this trace's
assignment (I did not grep beyond the five target paths + the two pulled-in
dependency files `truth.rs` and `accumulate.rs`).

**Are truth values baked from authoritative state or supplied ad hoc?**
Structurally: `from_edges` builds ONLY the topology (CSR/CSC index arrays +
a freshly-`with_capacity`'d, EMPTY `EdgeProperties`, csr.rs:205). Truth
values are supplied entirely separately and later, via direct field
assignment to `edge_properties` (`propagate.rs:91`
`store.edge_properties = EdgeProperties::new().with_nars_truth(vec![...], vec![...])`)
— this is an AD HOC overwrite of the whole `edge_properties` field with a
freshly constructed value, not a call through any `AdjacencyStore` method
that would enforce edge-id ↔ truth-value alignment. There is no code path in
the read scope that populates `edge_properties` FROM `BeliefArena`'s
`Belief.truth` fields — the two systems (BeliefArena's per-statement
`TruthValue`, and AdjacencyStore's per-edge `EdgeProperties` columns) are
structurally disjoint in every file read; nothing here derives one from the
other. **UNDETERMINED:** whether a caller outside this trace's scope wires
`BeliefArena` truth into `AdjacencyStore` edge properties, or whether these
are two independently-fed subsystems in production.

**Does CSR construction happen per query / per version / only in tests?**
Only in tests, as shown above — every `from_edges`/`AdjacencyStore::new`
call found is inside `#[cfg(test)]`. **UNDETERMINED beyond scope:** whether
a non-test call site elsewhere in the workspace builds a store per-query,
per-version, or once-and-cached; not found in the files assigned.

## f32 inventory (legacy-float surface on the reasoning path)

| Field / expression | file:line | Folded/compared where |
|---|---|---|
| `TruthValue.frequency: f32` | `truth.rs:12` | Every tactic's truth arithmetic (deduction/induction/abduction/analogy/revise, truth.rs:57-111); `Belief.truth` (belief.rs:93); `Candidate.truth` (tactics.rs:75); `challenge_target` (tactics.rs:458) |
| `TruthValue.confidence: f32` | `truth.rs:14` | Same call sites as `frequency`; also the sole driver of `evidence_weight()` (truth.rs:46-52) which every `revise()` call depends on |
| `Belief.contradiction: f32` | `belief.rs:103` | `revise_at`: `(b.truth.frequency - new.frequency).abs()` then `.max()` (belief.rs:194-195); read by `insight.rs:140,174` (`wonder()`, contradiction-rate signal) |
| `Snapshot.coherence: f32` | `insight.rs:45` | `detect()`: `after.coherence - before.coherence` (insight.rs:249) |
| `Snapshot.wonder: f32` | `insight.rs:48` | `detect()`: `after.wonder - before.wonder` (insight.rs:250) |
| `InsightMush.insight: f32` / `.mush: f32` | `insight.rs:221,226` | `flow_state()` thresholds (insight.rs:263-270): `> 0.5`, `< 0.3`, `< 0.1`, `< 0.2` — hand-tuned f32 comparison constants |
| `confidence_entropy` bin math | `insight.rs:188` | `(b.truth.confidence.clamp(0.0,1.0) * BINS as f32) as usize` — f32→usize cast, then `f32` Shannon-entropy sum (insight.rs:191-198) |
| `EdgeProperties.float_columns: HashMap<String, Vec<f32>>` | `properties.rs:12` | `truth_value()` (properties.rs:54-58) reads `"truth_f"`/`"truth_c"` as `f32`, immediately widened to `f64` at the `propagate.rs:39-40` call site — **the f32→f64 boundary crossing on the adjacency side** |
| `TruthPropagatingSemiring` internal arithmetic | `accumulate.rs:143-214` | All `f64` (`SemiringValue::Truth { frequency: f64, confidence: f64 }`, accumulate.rs:50-53) — this is where the f32 edge/input truths get promoted to f64 for the semiring add/multiply, then narrowed back to f32 at `propagate.rs:71-74` on the way out. **Net effect: the adjacency propagate path round-trips f32→f64→f32 every call**, a precision-irrelevant but structurally real conversion the palette256 migration would need to either preserve or collapse. |
| `Throttle.c_min: f32` | `tactics.rs:117` | `rcr_abduce`/`cas_abstract` gating: `truth.confidence < throttle.c_min` (tactics.rs:229,396,426) |
| EPS constant | `belief.rs:235` `const EPS: f32 = 1e-6;` | `admit_derived`'s expectation-gain gate (belief.rs:247) |
| `TruthValue::expectation()` | `truth.rs:36-38` | `self.confidence * (self.frequency - 0.5) + 0.5` — f32 arithmetic, called by `admit_derived` (belief.rs:247) and `Snapshot::coherence`-adjacent code (referenced in insight.rs doc comment, not directly called in the non-test path shown, but is the basis the doc comment at insight.rs:36-44 explicitly reasons about and rejected multiplying by) |

No `f64` appears anywhere in `belief.rs`, `tactics.rs`, or `insight.rs`
themselves — the ENTIRE `TruthValue`/`Belief`/tactic/S10 surface is f32. The
only f64 in the traced files is the transient semiring-arithmetic surface in
`accumulate.rs` / `propagate.rs`, which is adjacency-side, not
BeliefArena-side, and round-trips back to f32 before returning.

## UNDETERMINED

1. Whether any non-test caller (outside `belief.rs`/`tactics.rs`/
   `insight.rs`/`adjacency/*`/`accumulate.rs`) constructs `AdjacencyStore`
   from `BeliefArena` content, from a separate ingestion pipeline, or from
   Lance-persisted data — not found in scope; would need a workspace-wide
   grep for `AdjacencyStore::from_edges` / `AdjacencyStore::new` outside
   these files, which was NOT performed (task scope was the five listed
   paths + directly-pulled-in dependencies).
2. Whether `deg` / `by_pred` / `by_subj` (tactics.rs) should be classified
   FORBIDDEN-COPY or KERNEL-SCRATCH is a genuine design-judgment call, not a
   fact — see AMBIGUOUS #3.
3. Whether `EdgeProperties::with_nars_truth`'s caller-supplied `Vec<f32>`
   ever originates from a slice/view into other canonical state elsewhere in
   the workspace (AMBIGUOUS #2) — only test call sites were found, all
   passing literal `vec![...]`.
4. The `rcr_floor_and_budget` test's exact five-element vector was
   reconstructed by hand-tracing the nested nested nested loop against the
   test's own setup (subjects 1..=4, single shared predicate 9); this
   reasoning is included above as verification of the property, but I did
   not execute the test (no cargo runs permitted for this trace) to confirm
   the literal output — this is a traced-by-hand derivation, not a
   ground-truth-verified one.
