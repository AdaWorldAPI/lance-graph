# Epoch-indexed temporal graph: research agenda and falsification plan

**Status:** PROPOSED research agenda / companion to [outer-membrane graph lab](OUTER_MEMBRANE_GRAPH_LAB_VERTICAL_SLICE.md), not a shipped temporal explorer  
**Date:** 2026-10-13  
**Question:** Can epoch-indexed semantic edges, sparse deltas and resident mask folds drive a KGExplorer-style timeline at less cost and complexity than repeated materialization of graph snapshots?  
**Boundary:** Reuse existing LanceGraph identity, temporal, seal, planner, SoA, Quack, mask-risc, callcenter and z8run machinery. No second data engine or rewritten ontology.

## 0. Decision hypotheses, explicitly not settled

1. **Do not persist an edge at every clock tick.** Start with stable canonical edge identity, a validity interval [start, end) **or** sparse BORN/SET/DELETE changes attached to a coherent commit/cycle. Compare both choices experimentally. Neither is yet ratified as the universal graph ABI.
2. **Keep time as a query/projection coordinate first.** Epoch selection composes with HHTL, ClassView, RBAC, NAL evidence and adjacency masks, rather than forcing all codepaths to allocate time-specific graph copies.
3. **Three axes of time must remain distinguishable:** domain/valid time (when an assertion holds), transaction/knowledge time (when storage knew it), and cognitive/revision time (when evidence or Kanban changed). A wall-clock Unix epoch, HLC tick, Lance dataset version and sealed cycle are **not interchangeable**.
4. **Preserve the existing authority of the seal.** It fixes cross-owner total order, coalescing, cohort membership and sealed read horizon. Per-owner temporal trajectories offer a partial order and cannot manufacture those facts after the seal.
5. **Use the existing physical executor.** Temporal intersection/select/union/anti masks should lower through applicable existing Quack/mask-risc paths. SIMD/intrinsics/assembly are options **behind** the IR. A view/controller is not a new temporal execution engine.
6. **External observations are just views over canonical state.** Transwarp KGExplorer is a temporal-graph *interaction reference*, not a storage/algorithm dependency; the Rust [vadim-su/stellardb](https://github.com/vadim-su/stellardb) is a separate project and only a reference for graph-result projections and front-door admission.
7. **No new carrier before auditing existing carriers.** Read [identity/temporal/evidence primer](../../.claude/knowledge/identity-temporal-evidence-primer.md) and [temporal archaeology](../../.claude/temporal/README.md) first. Existing SoA, canonical NodeGuid, witness and Lance version are authorities. Do not create an unreviewed serialized side ledger.

## 1. Prioritized exploration checklist

The priority groups are **research sequence**, not permission to implement all items in one PR. Every experiment needs an implementation/control pair and a real negative control.

| ID | Priority | Question to explore | Evidence to read / experiment | Falsifier |
|---|---|---|---|---|
| E01 | P0 | What exactly is an "epoch": Unix ns/ms, HLC, per-owner cast sequence, sealed cycle, or Lance version? | Audit [planner temporal](../../crates/lance-graph-planner/src/temporal.rs), [temporal POV](../../crates/lance-graph-contract/src/temporal_pov.rs), [seal vs temporal](../../.claude/knowledge/seal-vs-temporal-ordering-information.md). Produce typed mapping with examples. | Mapping conflates two clocks or silently imposes a total order on incomparable owners. |
| E02 | P0 | Which time facts and deletion semantics are **already** in code? | Read [temporal audit](../../.claude/temporal/README.md), [deletion/tombstones](../../.claude/temporal/05-deletion-and-tombstones.md), [identity primer](../../.claude/knowledge/identity-temporal-evidence-primer.md), [versioned graph](../../crates/lance-graph/src/graph/versioned.rs). | "Already shipped" claim has no non-test call path / falsifiable probe. |
| E03 | P0 | What is the globally coherent graph read horizon? | Compare per-dataset versions to sealed-cycle frame/version and witness positions. Probe nodes/edges/epistemic projections together. | Reader combines nodes from cohort N with edges/evidence from N+1, or uses dataset-version equality as if it were a global commit id. |
| E04 | P0 | Is there a reliable sparse delta source on current pinned Lance? | Re-run [measured delta API notes](../../.claude/temporal/01-delta-api-lance11.md) against **current** Lance 13; reproduce red/green controls, especially physical vs stable row IDs. | Insert/update/delete records vanish, or stable IDs silently migrate. |
| E05 | P0 | Can one identify removal exactly, not just detect a changed seal? | Verify current `GraphDiff` fields and [deletion audit](../../.claude/temporal/05-deletion-and-tombstones.md). Compare explicit tombstone against version difference. | A deleted edge becomes indistinguishable from "never existed", or replay resurrects it. |
| E06 | P1 | Edge encoding: lifetime intervals vs sparse event deltas vs baseline snapshots? | Same synthetic graph with hot/churn/sparse patterns, compare storage bytes, commit count, playback and cold jump. | One alternative is dominated; retention breaks exact replay; version counts diverge without documentation. |
| E07 | P1 | How to represent half-open validity and unknown/open ends? | Pure reference implementation for [from, to), open-ended, empty, negative/late corrections, concurrent writes, integer overflow, timestamp units. | Off-by-one visibility or mixing 'unknown' with infinity; invalid intervals accepted. |
| E08 | P1 | How to retrieve only incident temporal edges? | Compare [container adjacency](../../crates/lance-graph-cognitive/src/container_bs/adjacency.rs), HHTL buckets, typed edge labels, reverse/forward indexes and scoped masks against a scan oracle. | Missing multi-edge, high-degree, reverse, duplicate-label or cross-basin edges. |
| E09 | P1 | Which query rewrites are mathematically safe? | Domain-agnostic interval algebra: intersection, overlap, complement, range count (only with exact coverage), prefix/run materialization, preimage. | Null/unknown, bag semantics, overflow, duplicate rows or unequal scopes violate a rewrite. |
| E10 | P1 | Can a sequence of nearby epochs replay via deltas? | Forward and backward cursor sweeps vs independent snapshot reads. Define inverse only where information preserved. | Applying deltas changes results relative to direct read or cannot restore a removed edge. |
| E11 | P1 | When does a checkpoint beat a resident fold? | Snapshot+delta cost vs indexed interval mask vs resident append-and-fold at varying churn and history. | "O(changed edges)" claimed with hidden O(all edges) rebuild, decode, copy or search. |
| E12 | P1 | How do late-arriving valid-time assertions affect "as known then"? | Bi-temporal tests: valid on Mon, recorded Wed, corrected Fri, revoked Sun; replay with reader horizon pinned to Tue/Wed/Fri. | Tuesday query can see Wednesday knowledge; correction overwrites original history. |
| E13 | P1 | How does NAL evidence change without mutating historical truth? | Model revisions as witnessed cognitive events or projections on existing canonical carriers; distinguish NARS f/c from ontology edge existence. | Confidence update implies false edge creation/removal or merges independent witnesses. |
| E14 | P1 | How do Rubicon phase epochs relate to edges and reader horizon? | [kanban contract](../../crates/lance-graph-contract/src/kanban.rs), [planner](../../crates/lance-graph-planner/src/lib.rs), sealed-cycle driver. | A visual slider/tick commits a phase transition, or planned/held stages become "executed". |
| E15 | P1 | Where do counterfactual/inverted temporal probes belong? | [inverted temporal ablation plan](../../.claude/plans/inverted-temporal-ablation-reasoner-oracles-v1.md), [temporal search](../../crates/lance-graph-cognitive/src/search/temporal.rs); test shuffled/reversed/late evidence. | Reversed time produces causal support indistinguishable from correctly ordered evidence; a forecast is promoted to fact. |
| E16 | P2 | Which path is faster: algebraic elimination, register reuse, scalar or SIMD mask fold? | Quack/mask-risc differential tests, inspect actual generated code, benchmark full chain across scalar/SSE/AVX2/AVX-512 available hosts. Rust 1.99 `asm!` register operands are a probe, not automatic speedup. | No measurable win over compiler optimizations; setup/spill or bandwidth dominates; cross-target output differs. |
| E17 | P2 | Can the timeline use compact data-plane handles across graph-flow/z8run? | [graph-flow-lance](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/crates/graph-flow-lance/src/lib.rs), [z8run-lance](https://github.com/AdaWorldAPI/z8run/blob/main/crates/z8run-lance/src/lib.rs) and generation checks. | Stale result handle resolves against a newer population or materializes all rows in a message. |
| E18 | P2 | How should external read/write and backpressure be gated? | [callcenter](../../crates/lance-graph-callcenter/src/lib.rs), [supervisor skeleton](../../crates/lance-graph-supervisor/src/supervisor.rs), [shader LAB REST](../../crates/cognitive-shader-driver/src/serve.rs). | Unauthorized history leaks, unbounded queue, or "200 OK" while dispatch returned `DispatchNotImplemented`. |
| E19 | P2 | What should the temporal KG UI actually display? | [Transwarp KGExplorer dynamic timeline](https://www.transwarp.cn/doc/stellardb/5.1/developer-guides--introduction_to_kgexplorer--dynamic-timeline), [Transwarp product](https://www.transwarp.io/product/stellardb), [Rust StellarDB graph projection](https://github.com/vadim-su/stellardb/blob/main/station/src/lib/station-graph.ts). | UI shows an edge without authorized endpoints, fabricates source timestamp, or equates past cognition with present belief. |
| E20 | P2 | Can hydration, caching and retention preserve historical queries? | [S3 hydration doctrine](../../.claude/knowledge/s3-hydration-lifecycle.md), [hydration crate](../../crates/lance-graph-hydrate/src/lib.rs), [cache audit](../../.claude/knowledge/lance-cache-surface.md). | Old pinned version unavailable after retention; dirty artifact evicted; Moka read-path ownership unproven. |

**First action for CC:** produce a `SHIPPED / TEST-ONLY / DORMANT / PROPOSED` table for E01–E05 on current `main`. The September archaeology was measured on Lance 11; it must not be extrapolated to current Lance 13 without a new probe. A doc-comment, test-only caller or future-plan code snippet does not count as a live integration.

## 2. Minimal temporal algebra (illustrative, not minted ABI)

For a stable edge identity with interval `[valid_from, valid_to)` and a selected valid epoch `t`:

```text
active_at(t)     := valid_from <= t AND t < valid_to
overlaps([a,b))  := valid_from < b AND valid_to > a
covers([a,b))    := valid_from <= a AND b <= valid_to
```

Preconditions: `a < b`, the correct time axis, aligned units, meaningful open-end handling, and an interval model in which a missing end means something explicit. The overlap expression for an empty requested interval must return false; never infer its answer from the expression without the precondition. Different predicates are appropriate for "ever during" (overlaps), "throughout" (covers), and "at instant" (active).

Add transaction/knowledge horizon `v` for a bitemporal projection when a matching transaction interval exists:

```text
visible(v,t) :=
    known_from <= v AND v < known_to
    AND active_at(t)
    AND HHTL_scope
    AND policy_admits
    AND evidence_visibility_policy
```

There is no claim that existing Lance datasets persist `known_to` for every edge. The existing `QueryReference` / `EpistemicMode` already define richer reader-horizon rules, including knowability, and must remain the authority. A proposed "three-axis" view adds cognitive revision horizon `c` when needed, not an arbitrary fourth timestamp in every edge payload. **The internal registers remain byte-layout/ISA contracts; interval, CE64, NAL and SPO are typed interpretations rather than immutable wire meanings of 8-bit lanes.**

**Event encoding alternative:** `(edge_id, coherent_cycle, op, witness_ref, changed_fields)` for `BORN / SET / DELETE / REVISE`, as a *candidate* reading over existing sparse append/SoA storage. Do not mint an independent "truth ledger": audit [split-tunnel sparse append / resident fold](../../.claude/knowledge/split-tunnel-sparse-append-resident-fold.md) first, whose reference implementation is measured in **HubSPO-rs**, not generally in LanceGraph. The recorded rule "one batch = one Lance version" requires single-writer and race/idempotency proofs. Its known append race is still open.

**A correction to the shorthand `G_(e+1) = G_e XOR Delta`:** symmetric difference is valid for exact toggle-membership sets, but not automatically for reweighted edges, NAL revisions, multi-edges, corrections or tombstones. For generic temporal playback use `apply(G_e, ordered_delta_e)` with explicit operations. Establish an invertible inverse only when the delta stores the previous state. Do not use XOR to destroy an old witness.

**Mask form:** `M_time AND M_HHTL AND M_policy AND M_evidence` remains a *typed* selection result. No scalar reduction is required just to pass a bitset into graph adjacency. Converting it to `Count` is an explicit terminal operation, after provenance and exactness checks.

## 3. Time and ordering: specific traps

- **Unix epoch time versus sealed cycle.** A Unix epoch can be numeric and sortable but does not prove causal ordering, especially across hosts, skew, delayed ingestion and equal timestamps. HLC can partially order observations but `(server_id, hlc_tick)` is not automatically the sealed cohort ID.
- **Total order versus per-owner partial order.** The [seal analysis](../../.claude/knowledge/seal-vs-temporal-ordering-information.md) demonstrates information not present in `temporal.rs`: cross-owner arrival order, last-writer coalescing, cohort membership and predecessor read horizon. Losing those changes answers, even when graph edges still exist.
- **Storage coherence.** [`VersionedGraph`](../../crates/lance-graph/src/graph/versioned.rs) stores nodes, edges and fingerprints as distinct datasets. A `GraphDiff` contains added nodes, modified nodes and added edges but does not provide symmetric deletion IDs. Version comparisons need a coherent cohort mapping, not assume each dataset advanced in lockstep after arbitrary writes.
- **Deletion.** A tombstone distinguishes "not yet known", "removed", "expired validity", "retracted claim" and "blocked by current policy". They may look identical in a simple graph visualization but have different epistemic semantics. Do not clean up Lance versions without demonstrating retention requirements.
- **Forecasts.** A future time slice may show Chronos/perturbation *scenarios* separately from observed evidence. Per the [holographic temporal access note](../../.claude/knowledge/holographic-temporal-access-projection.md), a forecast cannot promote a fact or NAL witness; UI must label it.
- **Privacy and tenancy.** As-of queries must be authorized under a defined policy snapshot or current enforced policy. Never infer "historically accessible" as "accessible to the current requestor" without a security decision.

## 4. Query-planning and performance probes

Three competing implementations on identical datasets (no benchmark claimed until run):

| Strategy | Point query, many epochs | Adjacent playback | Cost/restrictions |
|---|---|---|---|
| A. Scan intervals into Quack/mask-risc masks | Bounded per scan and mask composition | Potential rescans | Good simple baseline; prove vectorization and cost |
| B. Index by HHTL/edge/predicate/interval extents | Selective random access if index appropriate | May still rebuild per epoch | Index maintenance, high-degree nodes, interval overlap conditions |
| C. Resident sparse changes + checkpoint | Checkpoint + replay delta | Potential O(changes) apply | Requires stable order, retention/rehydration, memory budget, tombstone and collision safety |

**Benchmark dimensions:** N edges (1e3 / 1e5 / 1e6 where feasible), edge churn fraction (0.001%, 1%, 30%), temporal-selectivity, average/max degree, HHTL locality, number of epochs, point vs jump vs playback, one/many predicates, one/many consumers, cache cold/warm, single-writer/reader contention. Include distinct valid-time and transaction-time distributions; a fast dense sequential case alone is not evidence of an efficient point query.

**Metrics:** output correctness first, commit bytes/version count, index/update cost, replay/jump latency p50/p95/p99, allocations, RSS, cold hydration IO, CPU instructions/branch behavior, SIMD lane residency, bytes read, end-to-end frontend frame time, and bounded memory under backpressure. Keep build CPU tier and exact compiler/flags visible. The previously mentioned **0.6 ns** was a *hypothetical target*, not a measured LanceGraph epoch-fold time; any measured operation must exclude neither allocation nor data movement just to sound fast.

**Optimization order:** (1) algebraically eliminate the query if exact witness/extents suffice; (2) reuse already resident/compiled mask or register; (3) MOV-like/no-op projection when legitimate; (4) SIMD `VPTERNLOGQ`/vector Boolean combination on sufficiently large or already resident masks; (5) scalar reduction only when requested. Test auto-vectorized Rust, intrinsics and opt-in assembly; never infer improvement from Rust register-class support without measuring emitted instructions and full data movement. The `mask-risc` examples `program_collapse_probe.rs`, `ternlog_fused_probe.rs` and `extent_probe.rs` are existing measurement entry points.

**Falsification rule:** compare every strategy with a scalar reference that evaluates the original temporal predicate and edge/witness semantics. "Fast" but incorrect on late corrections, deleted edges, empty interval, multigraph, reverse traversal, overflow or mixed horizon is rejected.

## 5. Interactive temporal explorer: presentation contract

Use [Transwarp StellarDB KGExplorer `6.3.6](https://www.transwarp.cn/doc/stellardb/5.1/developer-guides--introduction_to_kgexplorer--dynamic-timeline) as an *interaction reference*: a dynamic graph timeline with time-selection controls. The accessible documentation confirms that section and its timeline configuration/case-export headings; detailed behavior of any particular widget should be validated in a running product rather than treated as proof of an API. The unrelated [Rust StellarDB Station](https://github.com/vadim-su/stellardb/blob/main/station/src/components/station-graph.tsx) provides a reference for result-to-2D/3D graph projections.

**UI proposal (not shipped):**

- Timeline with a **typed axis selector**: valid time, as-known/version, cognitive revision; fixed UTC timestamp-unit and local display timezone.
- Point-in-time **snapshot** mode and interval **overlaps** / **throughout** modes; start/end picker; visible definition always explicit.
- Same two canonical NodeGuids and single explicitly witnessed relation from the original lab slice; visualize an edge BORN → REVISED evidence → DELETE and demonstrate that independent timestamps need not coincide.
- Node/edge inspector: stable identity, semantic ClassId/predicate, source and destination, exact witness/provenance, source version / coherent cycle, validity, NAL f/c when available, RBAC visibility, and visible/retracted/expired/unknown distinctions. Never synthesize durable IDs from row position.
- Graph playback can highlight **added / removed / revised** changes, not implicitly create persistent edges or make an unobserved fact true. A "forecast" overlay is visually and semantically distinct.
- Rubicon/Kanban panel renders historical phases but does not execute work when scrubbing. `z8run-lance`/`graph-flow-lance` receive versioned handles, not a materialized population. Terminal materialization only when explicitly requested.
- "Why is this hidden?" diagnostics must not reveal unauthorized rows or attributes; return safe reason categories instead of leaking inaccessible metadata.

**Minimum UI oracle:** after changes at event epochs 100 (add), 104 (NAL revision), 120 (remove), snapshots at 99/100/103/104/119/120/125 must agree with the reference. Separate tests with delayed ingestion and as-known cutoffs prove that replaying valid time does not leak future-arriving knowledge. Add two disjoint intervals and one empty interval as negative tests.

## 6. Integration sequence and deliverables

**T0: Documentation / archaeology (this PR).** Catalog exact types and callers; do not write new epoch structs, source code, REST endpoints or UI. Add decision log of what is truly shipped, tested-only, dormant or proposed.

**T1: Correctness oracle.** Pure, independent interval/event reference over tiny fabricated graph (two nodes/one edge then high-degree graph); signed overflow, empty intervals, open end, late correction, tombstones, as-known leaks, one/many owner sequences. Specify explicit version-cohort map.

**T2: Storage/access experiment.** Compare A/B/C above using current Lance dependency and existing Quack/mask-risc probes, with source-state parity and deletion/identity proofs. Do not disable version retention or rewrite the existing SoA ABI.

**T3: Temporal query binding.** Prototype a *typed* request (reference to snapshot + temporal mode + access policy) lowered into the already-owned execution path. Do not conflate graph topology with temporal predicates. Quack query-specific lowering belongs Quack; any universal Set/interval algebra must be usable independently of Quack.

**T4: Membrane admission & audited authority.** Reuse [callcenter](../../crates/lance-graph-callcenter/src/lib.rs), bounded/authorized request ingress and supervisor actor routing. The existing supervisor presently returns `DispatchNotImplemented` on non-health envelopes and does not apply mailbox capacity in `spawn_linked`; [realtime drain](../../crates/lance-graph-callcenter/src/drain.rs) is unbounded. Prove these are fixed or **label lab-only**. No fabricated 200/commit.

**T5: Timeline visualization + cross-client handle parity.** Drive a data- and permission-bounded 2D/3D view through a source-generation-pinned `lance-abi` handle. Show Quack actual execution, independent reference result and a graph overlay. Validate any contract changes against consumers (OGAR, MedCare, HubSPO, z8run, graph-flow).

**T6: Scale, recovery and ablation.** Probe random jumps, sequential playback, late arrivals, rollbacks, crashing before/after seal, tenant changes, cache eviction/rehydration, temporal inversion, timeout/saturation and memory. Re-run under current Rust/Lance pin with disabled-optimization controls to invalidate benchmark artifacts.

**Stop conditions:** (a) cannot name which cohort a displayed edge belonged to; (b) false temporal knowledge propagation; (c) missing deletion identity; (d) misaligned ClassId/NodeGuid/HHTL identities; (e) second authoritative store/evaluator; (f) unbounded untrusted ingress; (g) no baseline controls; (h) changes to stable byte layout without compatibility plan.

## 7. Source map and concrete follow-up questions

| Owned surface | Existing starting point | Mandatory question before implementation |
|---|---|---|
| Temporal reader POV | [`temporal_pov.rs`](../../crates/lance-graph-contract/src/temporal_pov.rs) | Can current `VersionRange` express required half-open storage horizon without claiming domain-valid time? |
| Epistemic/causal time | [`planner/src/temporal.rs`](../../crates/lance-graph-planner/src/temporal.rs) | Is caller wiring active beyond tests; how do `server_id` / HLC / `knowable_from` interact with the snapshot? |
| Sealed cycle | [`seal vs temporal note`](../../.claude/knowledge/seal-vs-temporal-ordering-information.md), [`persist sink`](../../crates/lance-graph-planner/src/persist_sink.rs) | How is global order/arrival/coalescing and predecessor version preserved in a temporal mask query? |
| Lance deltas | [`temporal 01`](../../.claude/temporal/01-delta-api-lance11.md), [`temporal 09`](../../.claude/temporal/09-plan.md) | Do earlier measured red insert/update probes still hold for current Lance 13? |
| Graph version and tombstones | [`VersionedGraph`](../../crates/lance-graph/src/graph/versioned.rs), [`temporal 05`](../../.claude/temporal/05-deletion-and-tombstones.md) | What is deletion's durable identity, and when do independent dataset versions split? |
| Sparse append and resident fold | [`split-tunnel research`](../../.claude/knowledge/split-tunnel-sparse-append-resident-fold.md) | Can the HubSPO-rs proven case generalize without violating single writer / retention / one-version-per-batch guarantees? |
| HHTL and edges | [`hhtl.rs`](../../crates/lance-graph-contract/src/hhtl.rs), [`adjacency.rs`](../../crates/lance-graph-cognitive/src/container_bs/adjacency.rs) | How does epoch selection restrict a stable graph relationship without equating address prefix with semantic edges? |
| Actual compute | [`mask-risc`](../../crates/lance-graph-mask-risc/src/lib.rs), [`quack`](../../crates/lance-graph-quack/src/lib.rs) | Can masks remain non-scalar all the way through the query plan, and when does SIMD win? |
| NAL/reasoning | [`lance-graph-nal`](../../crates/lance-graph-nal/src/lib.rs), [`inverted temporal ablation`](../../.claude/plans/inverted-temporal-ablation-reasoner-oracles-v1.md) | Do retrospective evidence and counterfactuals remain non-factual until validated? |
| Temporal projections | [`holographic temporal access`](../../.claude/knowledge/holographic-temporal-access-projection.md) | Can temporal selection reuse existing coordinate projection without allocating a time-expanded graph? |
| External consumers | [`callcenter`](../../crates/lance-graph-callcenter/src/lib.rs), [`supervisor`](../../crates/lance-graph-supervisor/src/supervisor.rs), [`LAB REST`](../../crates/cognitive-shader-driver/src/serve.rs) | Is the ingress really authorized, bounded and routed, rather than a scaffold? |
| User interface | [Transwarp KGExplorer timeline](https://www.transwarp.cn/doc/stellardb/5.1/developer-guides--introduction_to_kgexplorer--dynamic-timeline), [Rust StellarDB Station](https://github.com/vadim-su/stellardb/blob/main/station/src/lib/station-graph.tsx), [`z8run-lance`](https://github.com/AdaWorldAPI/z8run/blob/main/crates/z8run-lance/src/lib.rs) | How can a date slider change a query handle without triggering cognition or minting an edge? |

**PR rule:** Keep this document and its companion architecture note as **proposals grounded in evidence**. Any implementation requires a separate, reviewable PR with a failing negative control demonstrated before a fix, measured costs, and ABI invariants.

