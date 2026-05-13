# DECISION_SPO_ARIGRAPH.md

> **Status:** Decided. **Decision:** Option B (federated, two-layer cache). **Date:**
> 2026-05-07. **Branch:** `claude/create-graph-ontology-crate-gkuJG`.
>
> **Authority:** This document is the binding ruling for SPO-1 within the scope of the
> `lance-graph-ontology` crate session. It does not legislate the broader `promote_to_spo`
> bridge work owned by SPO-1 itself, which remains the entropy-ledger row's plan.

## The question

The `lance-graph-ontology` crate must commit to one of three ways the existing two SPO
stores compose, because the registry that hydrates from TTL needs to know which store
its entities will ultimately settle in:

A. **Canonical SPO + ARiGraph as a view.** SPO is the single source of truth.
   ARiGraph's three layers become tagged subsets of SPO triples; ARiGraph's API stays
   but storage delegates downward.
B. **Federated.** SPO and ARiGraph remain separate stores. A planner-IR routing rule
   sends semantic / persistent reads to SPO and episodic / temporal reads to
   ARiGraph. The two layers may exchange data through narrowly-scoped bridges.
C. **ARiGraph canonical, SPO as compatibility surface.** ARiGraph's three-layer model
   becomes the storage substrate; SPO queries get rewritten into ARiGraph traversals.

## The decision: B (federated)

The two stores are not duplicates by design. They serve fundamentally different
operations:

- `lance-graph::graph::spo::*` is **fingerprint-keyed**, columnar, fingerprint+
  HammingMin-semiring. It is built for cold, durable, high-cardinality knowledge —
  the SPO-as-physics view (resonance, palette indices, NARS frequency/confidence
  packed into bytes). Lookups are O(1) by fingerprint.
- `lance-graph::graph::arigraph::triplet_graph` is **string-keyed**,
  `HashMap<String, Vec<usize>>`, episodic. It is built for warm, rapidly-mutating
  working memory with cheap lexical recall — the AriGraph-as-mind view (semantic +
  episodic + cognitive layers, ThinkingStyle activations, episode follows-edges).

That is the workspace's own framing, recorded in
`.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md:245`:

> | **SPO-1** | Stage 3 (×2 distinct purposes, **not duplicates by design**) |
> `triplet_graph` Smart (string-keyed methods); `spo::store` Smart (fingerprint-keyed
> methods) | Add `arigraph::SpoBridge::promote_to_spo(&TripletGraph, gate, &mut
> SpoStore)` — promotes warm string-keyed entries into cold fingerprint-keyed store |

The two stores are an L1/L2 cache pair. Promoting them to a single canonical layer
(option A or C) collapses an architectural distinction the rest of the system has
already absorbed:

- Cognitive cycles write episodic activations into ARiGraph at sub-millisecond
  cadence. Forcing every write to also produce a fingerprinted SPO row would
  serialise the hot path.
- SPO's CamPq UDF, palette indices, and NARS-truth columns are tuned for batch
  scans against million-edge stores. Forcing a string-key index on top would break
  the columnar contract that DataFusion and the planner rely on.
- The existing `SchemaExpander` trait at
  `crates/lance-graph-contract/src/ontology.rs` already produces `ExpandedTriple`s
  that the SPO bridge in `crates/lance-graph/src/graph/spo/ontology_bridge.rs`
  writes. ARiGraph's triplet graph keeps its string-keyed insert path. The two
  paths coexist and have not collided.

## Justification from the recon

Three findings drive the choice:

**Working code already federates.** The SPO store and ARiGraph triplet store live
in sibling modules under `crates/lance-graph/src/graph/`. Each has its own builder,
own storage type, own retrieval API. They share only `TruthValue` (the contract
type). The federated topology is what the workspace built, has tested, and ships
on `main` today. Choosing A or C is a refactor of working code; B is the description
of working code.

**The proposed remedy in the entropy ledger is one-way and additive.** The
ledger's resolution is `arigraph::SpoBridge::promote_to_spo(&TripletGraph, gate,
&mut SpoStore)` — a writer that promotes warm, string-keyed entries into cold,
fingerprint-keyed storage. That is a *gate* between two cache layers, not a
unification. Adopting B preserves it cleanly; A or C dissolves the layering it
gates.

**The ontology crate doesn't need either store as the canonical authority.** The
ontology registry hydrates TTL into `Ontology`/`Schema`/`LinkSpec` values held by a
Lance dictionary table. Those values flow into existing `SchemaExpander`s, which
then write `ExpandedTriple` rows into whichever store the consumer chose:

- The SPO store, via `spo/ontology_bridge.rs`, for cold persistent knowledge.
- ARiGraph's `triplet_graph::insert(...)`, for warm episodic state.

The registry doesn't care which store a particular consumer writes to. It only
needs to answer "given this `bridge_id` and `public_name`, what `SchemaPtr` does
this map to?" That answer is independent of the SPO/ARiGraph layering.

## What this means for the new crate

The decision yields a **zero-change** load on the ontology crate. The crate produces
`Ontology` values; consumers carry them into whichever of the two stores fits the
operation. The crate's own state (the `ontology_dictionary` Lance table) is a third,
independent location: append-only, dictionary-only, never the SPO or ARiGraph
substrate.

Concretely:

- **No new code in `lance-graph::graph::spo/*`.** The existing SPO is unchanged.
- **No new code in `lance-graph::graph::arigraph/*`.** The existing ARiGraph is
  unchanged.
- **No `promote_to_spo` bridge implementation in this session.** The entropy-ledger
  row remains owned by SPO-1 itself; it is unblocked by this decision but not
  closed by it. Closing SPO-1 is a future session's deliverable; the bridge fn
  signature `arigraph::SpoBridge::promote_to_spo(&TripletGraph, gate, &mut
  SpoStore)` is correct as drafted.
- **Ontology hydration produces `Ontology` values.** Whether a downstream consumer
  expands those into SPO triples (cold, durable) or ARiGraph triples (warm,
  episodic) is the consumer's choice, made through the existing `SchemaExpander`
  surface.

## What this rules out

Future sessions arriving with a "merge SPO and ARiGraph into one store" proposal
should bring evidence that overrides the three findings above (working code
federates; the ledger remedy is one-way; the ontology crate doesn't need
unification). Absent that evidence, B stands.

This decision does NOT freeze SPO-1. The entropy-ledger row remains open. Adding
the `promote_to_spo` bridge is the natural next step and is unblocked by this
ruling.

## What this decision does NOT decide

- Whether ARiGraph's three layers (semantic / episodic / cognitive) deserve their
  own dedicated TruthValue algebras. They share `contract::crystal::TruthValue`
  today; that may not be optimal long-term, but the ontology crate has no opinion.
- Whether `SchemaExpander` should grow a target-store discriminator. Not necessary
  for this session — consumers pick their store explicitly when they call the
  expansion path.
- Whether the planner IR should annotate steps with their target cache layer.
  That is planner work, not ontology work.

## Citations

- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md:70` — SPO-1 row.
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md:245` — SPO-1 disposition.
- `crates/lance-graph/src/graph/spo/ontology_bridge.rs` — existing
  `SchemaExpander` integration.
- `crates/lance-graph-contract/src/ontology.rs:1-120` — `Ontology`/`Schema`/
  `OntologyBuilder` definition.
