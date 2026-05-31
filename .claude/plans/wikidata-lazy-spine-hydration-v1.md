<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright The Lance Authors
-->

# IMPLEMENTATION PLAN: Wikidata lazy-spine hydration v1 — the NiblePath-keyed tiered hydration manager + its addressing layer

> **Status: QUEUED (all D-ids).** This is the implementation plan for the ONE
> missing runtime piece named in `delta-card-addressing-integration-map.md` and
> `agnostic-lazy-world-spine.md`: the **`NiblePath`-keyed tiered hydration
> manager**, plus the **sparse radix range-delegation register** it rides on,
> the **I/P/B frame model over Lance versioning**, the **RISC compose-cache**,
> and the **delta-card value model**. Every load-bearing primitive it composes is
> SHIPPED and grepped (see § Verified primitives); the manager itself is NEW.
>
> **Authored by:** W1 (autoattended wave). **Companions (the design this plans):**
> `.claude/knowledge/delta-card-addressing-integration-map.md` (THE design),
> `.claude/knowledge/agnostic-lazy-world-spine.md` (tiered-substrate framing).

---

## 0. What this plan is, and is NOT

**IS:** the runtime layer that turns the frozen Wikidata-HHTL skeleton
(`ontology::wikidata_hhtl::WikidataClass`, curated today) + the on-disk
ontologies (`data/ontologies/*.ttl`) into a **foveated, tiered, address-unified
substrate** — a tiny resident skeleton with on-demand hydration of cold detail
keyed by `contract::hhtl::NiblePath`, with eviction driven by the DOLCE
continuant/occurrent bit.

**IS NOT:**
- A Wikidata loader for the full 115M-entity dump. **There is no dump on disk**
  (grepped: only `ontology::wikidata_hhtl` curated fixtures + the
  `wikidata_landing` test). The full load is a deferred terminal D-id
  (D-LWS-9), explicitly gated behind the probes. Every earlier D-id is
  validatable on the **real on-disk ontologies** (`data/ontologies/*.ttl`:
  `dul.ttl`, `fibo-*`, `schemaorg.ttl`, `qudt-*.ttl`, `provo.ttl`, `time.ttl`,
  `odoo/odoo-core.ttl`, `skos`, `zugferd`) + the 6 curated `WikidataClass`
  fixtures.
- A change to the `aerial` proposer's dependency surface. **The firewall holds:**
  `aerial` (`lance-graph-arm-discovery`) stays the zero-dep proposer; the hub
  (`lance-graph` + `lance-graph-ontology` + `lance-graph-contract`) owns
  contract/ontology and the hydration manager. This plan adds NOTHING heavy to
  `aerial`.
- A rebalancer. The frozen-ISA addressing (`NiblePath`, append-only offsets)
  deletes the rebalancer by construction; this plan does not reintroduce one.
- A replacement for `VersionedGraph` / Lance versioning. The I/P/B frame model
  RIDES the existing versioning surface; it does not fork it.

---

## 1. Verified primitives (every symbol grepped on this branch before citing)

| Symbol | Path (grepped) | Role in this plan | Label |
|---|---|---|---|
| `contract::hhtl::NiblePath` | `lance-graph-contract/src/hhtl.rs:56` (`root`/`child`/`basin`/`parent`/`depth`/`is_ancestor_of`/`packed`/`leaf`/`try_child`/`EMPTY`/`FAN_OUT=16`/`MAX_DEPTH=16`) | THE address key for every tier | **built** (#442) |
| `contract::class_view::FieldMask` | `class_view.rs:69`; `inherit(delta)` @ 136, `from_positions`/`with`/`has`/`count`, `MAX_FIELDS=64` | delta-over-archetype presence mask (the KEY-side delta-card) | **built** (#441) |
| `contract::class_view::ClassView` | `class_view.rs` (trait); `ClassId = u16` @ 53; `StructuralSignature` | the deck (resolve-not-store schema) | **built** (#441) |
| `causal-edge::CausalEdge64` | `crates/causal-edge/src/edge.rs` | the resident-row edge + W-slot Markov pointer | **built** |
| `contract::witness_table::WitnessTable` | `witness_table.rs:96` (`WitnessTable<const N=64>`, `WitnessEntry` @ 65, `get`/`set`) | the per-cohort W-slot arc (6-bit cohort) | **built** |
| `contract::soa_view::MailboxSoaView` | `soa_view.rs:28` (trait) + `MailboxSoaOwner` @ 90 | the hot resident carrier (read-only `&[T]` borrow) | **built** (#437) |
| `bgz-tensor::attention::ComposeTable` | `attention.rs:49`; `compose(a,b)` @ 206, `compose_chain(a,b,c)` @ 215, `build` | per-hop u8 compose for the RISC closure | **built** |
| `bgz-tensor::hhtl_cache::RouteAction` | `hhtl_cache.rs:37` (`Skip`/`Attend`/`Compose`/`Escalate`); `HhtlCache::route(a,b)` @ 200; `HipCache` alias @ 510 | the foveated-prefetch decision cascade | **built** |
| `lance-graph::graph::neighborhood::clam` | `clam.rs`; `measure_cluster_radii` @ 74, `analyze_pareto_convergence`, `ParetoAnalysis`, `RadiusObservation` | the CLAM **radius probe** (NOT a clusterer — see note) | **built (probe only)** |
| `lance-graph::graph::versioned::VersionedGraph` | `versioned.rs:98`; `at_version(n)`, `version()`, `GraphDiff` @ 70, Merkle seals | the Lance versioning surface the I/P/B frames ride | **built** |
| `ontology::wikidata_hhtl::WikidataClass` | `wikidata_hhtl.rs:47`; `nibble_path()`/`presence_mask()`/`signature()`/`dcls_triple()`; `curated_wikidata_classes()` @ 144; `WikidataClassView` @ 215 | the frozen skeleton fixtures (keyframe seed) | **built** |
| `ontology::ttl_parse` | `ttl_parse.rs`; `TtlSource::from_path` @ 74, `parse_ttl_directory` @ 379, `parse_into_proposals` @ 106 | the real on-disk TTL loader (validation substrate) | **built** |
| `ontology::class_resolver::dolce_id` | `class_resolver.rs:45` (`ENDURANT=0`/`PERDURANT=1`/`QUALITY=2`/`ABSTRACT=3`) | the DOLCE basin + the derived 1-bit eviction key | **built** |
| `contract::splat::{SplatChannel, AwarenessPlane16K}` | `splat.rs:32`/`splat.rs:88`; `Counterfactual=3` | discovery-layer carrier (offline only; never on the hot path) | **built** |
| `jc::jirak::prove` | `crates/jc/src/jirak.rs:124`; `pub mod jirak` @ `lib.rs:35` | the Jirak weak-dependence Berry-Esseen proof (the D-ARM-7 engine) | **built (proof)** |
| `jc/examples/splat_louvain_modularity.rs` | grepped; imports `contract::splat::AwarenessPlane16K`; "Louvain modularity gain reduces to popcount-AND" | probe-1 driver (partition locality) | **built (example)** |

**RISK — symbols I wanted to cite but could NOT verify by grep (flagged, not cited as shipped):**
- **`EpisodicWitness64`** — cited in BOTH companion docs as a shipped type. **Zero
  hits in `crates/`.** The actual shipped surface is
  `WitnessTable<N=64>` + `WitnessEntry` (the `witness_table.rs` doc *describes*
  the Markov arc "through episodic-reference vectors" but ships no
  `EpisodicWitness64` type). This plan cites only `WitnessTable`/`WitnessEntry`
  and treats `EpisodicWitness64` as a **doc-level alias / CONJECTURE**, never as
  a shipped API.
- **Lance *fragment*-versioning** (fragment-level `compact`/`add_columns`) — the
  integration map names "Lance fragment-versioning" as the I/P/B substrate.
  Grep shows the repo wires **dataset-level** versioning (`VersionedGraph`,
  `at_version`, `version()`), NOT Lance fragment APIs (no `FragmentMetadata` /
  `add_columns` / `compact` usage in `crates/lance-graph/src/`). Lance *the
  dependency* supports fragments; this repo does not yet wire them. So the
  I/P/B-over-fragments mapping is labelled **NEW (must wire Lance fragment APIs)
  / CONJECTURE** below, riding `VersionedGraph` as the shipped seam.

> **Note on CLAM:** `neighborhood::clam` is a **measurement/probe** module
> (`measure_cluster_radii`, `analyze_pareto_convergence`) whose own header says
> *"This is a TEST, not a fact."* It does NOT ship a clustering engine that
> *produces* a P-frame placement. So every "CLAM-clustered delta" claim below is
> built on the **probe** (measure radii → decide placement offline), not on a
> shipped clusterer. The clusterer that consumes the radii is NEW.

---

## 2. The D-id index (all Queued)

| D-id | Title | Builds on (shipped) | Gated by |
|---|---|---|---|
| **D-LWS-1** | Sparse radix range-delegation register | `NiblePath`, `WikidataClass::nibble_path`, `ttl_parse` | Probe 1 (locality) sizes fan-out |
| **D-LWS-2** | Delta-card value model (`deck ⊗ delta`) | `FieldMask::inherit`, `ClassView`, `WikidataClass::presence_mask` | Probe 2 (residual) |
| **D-LWS-3** | RISC compose-cache + per-predicate composability flag | `ComposeTable::{compose,compose_chain}`, blasgraph `mxm` | Probe 3 (compose hit-rate) |
| **D-LWS-4** | I/P/B frame model over Lance versioning | `VersionedGraph`, `clam::measure_cluster_radii` | Probe 1 + Probe 3 (GOP cadence) |
| **D-LWS-5** | The `NiblePath`-keyed tiered hydration manager | D-LWS-1..4 + `MailboxSoaView`, `RouteAction`, `dolce_id`, `WitnessTable` | all 3 probes; **D-ARM-7** before any write |
| **D-LWS-6** | Foveated prefetch cascade (RouteAction-driven) | `HhtlCache::route`, `ComposeTable` | Probe 3 |
| **D-LWS-7** | Eviction policy on the DOLCE continuant/occurrent 1-bit | `dolce_id`, D-LWS-5 | — |
| **D-LWS-8** | Probe harness (the 3 falsifiers, on real TTL + fixtures) | `splat_louvain_modularity`, `clam`, `FieldMask` | — (this PRODUCES the gates) |
| **D-LWS-9** | DEFERRED: full Wikidata load (115M) into the spine | all above, all probes PASSED, D-ARM-7 landed | every probe + D-ARM-7 |

**Sequencing DAG:**
```
            D-LWS-8 (probes) ──────────────────────────────┐ (gates everything)
                  │                                          │
   D-LWS-1 ───────┼──► D-LWS-4 ──┐                           │
   D-LWS-2 ───────┤              ├──► D-LWS-5 ──► D-LWS-6     │
   D-LWS-3 ───────┘              │        │                  │
                                 │        └──► D-LWS-7        │
                                 └─────────────────► D-LWS-9 (deferred, all gates)
                                            ▲
                                   D-ARM-7 (Jirak floor) ─── hard prereq for any WRITE
```
