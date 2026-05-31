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

---

## 3. Hard prerequisites — the gates (state these before any D-id ships behavior)

Three falsifier probes and one statistical floor gate this whole arc. They are
not optional decoration; they are **kill-switches**. A D-id may be *built*
(types compile, fixtures pass) without its gate, but it MUST NOT graduate from
fixture to behavior-on-real-data until its gate is green.

### Gate P1 — Partition locality (CONJECTURE → must measure)
- **Driver:** `jc/examples/splat_louvain_modularity.rs` (Louvain modularity =
  popcount-AND over `contract::splat::AwarenessPlane16K` planes) +
  `neighborhood::clam::measure_cluster_radii` on the real P279/subClassOf +
  edge graph derived from `data/ontologies/*.ttl` (e.g. the FIBO or
  schema.org subtree; biology subtree once Wikidata lands).
- **Pass:** high modularity ⇒ ≥~90% of edges are intra-cohort ⇒ 16-bit
  intra-cohort references + the family frontier are real, and the natural
  fan-out (the 4/12/16 split) is observed, not assumed.
- **Gates:** D-LWS-1 fan-out choice; D-LWS-4 GOP P-frame placement; D-LWS-5
  cohort residency.
- **Honest status:** `clam.rs` header literally says the radii-coincide-with-
  ontology-boundaries claim "is a TEST, not a fact." Treat as **CONJECTURE**.

### Gate P2 — Delta-card truthfulness (CONJECTURE → must measure)
- **Driver:** D-LWS-8 reconstructs content from N delta bits
  (`FieldMask`/value delta over the inherited `WikidataClass` archetype) vs
  ground truth; histograms the residual per cohort.
- **Pass:** low residual ⇒ the cohort is real and 8–16 delta bits suffice;
  high residual ⇒ wrong cohort or genuinely novel entity (needs a wider delta
  or a fork — never a 2-bit axis).
- **Gates:** D-LWS-2 (the value model only ships its bit-width claim once the
  residual histogram backs it).

### Gate P3 — Compose vs materialize (CONJECTURE → must measure)
- **Driver:** D-LWS-8 measures the ≤7-hop reachability hit-rate +
  compose-cache eviction churn (via `ComposeTable::compose_chain` / blasgraph
  `mxm`) against a stored-edge baseline.
- **Pass:** the N²-avoidance holds (closure is ≤7 cached hops, not a stored
  edge), and the churn sets the GOP/compaction cadence.
- **Gates:** D-LWS-3 (compose-cache); D-LWS-4 (GOP cadence); D-LWS-6 (prefetch
  cascade Compose arm).

### Gate D-ARM-7 — the Jirak floor (HARD PREREQUISITE for any live write)
- **Status (grepped):** `STATUS_BOARD.md` D-ARM-7 row = **"Queued — HARD
  PREREQUISITE"**; ISSUE `ARM-JIRAK-FLOOR` = **OPEN**. The engine
  `jc::jirak::prove` exists (Jirak-Cartan Pillar 5, weak-dependence
  Berry-Esseen rate `n^(p/2-1)`); the *gate function* (rule → significant?)
  that derives a threshold from it does NOT yet exist.
- **Rule:** **No hydrated rule, discovered edge, or proposed reclassification
  may be written to a live store (`SpoStore`, `VersionedGraph`, or any P-frame
  delta that persists) until D-ARM-7 lands and the candidate passes the Jirak
  weak-dependence significance floor BEFORE the classical `min_support`/
  `min_confidence` gate.** This binds D-LWS-5 (any persist), D-LWS-3 (any
  derived edge promoted to a generator), and D-LWS-9 (the full load). Cites
  `I-NOISE-FLOOR-JIRAK`.
- **Read-only is exempt:** hydrating cold rows into the hot SoA for *reading*
  is not a write and is not gated by D-ARM-7. Only mutation of the persistent
  substrate is.

---

## D-LWS-1 — Sparse radix range-delegation register

**Status: Queued. Label: NEW (composes shipped `NiblePath`).**

### Scope
A **path-compressed radix/Patricia trie over the frozen ontology**, holding
**occupied branch points only** — the "range register" of the integration map
§3. Each entry is `nibble-range → {Empty | Leaf(file_or_arena) | Delegate(sub-table)}`:
- a sparse DOLCE branch collapses to one `Leaf`;
- a dense branch (the future 40M scholarly-articles cohort) becomes a
  `Delegate` → sub-table → many leaves;
- single-child chains collapse (no branch = no information = nothing stored).

The register's size ≈ the occupied branch count (≈ the OWL/DOLCE class count,
KB–MB), **never** the 256⁴ = 4.3B virtual address space.

**It reuses `NiblePath` as the address — it does NOT invent a new key.** A
register lookup walks `NiblePath` nibble by nibble (`child`/`try_child`),
matching compressed ranges; `is_ancestor_of` decides delegation containment;
`basin()` extracts the DOLCE root nibble; `packed()` yields the `(u64, u8)` the
directory stores.

### The shipped primitive it builds on
- `contract::hhtl::NiblePath` — the entire address algebra (`root`, `child`,
  `try_child`, `basin`, `parent`, `depth`, `is_ancestor_of`, `packed`,
  `FAN_OUT=16`, `MAX_DEPTH=16`). **The register stores ranges of NiblePaths;
  it never re-encodes identity.**
- `ontology::wikidata_hhtl::WikidataClass::nibble_path()` — the seed: every
  curated class already emits its `NiblePath` from `dolce_id` + subclass path.
  D-LWS-1's register is the inverse index over exactly these paths.
- `ontology::ttl_parse::{parse_ttl_directory, parse_into_proposals}` — the
  occupied branch points for the *first* register are the classes parsed from
  `data/ontologies/*.ttl` (FIBO/DUL/schema.org/QUDT), NOT a Wikidata dump.

### Firewall / honesty
- Lives in the **hub** (`lance-graph-contract` for the type if zero-dep clean,
  else `lance-graph-ontology`). Proposed home: `contract::hhtl` sibling module
  `contract::radix_register` (zero-dep: it is pure `NiblePath` + ranges + a
  `Vec`-backed trie; no Lance, no Arrow). **Verify zero-dep before placing in
  contract;** if it needs ontology types, place in `lance-graph-ontology`.
- `aerial` is NOT touched. The register is an addressing structure the hub
  owns; the proposer never sees it.
- **Honest substrate:** built and tested on the on-disk TTL classes + the 6
  `curated_wikidata_classes()` fixtures. The 38× headroom / 2.6%-full /
  4.3B-virtual numbers are **DESIGN TARGETS**, asserted on fixtures, not
  measured on 115M (that is D-LWS-9).

### Which probe / gate
- **Gate P1** sizes the fan-out: the register's branching factor (4/12/16
  split, or the frozen 16-way `NiblePath` default) is a frozen-ISA choice that
  P1's Louvain/CLAM measurement must back before it is frozen append-only. Until
  P1 is green, D-LWS-1 ships the **16-way `NiblePath`-native** register (the
  conservative, already-frozen choice) and leaves the re-parameterization
  (256⁴ byte-aligned) as a documented CONJECTURE.

### Acceptance (fixture-level)
- Round-trip: every `curated_wikidata_classes()` path inserts, looks up, and
  the register reconstructs the exact `NiblePath` (CAM-exact, no similarity).
- Path compression: a single-child chain (person → human) stores ONE branch
  point, not two (assert occupied-branch count < path count).
- Delegation: a synthetic dense cohort (≥2 leaves under one nibble range)
  produces a `Delegate`, a sparse one a `Leaf`.
- Empty-space proof: the 97% unoccupied virtual space materializes zero
  entries (assert register size ≈ occupied count, not fan-out^depth).
