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

---

## D-LWS-2 — Delta-card value model (`reconstruct = deck ⊗ delta`)

**Status: Queued. Label: NEW (composes shipped `FieldMask::inherit` + `ClassView`).**

### Scope
The VALUE side of the one idea: **a card stores the surprise; the deck stores
the expectation.** An entity's stored content is a **small delta over the
inherited frozen archetype** (its class deck). Reconstruct = `deck ⊗ delta`.
This D-id ships:
1. A `DeltaCard` type = `{ class_path: NiblePath, presence_delta: FieldMask,
   value_bits: <small> }` — the per-entity surprise, nothing else. The modal
   member of a class is the **empty card** (`FieldMask::EMPTY` delta, zero value
   bits): it *is* its archetype, stores nothing.
2. A `reconstruct(deck: &ClassView, card: &DeltaCard) -> ResolvedEntity` that
   overlays the card onto the deck.

### The shipped primitive it builds on
- `contract::class_view::FieldMask::inherit(delta)` (verified @ `class_view.rs:136`)
  — **this IS the `deck ⊗ delta` operator for the presence half.** The archetype's
  mask `inherit`s the card's delta mask. The KEY-side (#442 `wikidata_landing`
  already proved "human ⊂ person inherits path + mask-as-delta"); D-LWS-2
  generalizes it to the VALUE side.
- `contract::class_view::ClassView` (trait) + `ClassId = u16` +
  `StructuralSignature` — the deck. Resolve-not-store: the deck holds
  fields/labels/DOLCE; the card holds neither (zero schema bits, zero label bits).
- `ontology::wikidata_hhtl::WikidataClass::{presence_mask, signature, dcls_triple}`
  — the fixture decks. `dcls_triple()` already returns the
  `(ClassId, StructuralSignature, FieldMask)` triple a card resolves against.

### The honest boundary (carry this verbatim from the integration map)
The delta carries the **compressible profile** (the inherited-archetype
deviation), NOT irreducible specifics. Non-composable, irreducible facts
(`birth_date`, `population`, a novel signature step) are **stored values**, never
a 2-bit axis — this is the **generator-vs-derivable split** (shared with
D-LWS-3). A fusion entity outside any cohort = a wider delta or a fork.

### Firewall / honesty
- Lives in the hub. The `DeltaCard` type is a candidate for
  `lance-graph-contract` (zero-dep: `NiblePath` + `FieldMask` + a small value
  payload). **Verify zero-dep;** the `reconstruct` against a live `ClassView`
  may belong in `lance-graph-ontology`.
- `aerial` untouched. (`aerial` *proposes* which cohort a row joins via splat,
  offline — D-LWS-2 only *reconstructs* given a chosen deck. The proposer's
  similarity never enters the value model.)

### Which probe / gate
- **Gate P2 (delta-card truthfulness)** is THIS D-id's falsifier. The bit-width
  claim ("8–16 delta bits suffice") ships only once D-LWS-8's per-cohort
  residual histogram is low on the real fixtures. Until then D-LWS-2 ships the
  *mechanism* (`reconstruct`) with the bit-width left as a measured parameter,
  NOT a hardcoded constant.
- **Free-energy framing (CLAUDE.md The Click):** the card's bit-width IS the
  residual surprise `F = (1−likelihood) + kl`; the archetype is the prior, the
  delta is the prediction error. Stated as design rationale, not a code claim.

### Acceptance (fixture-level)
- The modal member of each `curated_wikidata_classes()` cohort reconstructs from
  an EMPTY card (zero delta bits) — "absence IS the inheritance."
- A surprising member (e.g. a class with an extra presence bit vs its parent)
  reconstructs from a card carrying exactly that one `FieldMask` bit, verified
  via `FieldMask::inherit`.
- Round-trip exactness: `reconstruct(deck, encode(entity)) == entity` for the
  presence half (CAM-exact; the value-bit half is exact up to the P2-measured
  width).

---

## D-LWS-3 — RISC compose-cache + per-predicate composability flag

**Status: Queued. Label: NEW (composes shipped `ComposeTable` + blasgraph `mxm`).**

### Scope
**Store the generators, compute the closure.** Storing "every entity related to
every other" = 113M² ≈ 10¹⁶ edges (catastrophe). Instead store
parent/child/spouse generators (~N) and **derive** "related to Y in ≤7 hops" on
demand. This D-id ships:
1. A **per-predicate composability flag** (~12k Wikidata predicates, but
   seeded on the on-disk ontology predicates first): each predicate is
   `Generator(store)` or `Derivable(compose)`. Non-composable facts
   (`birth_date`, `population`) are `Generator` always (irreducible values).
2. A **compose-cache**: derived multi-hop edges computed via
   `ComposeTable::compose_chain` (each hop = a u8 table lookup) / blasgraph
   `mxm` matrix-power, cached as evictable B-frame entries (≤7 hops).

### The shipped primitive it builds on
- `bgz-tensor::attention::ComposeTable` (verified @ `attention.rs:49`):
  `compose(a, b) -> u8` (one hop), `compose_chain(a, b, c) -> u8` (two hops),
  `build(palette)`. **The closure is a fold of `compose` over the path — the
  N²-avoidance is literally this table.**
- blasgraph `mxm` (matrix-power semiring multiply in
  `lance-graph/src/graph/blasgraph/`) — the bulk alternative for dense
  reachability fronts.
- The DOLCE 1-bit (`class_resolver::dolce_id`): **generators = `continuant` =
  permanent/cold**; **composed multi-hop paths = `occurrent` = temporary/
  evictable** (shared eviction policy with D-LWS-7).

### The hub problem dissolves
*United States*, *human*, *Earth* never store their millions of inbound
back-edges — they are **reached** by composing forward generators. Hubs were
only a problem if you imagined materializing them. Stated as design rationale.

### Firewall / honesty
- `ComposeTable` lives in `bgz-tensor` (standalone, excluded crate, zero-dep).
  The compose-cache + composability flag live in the hub
  (`lance-graph-ontology` for the predicate flag table; `lance-graph` for the
  blasgraph `mxm` driver). `aerial` untouched.
- **Honest substrate:** the predicate flag table is seeded and tested on the
  on-disk ontology predicates (FIBO/schema.org/QUDT relations), NOT the 12k
  Wikidata predicates. The 12k figure is a DESIGN TARGET for D-LWS-9.

### Which probe / gate
- **Gate P3 (compose vs materialize)** is THIS D-id's falsifier: the ≤7-hop
  hit-rate + eviction churn vs a stored-edge baseline. If the hit-rate is low
  (closure NOT reachable in ≤7 hops) the composability flags are wrong, or the
  generator set is too sparse — D-LWS-3 does not graduate from fixture to
  behavior until P3 is green.
- **D-ARM-7:** if a *derived* edge is ever promoted to a stored generator (a
  reclassification of the composability flag), that write passes the Jirak floor
  first. Read-time composition is exempt.

### Acceptance (fixture-level)
- A 3-hop derivable relation over the fixture graph reconstructs via
  `compose_chain` and equals the stored-edge ground truth.
- A `Generator` predicate (e.g. a fixture `birth_date`) is never composed — the
  flag forces a stored lookup.
- Eviction: a composed B-frame entry under an `occurrent` predicate evicts; a
  `continuant` generator does not.

---

## D-LWS-4 — I/P/B frame model over Lance versioning

**Status: Queued. Label: NEW (rides shipped `VersionedGraph`; the fragment-level GOP is CONJECTURE — see RISK).**

### Scope
The cold floor IS a keyframe/delta store (the x264/265 capstone). Map:

| video | spine | shipped seam |
|---|---|---|
| **I-frame** | frozen radix trie (D-LWS-1) + compacted base (self-decodable, exact, rare) | `VersionedGraph` base version + D-LWS-1 register |
| **P-frame** | appended entities + CLAM-clustered new arrivals + corrections (cheap, references the keyframe) | a new `VersionedGraph` version (append-only write) |
| **B-frame** | the RISC compose-cache (D-LWS-3) — multi-hop derived, references multiple bases, evictable | in-memory compose-cache, never persisted |
| **GOP** | keyframe + accumulated deltas, periodically re-baselined by **compaction** | a deliberate version-gated re-emit |

This D-id ships the **frame-classification + overlay-resolve logic**: given a
`NiblePath` + a version, resolve = base I-frame + N P-frame deltas overlaid
(the LSM/video-seek read-amplification, bounded by GOP length).

### The shipped primitive it builds on
- `lance-graph::graph::versioned::VersionedGraph` (verified @ `versioned.rs:98`):
  `at_version(n)` (time-travel = seek to a frame), `version()` (current frame
  number), `GraphDiff {from_version, to_version}` (the P-frame delta between two
  versions), Merkle seals (`graph_seal_check` — the keyframe integrity check).
  **Each write already creates a new Lance version → that IS a P-frame append.**
- `neighborhood::clam::measure_cluster_radii` — CLAM is adaptive *inside the
  delta*: it **proposes** placement of new arrivals as a P-frame (offline,
  similarity), the keyframe never moves. (Probe only — the clusterer that acts
  on the radii is NEW; see §note on CLAM.)
- D-LWS-1 (the frozen radix) is the I-frame's address half; D-LWS-3 (compose-
  cache) is the B-frame.

### The frozen-vs-adaptive tension resolves here
CLAM is adaptive inside the delta (proposes); the keyframe never moves;
**compaction = re-emit a fresh keyframe = the amortized schema upgrade** (the
one deliberate version-gated moment, carrying the ontology-version byte per
`I-LEGACY-API-FEATURE-GATED`). Adaptive proposes (in the delta); frozen ships
(the keyframe). Deltas are *exact* (a P-frame is lossless); CLAM similarity
*decides* the delta, is never stored *as* the address (the two-trees iron-rule
guard: addressing = exact CAM; similarity = discovery-only,
`faiss-homology`/`I-VSA-IDENTITIES`).

### Firewall / honesty / RISK
- **RISK (carried from §1):** the integration map says "Lance fragment-
  versioning." Grep shows this repo wires **dataset-level** `VersionedGraph`,
  NOT Lance **fragment** APIs (`add_columns`/`compact`/`FragmentMetadata` —
  zero usage in `crates/lance-graph/src/`). Two honest options, both
  documented, decision deferred to the integration-lead:
  - **(a) Ride dataset versioning (built seam, ships now):** I=base version,
    P=append version, GOP-compaction = re-emit a baseline dataset. Coarser
    granularity (whole-dataset, not fragment).
  - **(b) Wire Lance fragment APIs (NEW, finer GOP):** use Lance's native
    `Fragment` + `compact` so a P-frame is a fragment append and GOP-compaction
    is fragment compaction (the integration map's literal intent). This is a
    NEW Lance-binding task, not a shipped seam — labelled CONJECTURE until a
    spike proves the Lance version on `Cargo.lock` (`lance =6.0.0`) exposes the
    needed fragment surface to this crate.
- `aerial` untouched. The frame model is a hub-side cold-floor concern.

### Which probe / gate
- **Gate P1** backs P-frame placement (CLAM radii must coincide with cohort
  boundaries for "CLAM clusters new arrivals" to be real).
- **Gate P3** sets the **GOP/compaction cadence** (eviction churn → how often to
  re-baseline).
- **D-ARM-7:** a P-frame that persists a *discovered* rule/edge passes the Jirak
  floor first (a P-frame append is a live write).

### Acceptance (fixture-level)
- Resolve-by-overlay: an entity whose value lives in a P-frame delta resolves to
  `base ⊗ delta` and equals ground truth (riding `at_version` / `GraphDiff` on a
  fixture `VersionedGraph`).
- Read-amplification bound: resolving across K P-frames touches exactly K+1
  versions (assert the seek cost = GOP length).
- Compaction: re-emitting a keyframe collapses K P-frames into one base; a
  subsequent resolve touches 1 version (assert amplification reset).

---

## D-LWS-5 — The `NiblePath`-keyed tiered hydration manager (THE missing runtime piece)

**Status: Queued. Label: NEW (the synthesis — composes ALL of D-LWS-1..4 + shipped `MailboxSoaView`/`RouteAction`/`dolce_id`/`WitnessTable`).**

### Scope
The one missing runtime piece named in both companion docs. It is the
**hot mailbox-SoA ↔ cold Lance** manager, keyed by `NiblePath`:
- **lazy-load** a basin's cold rows on first touch (cold `VersionedGraph` read →
  hot `MailboxSoaView` SoA), addressed by `NiblePath`, **NOT** by DataFusion
  join (address, not join — the cold path splits in two; the join serves only
  business-SQL ground truth, off the HHTL hot path);
- **foveated adjacency prefetch** via the `RouteAction` cascade (D-LWS-6);
- **evict** cold/occurrent arenas on the DOLCE 1-bit (D-LWS-7).

It is a **manager/coordinator**, not a store: it owns the residency decision
(what is hot), delegates addressing to D-LWS-1, value reconstruction to
D-LWS-2/D-LWS-4, and adjacency to D-LWS-3/D-LWS-6.

### The shipped primitives it builds on
- `contract::hhtl::NiblePath` — the single allocation key. One O(1) address =
  ontology position = memory arena = spatial coord.
- `contract::soa_view::MailboxSoaView` / `MailboxSoaOwner` (verified
  `soa_view.rs:28/90`) — the hot resident carrier. The manager hydrates INTO a
  `MailboxSoaOwner` and hands out read-only `&[T]` views (E-SOA-VIEW-IS-A-BORROW;
  never copies, never caches a label — the SoA stays agnostic forever, core
  inv #1 / C2).
- `lance-graph::graph::versioned::VersionedGraph` — the cold floor read
  (`at_version`), via D-LWS-4's overlay-resolve.
- `contract::witness_table::WitnessTable<64>` + `WitnessEntry` — the per-cohort
  (6-bit) Markov W-slot arc the resident row carries; traversal walks W-refs
  backward without dereferencing the full SPO store per hop. **(NOT a 16384-bit
  VSA bundle — that is retired legacy, survives only as the discovery carrier.)**
- `causal-edge::CausalEdge64` — the resident-row planner edge whose W-slot
  points into the `WitnessTable`.
- `ontology::class_resolver::dolce_id` — the residence policy key (D-LWS-7).

### The bounded-hot / unbounded-cold invariant
- Wikidata is **32-bit-addressed (cold), never resident**; the hot envelope is
  the documented **64K–256K** concurrent mailbox window
  (`MailboxSoaView`/`witness_table.rs` envelope). You **foveate** the spine:
  256K holds whole corpora + a hydrated Wikidata slice at once. The manager's
  job is to keep the foveal region hot and let the periphery stay cold.
- The widths nest: 6-bit cohort ⊂ 16-bit book ⊂ 18-bit hot envelope (256K) ⊂
  32-bit world. (The 16-bit book tier is CONJECTURE — see RISK on
  `EpisodicWitness64`; the 6-bit cohort `WitnessTable` and 32-bit `mailbox_ref`
  ends are in code.)

### Firewall / honesty
- The manager lives in the **hub** — proposed home `lance-graph` (it needs
  `VersionedGraph` + blasgraph, which are hub-only) with the residency policy
  types in `lance-graph-contract` if zero-dep. **`aerial` is NOT a dependency
  and is NOT depended upon by this manager.** The proposer feeds *discovery*
  (what lands where, offline); the manager does *runtime residency*. Distinct
  layers.
- **Honest substrate:** D-LWS-5 is built and tested hydrating the 6
  `curated_wikidata_classes()` fixtures + the on-disk TTL classes from a fixture
  `VersionedGraph`. **No 115M load** — that is D-LWS-9, gated on all probes +
  D-ARM-7.

### Which probe / gate
- **All three probes** gate the manager's behavior-on-real-data:
  P1 (cohort residency is local), P2 (hydrated cards reconstruct truthfully),
  P3 (adjacency composes, doesn't materialize).
- **D-ARM-7 is a HARD PREREQUISITE for any WRITE the manager performs** (any
  P-frame persist, any reclassification, any hydrated rule). Read-only
  hydration is exempt. The manager MUST refuse to persist a discovered artifact
  until D-ARM-7's gate function is wired and passed.

### Acceptance (fixture-level)
- First-touch hydration: addressing a cold `NiblePath` loads exactly that
  basin's rows into the `MailboxSoaOwner`, and a second touch is a hot hit (no
  re-read).
- Address-not-join: the hydration path issues a `VersionedGraph` columnar read
  keyed by `NiblePath`, NOT a DataFusion join (assert no join on the hot path).
- Agnostic SoA: the hot view exposes only structure + address + the
  `CausalEdge64`/`WitnessTable` arc; NO label is ever stored hot (assert the
  SoA carries no string).
- Bounded envelope: hydrating > the 256K envelope triggers eviction (D-LWS-7),
  never unbounded growth.
- **Write-refusal:** attempting to persist a discovered rule without a passed
  Jirak gate returns an error (the D-ARM-7 prerequisite is enforced in code, not
  just documented).

---

## D-LWS-6 — Foveated prefetch cascade (RouteAction-driven)

**Status: Queued. Label: NEW (composes shipped `HhtlCache::route` + `ComposeTable`).**

### Scope
Like Google-Maps tile prefetch: the adjacent area streams into the hot context
before reasoning pans to it. When the foveal `NiblePath` is hydrated (D-LWS-5),
the manager prefetches adjacency via the **`RouteAction` cascade**: for each
candidate neighbor archetype pair `(a, b)`, the route decides
`Skip | Attend | Compose | Escalate`. Only `Attend`/`Compose` neighbors are
prefetched; `Skip` (the ~60% majority) costs nothing.

### The shipped primitive it builds on
- `bgz-tensor::hhtl_cache::RouteAction` (verified @ `hhtl_cache.rs:37`) +
  `HhtlCache::route(a, b) -> RouteAction` (verified @ `:200`; `HipCache` alias
  @ `:510`). The doc literally calls `route` "the prefetch decision." The
  cascade's documented distribution (Skip ~60% / Attend ~35% / Compose rare /
  Escalate ~5%) is exactly the foveated-periphery economics.
- `bgz-tensor::attention::ComposeTable` — the `Compose` arm resolves a
  multi-hop neighbor via `compose_chain` (shared with D-LWS-3).

### Firewall / honesty
- `RouteAction`/`HhtlCache`/`ComposeTable` all live in `bgz-tensor` (standalone,
  zero-dep). The prefetch driver lives in the hub (D-LWS-5's manager). `aerial`
  untouched.
- **Honest substrate:** the cascade is tested on the fixture adjacency derived
  from on-disk TTL relations. The Skip/Attend percentages are bgz-tensor's
  documented design figures, asserted as a sanity range on fixtures, not
  measured on 115M.

### Which probe / gate
- **Gate P3:** the prefetch hit-rate (did the prefetched periphery get used?) is
  part of the compose-vs-materialize measurement; low hit-rate ⇒ the cascade is
  over-fetching (re-tune the route thresholds).

### Acceptance (fixture-level)
- A `Skip` pair is never hydrated; an `Attend` pair is; a `Compose` pair
  resolves via `compose_chain` without a stored edge.
- Prefetch is bounded by the 256K envelope (prefetch yields to eviction).

---

## D-LWS-7 — Eviction on the DOLCE continuant/occurrent 1-bit

**Status: Queued. Label: NEW (composes shipped `dolce_id`).**

### Scope
The ontology's own top split IS the cache policy. **DOLCE = a 1-bit
permanent/temporary residence policy:**
- **continuant** (Endurant / Quality / Abstract — wholly present, persists) ⇒
  **permanent / cold-persist / resident-priority**;
- **occurrent** (Perdurant — temporal parts, happens-then-ends) ⇒
  **ephemeral / evictable** (the Baton/event traffic; the B-frame compose-cache).

One eviction policy, derived from the ontology. The manager evicts occurrent
arenas first under envelope pressure; continuant generators are sticky.

### The shipped primitive it builds on
- `ontology::class_resolver::dolce_id` (verified @ `class_resolver.rs:45`):
  `ENDURANT=0`, `PERDURANT=1`, `QUALITY=2`, `ABSTRACT=3`. **The derived 1-bit =
  `dolce_id == PERDURANT` ⇒ occurrent ⇒ evictable; else continuant ⇒
  permanent.** The 4-facet `dolce_id 0..3` stays cache-resolvable (do NOT drop
  the axis — the residence bit is *derived*, per the invariant guard); eviction
  keys on the derived bit.
- `WikidataClass::dolce_id` field — every fixture class already carries it.

### Firewall / honesty
- Lives in the hub (D-LWS-5's manager + the `dolce_id` resolver). `aerial`
  untouched.
- **Invariant guard (verbatim):** keep `dolce_id 0..3` in the cache; the
  residence bit is *derived*, not a replacement — never collapse the 4-facet
  axis to 1 bit at rest.

### Which probe / gate
- No probe gates eviction correctness directly; P3's eviction-churn measurement
  informs the GOP cadence (D-LWS-4), and the occurrent/B-frame eviction is the
  same policy as the compose-cache (D-LWS-3).

### Acceptance (fixture-level)
- Under simulated envelope pressure, an occurrent (`PERDURANT`) arena evicts
  before a continuant one.
- A continuant generator survives eviction (sticky).
- The 4-facet `dolce_id` is still resolvable post-eviction (the axis is not
  destroyed).

---

## D-LWS-8 — Probe harness (the 3 falsifiers, on real TTL + fixtures)

**Status: Queued. Label: NEW (composes shipped `splat_louvain_modularity` + `clam` + `FieldMask`).**

### Scope
This D-id PRODUCES the three gates. It is the falsifier harness, runnable on the
**real on-disk ontologies** (`data/ontologies/*.ttl`) + curated fixtures — NOT
on a Wikidata dump. Three probes:
1. **Partition locality (P1):** run `jc/examples/splat_louvain_modularity.rs`
   (Louvain = popcount-AND over `AwarenessPlane16K`) + `clam::measure_cluster_radii`
   on the FIBO/schema.org/DUL subtree; report modularity + whether CLAM radii
   coincide with cohort boundaries.
2. **Delta-card residual (P2):** reconstruct each fixture entity from its
   `FieldMask` delta over its archetype; histogram the residual per cohort.
3. **Compose hit-rate (P3):** measure ≤7-hop reachability + compose-cache churn
   via `ComposeTable::compose_chain` / blasgraph `mxm` vs a stored-edge baseline.

### The shipped primitives it builds on
- `jc/examples/splat_louvain_modularity.rs` (verified; Louvain-CLAM locality).
- `lance-graph::graph::neighborhood::clam::{measure_cluster_radii,
  analyze_pareto_convergence, ParetoAnalysis}` (verified; the radius probe).
- `contract::class_view::FieldMask::inherit` (verified; the residual measurement).
- `ontology::ttl_parse::parse_ttl_directory` (verified; the real-data loader).

### Firewall / honesty
- The harness lives where the examples live (`crates/jc/examples/` for the
  Louvain driver; a hub-side test/bench for P2/P3). `jc` is the cert crate; the
  hub owns the residual + compose measurements. `aerial` untouched.
- **This is the honesty backbone of the whole plan:** every CONJECTURE label in
  D-LWS-1..7 is discharged (promoted to FINDING or corrected) by a D-LWS-8 probe
  result recorded in the companion knowledge docs, per the CLAUDE.md insight
  update cycle (Claim → Probe → Result → promote/correct).

### Which probe / gate
- D-LWS-8 IS the gates. It does not consume a gate; it produces P1/P2/P3.

### Acceptance
- Each probe runs to completion on real TTL + fixtures and emits a pass/fail
  against its documented threshold (§3). Results recorded in
  `delta-card-addressing-integration-map.md` Probes section + `EPIPHANIES.md`.

---

## D-LWS-9 — DEFERRED: full Wikidata load (115M) into the spine

**Status: Queued — DEFERRED (terminal). Label: NEW + CONJECTURE (no dump on disk).**

### Scope
The full 115M-entity Wikidata load into the spine: the ndjson→`WikidataClass`
loader (named as "Remaining" in the D-ARM-14 STATUS_BOARD row), the dense-cohort
`Delegate` sub-tables (40M scholarly articles), the 12k-predicate composability
table, the full I/P/B GOP over the real corpus.

### Hard prerequisites (ALL must be green)
- **Every probe (P1, P2, P3) PASSED** on the real TTL + fixtures (D-LWS-8). If
  any probe fails, the design is wrong at the fixture scale and the 115M load is
  premature.
- **D-ARM-7 landed** and wired: no hydrated rule / discovered edge /
  reclassification writes the live store without passing the Jirak floor.
- D-LWS-1..7 shipped and behavior-validated on fixtures.

### Honest substrate
- **There is NO 115M Wikidata dump on disk** (grepped). This D-id cannot start
  until a dump is provisioned AND the gates are green. It is the only D-id that
  touches real Wikidata scale; everything before it is validatable today on
  `data/ontologies/*.ttl` + 6 curated classes.
- Labelled **CONJECTURE** end-to-end until the gates discharge the design.

---

## 4. Firewall summary (the one-line contract per crate)

| Crate | Role | This plan's rule |
|---|---|---|
| `lance-graph-arm-discovery` (`aerial`) | zero-dep PROPOSER | **untouched.** Never gains a heavy dep. Feeds discovery offline (splat → cohort proposals); never does runtime residency. |
| `lance-graph-contract` | zero-dep CONTRACT | gains zero-dep types only (`DeltaCard`, radix-register type, residency-policy enum) IF verified zero-dep; else they go to the hub. |
| `lance-graph-ontology` | ONTOLOGY (hub) | owns the radix register seed, composability flag table, `reconstruct`, `dolce_id` residence. |
| `lance-graph` | SPINE (hub) | owns the hydration manager, the I/P/B overlay over `VersionedGraph`, the blasgraph `mxm` driver, the prefetch cascade. |
| `bgz-tensor` | standalone codec | provides `ComposeTable` + `RouteAction`/`HhtlCache` (consumed, not modified). |
| `jc` | standalone cert | provides `jirak::prove` (D-ARM-7 engine) + the Louvain probe example. |

**The firewall holds:** aerial stays the zero-dep proposer; the hub owns
contract/ontology and the entire runtime hydration layer. No D-id in this plan
proposes making aerial depend on heavy crates.

---

## 5. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | **`EpisodicWitness64` does not exist** (cited in both companion docs). | Plan cites only `WitnessTable<64>`/`WitnessEntry` (verified). The 16-bit "book" witness tier is CONJECTURE. Flag to integration-lead: the companion docs should be corrected or `EpisodicWitness64` shipped. |
| R2 | **Lance *fragment*-versioning not wired** (only dataset-level `VersionedGraph`). | D-LWS-4 ships option (a) dataset-versioning now; option (b) fragment APIs is a NEW spike, CONJECTURE until the `lance =6.0.0` fragment surface is confirmed reachable from this crate. |
| R3 | **CLAM is a probe, not a clusterer.** | Every "CLAM-clustered" claim builds on `measure_cluster_radii` (offline placement decision); the clusterer that acts on radii is NEW. P1 gates whether radii coincide with cohorts at all. |
| R4 | **All three probes are CONJECTURE.** A failing probe invalidates the design at fixture scale. | D-LWS-8 runs them on real TTL + fixtures BEFORE D-LWS-9; gates are kill-switches, not decoration. |
| R5 | **D-ARM-7 (Jirak floor) is Queued, not shipped.** | Hard prerequisite enforced in code (D-LWS-5 write-refusal acceptance test), not just documented. No live write without it. |
| R6 | **Fan-out freeze is one-shot** (frozen ISA, append-only). | D-LWS-1 ships the conservative 16-way `NiblePath`-native register; the 256⁴ re-parameterization is CONJECTURE, frozen only after P1. |
| R7 | **Zero-dep placement of new contract types unverified.** | Each new type's home is decided by a `cargo check`/`cargo tree` zero-dep verification at implementation time; the plan names both candidate homes. |

---

## 6. Board hygiene (for the implementing session, NOT this planning agent)

Per CLAUDE.md Mandatory Board-Hygiene Rule, the session that IMPLEMENTS any
D-LWS-* must, in the same commit:
- prepend `.claude/board/INTEGRATION_PLANS.md` (this plan's index entry);
- add the D-LWS-1..9 rows to `.claude/board/STATUS_BOARD.md` (Queued → … → Shipped);
- prepend `.claude/board/AGENT_LOG.md` on completion.

**This planning agent (W1) does NOT touch any `.claude/board/*` file** — the
orchestrator owns those (per the wave iron rules). This plan file is the only
artifact W1 writes.

---

## 7. Cross-references
- THE design: `.claude/knowledge/delta-card-addressing-integration-map.md`.
- Framing: `.claude/knowledge/agnostic-lazy-world-spine.md`.
- Probe-1 driver: `crates/jc/examples/splat_louvain_modularity.rs`.
- Jirak floor: `crates/jc/src/jirak.rs`; ISSUE `ARM-JIRAK-FLOOR`; STATUS_BOARD D-ARM-7.
- Related plans: `.claude/plans/streaming-arm-nars-discovery-v1.md` (D-ARM arc),
  `.claude/specs/wikidata-hhtl-load.md` (120→38GB structural compression).
- Iron rules: `I-VSA-IDENTITIES`, `I-NOISE-FLOOR-JIRAK`,
  `I-LEGACY-API-FEATURE-GATED`; `CLAUDE.md` The Click (free-energy = prior +
  prediction-error).
