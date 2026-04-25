# Scenario / World / Counterfactual — Cross-cutting Design

**Status**: shipped (this PR adds the `ScenarioBranch` facade + wires
`archetype::World::fork` / `at_tick`).

**Audience**: anyone touching scenario branching, counterfactual
reasoning, time travel, or the Foundry-parity LF-70/71/72 chunks.

**Owner**: `scenario-world` agent (`.claude/agents/scenario-world.md`).

---

## TL;DR

Three operations are conflated in casual discussion but are
architecturally distinct:

| Operation | Question | Surface | Cost |
|---|---|---|---|
| **Time travel** | "Read state as it was at tick T" | `VersionedGraph::at_version(v)` / `archetype::World::at_tick(t)` | One Lance checkout |
| **Branching** | "Run divergent future under hypothesis H, name it `recession_2027`" | `contract::scenario::ScenarioBranch` + `ScenarioWorld` trait | New dataset path + tag + simulation |
| **Intervention** | "What if X were x'?" over a single fingerprint | `cognitive::world::counterfactual::intervene` | Pure bind/unbind math, no storage |

The `ScenarioBranch` facade composes the four pieces that already
existed:

1. **Pearl Rung 3 intervention math** in
   `lance-graph-cognitive::world::counterfactual` (5 tests passing).
2. **Lance dataset versioning + diff** in
   `lance-graph::graph::versioned::VersionedGraph` (`at_version`,
   `tag_version`, `diff`).
3. **Archetype meta-state branching** in
   `lance-graph-archetype::world::World` (URI-encoded `fork(branch)`,
   `at_tick(t)` rewind).
4. **Situational gestalt DTO** in
   `lance-graph-contract::world_model::WorldModelDto` (gestalt state,
   qualia, proprioception).

The facade is `ScenarioBranch` + `ScenarioDiff` + `ScenarioWorld`
trait in `lance-graph-contract::scenario`. Total addition to the
substrate: ~300 LOC of contract types + ~50 LOC wiring two stub
methods.

---

## Why the column-widening alternative was rejected

LF-71 v1 proposed adding a `scenario_id` column to every BindSpace
SoA + SPO row. This was rejected on five grounds:

1. **Hot-path tax**. Widens every SIMD sweep over `FingerprintColumns`,
   `QualiaColumn`, `MetaColumn`, `EdgeColumn` by 8 bytes × N rows.
   At 1M rows that's 8 MB of column read on every dispatch.

2. **Substrate duplication**. Lance datasets already provide
   ACID-versioned, time-travel-readable snapshots. Re-implementing
   branching at row layer is a parallel state machine with the same
   semantics.

3. **`I-VSA-IDENTITIES` violation**. Per CLAUDE.md iron rule, scenario
   identity is *meta about which content version*, not content itself.
   It belongs on the addressing/version layer, not as a column.

4. **Archetype unification break**. Archetypes, personas,
   thinking-styles unify by being **role catalogues with disjoint slice
   allocations** in the bundle, not new SoA columns. A `SCENARIO_KEY`
   role-bind in the existing fingerprint slice headroom would carry
   scenario identity for free.

5. **Ripple-model inconsistency**. Per
   `.claude/knowledge/user-agent-topic-ripple-model.md`, scenarios are
   spine forks in the ripple field. The fork point is metadata; the
   field state at the fork point is the substrate. Adding a column
   collapses fork metadata into substrate.

---

## Why a separate scenarios crate was rejected

The four pieces a scenario needs already exist in distinct crates with
clean dependencies. A new crate would re-state shape; a facade composes
existing surfaces.

- Cross-consumer types (SMB session, future Foundry-parity work) need
  zero-dep access — that means the contract crate.
- Concrete `ScenarioWorld` impls live downstream where they can reach
  `VersionedGraph` (lance-graph) and `multi_intervene`
  (lance-graph-cognitive).

This is the same split pattern as `OrchestrationBridge` (trait in
contract, impls in callcenter / planner / cypher).

---

## The three resolutions of `ScenarioDiff`

A scalar divergence collapses gestalt structure. Per the ripple-model
doc, a good shared gestalt stores both overlap and conflict. So
`ScenarioDiff` carries three resolutions:

```rust
pub struct ScenarioDiff {
    // Names
    pub a_name: String,
    pub b_name: String,

    // Graph layer — what changed structurally?
    // Composes VersionedGraph::diff
    pub new_entities_in_a: u32,
    pub new_entities_in_b: u32,
    pub modified_entities: u32,

    // Fingerprint layer — what changed at bit level?
    // Composes worlds_differ
    pub fingerprint_divergence: f32,

    // Gestalt layer — what changed relationally?
    // Aggregates WorldModelDto.field_state.dissonance
    pub world_model_dissonance: f32,
}
```

Maps onto the ripple-model framings:
- **Graph diff** ↔ spine-trajectory comparison (causal arc accumulation).
- **Fingerprint diff** ↔ ripple-field divergence (interference at bit
  resolution).
- **Gestalt diff** ↔ shared-gestalt overlap/mismatch (the user/agent/
  topic/angle four-pole interaction).

---

## Decision summary table

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Where to encode scenario identity | URI suffix on `archetype::World` + role-bind in trajectory | Column on BindSpace SoA | Hot-path tax + substrate duplication |
| Where to declare facade types | `lance-graph-contract::scenario` (zero-dep) | New `lance-graph-scenario` crate | Composition over re-statement |
| Storage handle ownership | Downstream resolver | Direct `lance::Dataset` in archetype | Keep archetype lance-free |
| Default inference mode | `CounterfactualSynthesis` (NARS=6) | `Deduction` (NARS=0) | Match user intent on a "what if?" call |
| Determinism strategy | Captured `fork_seed: u64` (Apache-Temporal-extracted) | Implicit non-determinism | Counterfactual research requires reproducibility |
| Forecasting primitive | Palette compose-chain (Chronos-extracted) | Embed Chronos | Our palette + ComposeTable already richer |
| Diff resolution | Three layers (graph, fingerprint, gestalt) | Single divergence scalar | Gestalt is multi-resolution by construction |

---

## Tools we considered and what we extracted

### NARS — already wired

`NarsInference::CounterfactualSynthesis` is the 7th inference type with
its own role-key slot at `[9996..10000)`. It was previously unused.
`ScenarioBranch` defaults `inference_mode: 6` so forward simulation
under a branch's hypothesis routes through it.

### Archetype — scenario priors for free

12 archetype families × 12 voice channels = 144 identity fingerprints,
each at a disjoint slice in the 16K VSA space. A `ScenarioBranch` with
`archetype_prior: Some(7)` bundles archetype 7's identity into every
trajectory, biasing forward inference toward that archetype's typical
patterns. No code in the substrate — pure role-bind composition.

### ONNX — branch-local model dispatch

ONNX models (Jina v5, ModernBERT, etc.) already run on the workspace.
`ScenarioWorld::simulate_forward(branch, steps, model)` passes an
optional `ModelBinding` (already in `contract::ontology`) so each
forward step can consult an external model with branch-local inputs.
This is what makes "given recession archetype, what does the
customer-churn model predict?" actually work.

### Chronos — extracted method only

Chronos itself (time-series-as-tokens via patch quantization) is too
primitive for our substrate. The portable idea: **chain palette
compose-table lookups** to forecast next archetype. We already have
palette + ComposeTable in `bgz17`. `ScenarioWorld::forecast_palette`
exposes this as the in-cache forecaster (~2ns/step).

### Apache Temporal — extracted method only

Apache Temporal is the wrong tool for simulation (it's for workflow
replay-on-error, not counterfactual forecasting). One useful idea
ports: **deterministic replay** via captured RNG seed at fork point.
Every `ScenarioBranch` carries `fork_seed: u64`. `ScenarioWorld::replay`
re-runs with the same seed → identical trajectory.

---

## LF-80/81 reframed: marketplace = portable auditable counterfactuals

Earlier framing called LF-80 (`OntologyBundle`) and LF-81 (cross-tenant
install) "speculative". With `ScenarioBranch` shipped, that framing
inverts. The Foundry "marketplace" stage isn't package distribution —
it's **portable, signed, auditable scenario packs** that regulated
industries (finance, insurance, supply chain, regulatory compliance)
pay hard for.

### The product the substrate now affords

A `ScenarioBundle` is a sealed unit containing:

| Component | Source | Role in bundle |
|---|---|---|
| `Ontology` | `contract::ontology` | The schema universe the scenario operates over |
| `ScenarioBranch` (one or many) | `contract::scenario` | The hypothesis: fork point + interventions + archetype prior + seed |
| `ModelBinding` set | `contract::ontology::ModelBinding` | The ONNX models invoked per forward step |
| `AuditEntry` chain | `contract::property::AuditEntry` | The provenance record — every intervention, every replay, every diff, signed |
| Evidence stream URLs | `spider-rs` ingest specs | External web data sources grounding forward steps |
| `LineageHandle` set | `contract::property::LineageHandle` | Where each piece of evidence came from + when |

This composes existing types only; no new substrate. The bundle is the
deliverable; the audit trail is the moat.

### Why this matters for enterprise

Regulated industries don't pay for forecasts — they pay for **defensible
forecasts**. A risk officer at a bank presenting "we modeled the
recession-2028 scenario" to regulators must answer:

- What hypothesis did you make? → `ScenarioBranch.interventions`
- Under what archetype assumptions? → `ScenarioBranch.archetype_prior`
- What evidence grounded the forecast? → `LineageHandle` per evidence
  point + `spider-rs` source URLs in the bundle
- What models did you run? → `ModelBinding` set with version pins
- Can you reproduce it? → `ScenarioBranch.fork_seed` +
  `ScenarioWorld::replay`
- What changed since the original run? → `ScenarioWorld::diff_branches`
  comparing replay against current state

Every one of these is a substrate primitive that already exists.

### spider-rs as the evidence ingest tier

`spider-rs` is a Rust web crawler/scraper. Its role in a counterfactual
bundle: provide reproducible, time-stamped evidence streams that
forward simulation can consult. Cleanly fits as a connector under the
external unified data-layer DTO (LF-10..14 tier) — it's just another
source impl alongside PostgreSQL, MongoDB, MS Graph, etc.

The integration shape:
1. spider-rs scrapes an evidence URL set per scenario (news, market
   data, regulatory filings, supply-chain telemetry).
2. Each scrape → `LineageHandle` capturing source + timestamp + content
   hash.
3. Forward simulation consults the evidence via `ModelBinding` (e.g.,
   sentiment ONNX over scraped news) per step.
4. The full evidence corpus + lineage ships in the `ScenarioBundle`
   for replay.

### Cross-tenant install (LF-81) becomes audit-portable scenarios

LF-81's "cross-tenant install" reframes as: "Bank A's compliance team
exports a `recession_2028` ScenarioBundle. Bank B's risk team imports
it, runs `replay()` to verify reproducibility, then runs
`simulate_forward()` against their own `Ontology` to get bank-B-specific
projections."

The cross-tenant operation isn't sharing a package — it's sharing a
**verified hypothesis with full provenance** that Bank B can reproduce
exactly before adapting.

### Recommendation update

| Item | Original verdict | New verdict | Reason |
|---|---|---|---|
| LF-80 OntologyBundle | "Speculative" | **High leverage** | Reframes as ScenarioBundle with audit trail — enterprise sells itself |
| LF-81 cross-tenant install | "Speculative" | **High leverage** | Reframes as portable verified hypothesis exchange |
| spider-rs integration | "Connector tier" | **Promote to LF-15 or earlier** | Evidence ingest is the missing piece for groundable forward simulation |

The shipping order changes: build LF-50/52 (`ModelRegistry`,
`LlmProvider`) → LF-15 spider-rs connector → LF-80
(`ScenarioBundle` = `Ontology` + `ScenarioBranch` set + bindings +
audit chain) → LF-81 (cross-tenant verify + replay).

LF-80/81 become the **anchor product**, not a tail item. Everything
else in the Foundry checklist serves the bundle.

---

## Cross-references

- `crates/lance-graph-contract/src/scenario.rs` — the facade.
- `crates/lance-graph-cognitive/src/world/counterfactual.rs` — Pearl
  Rung 3 intervention math.
- `crates/lance-graph/src/graph/versioned.rs` — Lance versioning +
  diff.
- `crates/lance-graph-archetype/src/world.rs` — `World::fork` /
  `at_tick`.
- `crates/lance-graph-contract/src/world_model.rs` — `WorldModelDto`,
  `FieldState`, `GestaltState`.
- `.claude/agents/scenario-world.md` — specialist agent card with the
  full decision tree and rejected alternatives.
- `.claude/knowledge/user-agent-topic-ripple-model.md` — theoretical
  framing.
- `.claude/knowledge/vsa-switchboard-architecture.md` — why scenario
  identity belongs in role-bind, not column.
- ADR-0001 §61-72 — dataset branching design.
- ADR-0001 §95 — tick semantics.
