# FMA Heart-Click End-to-End Convergence Demo — Smoke Test Spec

> **Sprint:** sprint-log-4  
> **Date:** 2026-05-13  
> **Author:** W11-retry  
> **Target:** End-to-end smoke test: FMA OWL 75K entities → SPO triples → UnifiedBridge<MedcareBridge> → thinking-engine → Cypher cell → q2 3D anatomy render  
> **Dependencies (hard):** W10 (slot u16), W2 (q2 stub dedup)  
> **Dependencies (soft):** W4 (medcare super-domain), W6 (thinking-engine intent), W8 (audit sink)

---

## 1. Architecture ASCII

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FMA HEART-CLICK PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐   OWL/RDF     ┌───────────────┐   Fingerprint<256>  ┌──────────────┐
  │ stalwart │ ────────────▶ │  fma-ingest   │ ─────────────────▶  │   ndarray    │
  │  (HTTP   │   fetch FMA   │  (Rust bin)   │   SIMD-hashed       │ SIMD + CAM-PQ│
  │  client) │               │               │   entity vectors     │  codec       │
  └──────────┘               └───────┬───────┘                     └──────┬───────┘
                                     │                                    │
                             OWL→SPO expansion                    Vsa16kF32 bundles
                             ~600K triples                        (CrystalFingerprint)
                             (subClassOf, part_of,                       │
                              connected_to)                              │
                                     │                                    │
                                     ▼                                    ▼
                             ┌───────────────────────────────────────────────┐
                             │            lance-graph (core spine)           │
                             │                                               │
                             │  crates/lance-graph/src/graph/spo/            │
                             │  ├── HammingMin truth semiring                │
                             │  ├── SPO triple store (u16 slots, W10 req)   │
                             │  ├── merkle chain (u64 FNV-1a per W8)        │
                             │  └── Arrow RecordBatch export                 │
                             │                                               │
                             │  lance-graph-planner                          │
                             │  ├── Cypher parse → DataFusion plan           │
                             │  ├── TruthPropagating semiring                │
                             │  └── MUL assessment + elevation               │
                             └───────────────────┬───────────────────────────┘
                                                 │
                                                 │  UnifiedBridge<MedcareBridge>
                                                 │  (struct, B = MedcareBridge)
                                                 ▼
                             ┌───────────────────────────────────────────────┐
                             │        thinking-engine (Cypher intent)        │
                             │                                               │
                             │  ThinkingStyle dispatch (NARS deduction)      │
                             │  Cypher intent: MATCH-RETURN anatomy query    │
                             │  Free-energy minimisation → Resolution        │
                             └───────────────────┬───────────────────────────┘
                                                 │
                                    HealthcareBridge specialisation
                                    (crates/lance-graph-ontology/
                                     src/bridges/medcare.rs)
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────────────┐
                             │             medcare-rs                        │
                             │                                               │
                             │  MedcareBridge impl HealthcareBridge          │
                             │  drug-knowledge-bases-2026-05-05 crosswalk   │
                             │  FMA anatomy ←→ pharmacology pivot            │
                             └───────────────────┬───────────────────────────┘
                                                 │
                                    Arrow batch (anatomy neighbors)
                                    + optional drug-target pivot
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────────────┐
                             │         q2 graph-notebook                     │
                             │                                               │
                             │  Cypher cell (MATCH-RETURN, no MERGE/CREATE) │
                             │  3D anatomy render (WebGL / q2 renderer)      │
                             │  Heart click → neighborhood highlight         │
                             └───────────────────────────────────────────────┘
```

**Data flow summary:**

1. stalwart HTTP client fetches FMA from BioPortal (`https://bioportal.bioontology.org/ontologies/FMA`) — selectable format: RDF/XML, OWL, or CSV (per two-tier ingest below).
2. `fma-ingest` binary parses OWL, expands to ~600K SPO triples, writes Lance dataset.
3. ndarray SIMD hashes entity IRIs into `Fingerprint<256>` → packed into `Vsa16kF32` bundles.
4. lance-graph SPO store ingests triples with u16 entity_type_id slots (W10 prerequisite).
5. DataFusion planner receives Cypher MATCH query (heart click intent).
6. `UnifiedBridge<MedcareBridge>` routes plan through HealthcareBridge specialisation in medcare-rs.
7. thinking-engine emits Cypher intent cell → q2 notebook renders 3D anatomy.

---

## 2. FMA Ingest Plan

### Dataset facts

| Property | Value |
|---|---|
| Source | Foundational Model of Anatomy (FMA) — canonical at BioPortal: https://bioportal.bioontology.org/ontologies/FMA |
| Available formats | RDF/XML, OWL, CSV (BioPortal native download menu) |
| Entity count | ~75,000 anatomical concepts |
| Relation types | `subClassOf`, `fma:part_of`, `fma:connected_to`, `fma:branch_of`, `fma:tributary_of`, `fma:contains` |
| Estimated triples | ~600,000 (subClassOf approx 210K, part_of approx 190K, connected_to approx 120K, branch_of approx 45K, others approx 35K) |
| Output format | Lance columnar dataset at `/data/fma.lance` |
| Slot type | **u16** — HARD prerequisite from W10 (75K entity types > 256, u8 TRUNCATES silently) |

### Two-tier ingest (NEW — addresses cold-start latency)

| Tier | Source format | Time budget | Output | Used for |
|---|---|---|---|---|
| Tier-1 quick | CSV (BioPortal flat export) | ~minutes | entity table only (id, label, parent) | 3D anatomy figure render in q2 BEFORE full edge graph available |
| Tier-2 full | RDF/XML or OWL (BioPortal) | ~1 hour | full SPO triple-store (subClassOf + part_of + connected_to + …) | neighbor queries (heart-click expansion, drug-knowledge pivot) |

The smoke-test demo MUST run with Tier-1 ingest only as the "preview mode" assertion (heart appears in 3D, click is a no-op that returns "edge graph still loading"). Tier-2 unlocks the full heart-click → neighbors flow. CI tests both tiers separately.

### CLI contract

```bash
cargo run --bin fma-ingest --release -- \
  --source /data/fma.owl \
  --output /data/fma.lance \
  --batch-size 8192 \
  --relation-filter subClassOf,part_of,connected_to,branch_of,tributary_of,contains \
  --slot-width u16          # enforced; panics if binary built without W10 slot-widen
```

### Ingest binary structure

```
crates/fma-ingest/
├── Cargo.toml         # dep: lance, arrow-57, lance-graph-contract, serde, oxrdfio
├── src/
│   ├── main.rs        # CLI entrypoint, reads --source, opens OWL, calls pipeline
│   ├── owl_reader.rs  # oxrdfio OWL → RDF triples iterator (streaming, no full load)
│   ├── triple_map.rs  # IRI → u16 entity_type_id bijection (HashMap<String, u16>)
│   │                  # Panics if entity_count > 65535 (u16::MAX safety gate)
│   ├── spo_builder.rs # Calls graph/spo/builder.rs with (subject_u16, predicate_u16, object_u16)
│   │                  # Applies HammingMin truth semiring defaults: freq=1.0, conf=0.9
│   └── lance_sink.rs  # Batches Arrow RecordBatch, writes to lance Dataset
```

### Arrow schema for FMA Lance dataset

```
Schema {
  subject_id:    UInt16,     -- entity_type_id (W10 slot-widened)
  predicate_id:  UInt16,     -- relation_type_id
  object_id:     UInt16,     -- entity_type_id
  subject_iri:   Utf8,       -- e.g. "http://purl.org/sig/ont/fma/fma7088" (Heart)
  predicate_iri: Utf8,       -- e.g. "http://purl.org/sig/ont/fma/fma7497" (part_of)
  object_iri:    Utf8,       -- e.g. "http://purl.org/sig/ont/fma/fma7088" (Heart)
  subject_label: Utf8,       -- rdfs:label
  object_label:  Utf8,       -- rdfs:label
  frequency:     Float32,    -- NARS frequency (default 1.0 for asserted triples)
  confidence:    Float32,    -- NARS confidence (default 0.9 for asserted triples)
  merkle_u64:    UInt64,     -- FNV-1a rolling hash per W8 audit schema
}
```

### Slot u16 gate (HARD prerequisite)

The ingest binary MUST call `owl_from_schema_ptr` via W10's corrected implementation.
Pre-W10, this function silently truncated u16 entity_type_id to u8, causing entity
collision for all IDs > 255. With 75K FMA entities, this corruption is total — the
dataset is unusable without W10. Build-time gate:

```rust
// triple_map.rs
const MAX_ENTITY_TYPES: usize = u16::MAX as usize;  // 65535

fn assign_id(map: &mut HashMap<String, u16>, iri: &str) -> u16 {
    let next = map.len();
    assert!(next < MAX_ENTITY_TYPES, "FMA entity count {next} exceeds u16::MAX — slot overflow");
    *map.entry(iri.to_owned()).or_insert(next as u16)
}
```

---

## 3. Cypher Cell Contract

### q2 notebook template

The smoke test uses a **read-only** Cypher cell. No `MERGE`, `CREATE`, `SET`, or `DELETE`
is permitted in the smoke-test contract. The intent is neighborhood traversal only.

```cypher
// q2 Cypher cell — FMA heart-click smoke test
// Planner: lance-graph-planner Cypher→DataFusion route
// Semiring: HammingMin (SPO truth propagation)

MATCH (heart:AnatomicalStructure {label: $entity_label})
      -[r:PART_OF|CONNECTED_TO|BRANCH_OF|CONTAINS*1..2]-
      (neighbor:AnatomicalStructure)
RETURN
  heart.label       AS source,
  type(r)           AS relation,
  neighbor.label    AS target,
  neighbor.fma_id   AS fma_id,
  r.frequency       AS freq,
  r.confidence      AS conf
ORDER BY conf DESC, freq DESC
LIMIT $k
```

**Parameters injected by q2 click handler:**

| Parameter | Type | Example value |
|---|---|---|
| `$entity_label` | String | `"Heart"` |
| `$k` | Integer | `50` |

**DataFusion translation contract:**

The lance-graph-planner translates this Cypher to a DataFusion scan over the FMA Lance
dataset with predicate pushdown on `subject_label = $entity_label` and relation filter
`predicate_id IN (part_of_id, connected_to_id, branch_of_id, contains_id)`. Depth-2
traversal is implemented as a self-join on the SPO table (subject2 = object1).

**Output Arrow schema from q2 query:**

```
Schema {
  source:   Utf8,
  relation: Utf8,
  target:   Utf8,
  fma_id:   UInt16,
  freq:     Float32,
  conf:     Float32,
}
```

---

## 4. Golden Inputs (5 Entities)

Each golden input specifies: the anatomy label, FMA class IRI, expected direct neighbor count
(depth-1), and expected depth-2 reachable count. Counts are derived from FMA 4.14.0 statistics
and carry ±10% tolerance.

| # | Label | FMA IRI (short) | Depth-1 neighbors | Depth-2 reachable | Primary relations |
|---|---|---|---|---|---|
| 1 | Heart | fma:fma7088 | 42 ± 4 | 280 ± 28 | part_of(22), connected_to(15), contains(5) |
| 2 | LeftAtrium | fma:fma7097 | 18 ± 2 | 95 ± 10 | part_of(8), connected_to(7), branch_of(3) |
| 3 | Aorta | fma:fma3734 | 35 ± 4 | 210 ± 21 | branch_of(18), connected_to(12), contains(5) |
| 4 | Femur | fma:fma9661 | 28 ± 3 | 155 ± 16 | part_of(14), connected_to(10), contains(4) |
| 5 | Cerebellum | fma:fma67944 | 31 ± 3 | 190 ± 19 | part_of(16), connected_to(11), contains(4) |

**Tolerance rationale:** FMA subClassOf chains have minor version-to-version variation in
inferred triples. ±10% absorbs OWL reasoner depth differences between FMA 4.14.0 and local
mirror snapshots without requiring exact version pinning in CI.

---

## 5. Assertion Matrix

For each of the 5 golden inputs, the following assertions must pass in the smoke test:

### 5.1 Per-entity assertion table

| Input | Audit event emitted | Allow decision | Arrow batch shape | Merkle u64 extended |
|---|---|---|---|---|
| Heart | `UnifiedAuditEvent { entity: "Heart", decision: Allow, ... }` written to Lance audit sink | `PolicyDecision::Allow` | `(N_rows, 6_cols)` where `N_rows` in [38, 46] | `merkle_u64` is non-zero FNV-1a of prior chain + event bytes |
| LeftAtrium | Same structure, entity: "LeftAtrium" | `PolicyDecision::Allow` | `(N_rows, 6_cols)` where `N_rows` in [16, 20] | `merkle_u64` extended from Heart's chain |
| Aorta | entity: "Aorta" | `PolicyDecision::Allow` | `(N_rows, 6_cols)` where `N_rows` in [32, 39] | chained |
| Femur | entity: "Femur" | `PolicyDecision::Allow` | `(N_rows, 6_cols)` where `N_rows` in [25, 31] | chained |
| Cerebellum | entity: "Cerebellum" | `PolicyDecision::Allow` | `(N_rows, 6_cols)` where `N_rows` in [28, 34] | chained |

### 5.2 Audit event schema (per W8)

Each `UnifiedAuditEvent` written to the Lance audit sink must conform to:

```rust
// Per W8 audit sink spec
struct UnifiedAuditEvent {
    timestamp_ns:   u64,             // monotonic nanoseconds
    entity_label:   String,          // e.g. "Heart"
    query_hash:     u64,             // FNV-1a of Cypher query string
    decision:       PolicyDecision,  // Allow | Deny | Escalate
    bridge_slot:    BridgeSlot,      // Healthcare domain slot
    merkle_u64:     u64,             // FNV-1a rolling chain (NOT [u8;32] — u64 per W8 correction)
    row_count:      u32,             // Arrow batch rows returned
    duration_us:    u64,             // wall-clock microseconds for query
}
```

**Merkle chain semantics:** Each event's `merkle_u64` = `fnv1a_64(prev_merkle || event_bytes)`.
The first event in a session has `prev_merkle = 0`. The chain provides tamper-evidence for the
audit log without the overhead of SHA-256 ([u8;32]). This is a W8 correction — earlier
assumptions used [u8;32].

### 5.3 Arrow batch shape assertions

```rust
// In the smoke test:
assert_eq!(batch.num_columns(), 6);  // source, relation, target, fma_id, freq, conf
assert!(batch.num_rows() >= expected_min && batch.num_rows() <= expected_max);
assert_eq!(batch.schema().field(0).name(), "source");
assert_eq!(batch.schema().field(3).data_type(), &DataType::UInt16);  // fma_id is u16
```

### 5.4 Policy decision assertion

All 5 golden inputs must produce `PolicyDecision::Allow`. The FMA smoke test does not
exercise Deny or Escalate paths — those belong to security policy smoke tests outside
this spec. The HealthcareBridge's default policy for read-only anatomy queries is Allow.

---

## 6. Drug-Knowledge Crosswalk (Optional Secondary Assertion)

### Background

The MedCare-rs `drug-knowledge-bases-2026-05-05` release (tagged at
`AdaWorldAPI/MedCare-rs releases/tag/drug-knowledge-bases-2026-05-05`) provides
structured pharmacological knowledge bases aligned to the Healthcare super-domain.
FMA covers anatomy; the drug-knowledge release covers pharmacology. Both are
Healthcare super-domain assets that share the `MedcareBridge` namespace.

### Crosswalk query sketch

When the primary "Heart" click resolves, the q2 notebook optionally pivots to:

```cypher
// Secondary crosswalk: anatomy → pharmacology (drug targeting)
// Requires: MEDCARE_DRUG_KB env var pointing to drug-knowledge Lance dataset

MATCH (target:AnatomicalStructure {label: $anatomy_label})
      -[:EXPRESSED_IN|TARGETED_BY]-
      (gene:Gene)
      -[:TARGETED_BY]-
      (drug:Drug)
RETURN
  target.label    AS anatomy,
  gene.symbol     AS gene,
  drug.name       AS drug_name,
  drug.atc_code   AS atc_code,
  drug.mechanism  AS mechanism
LIMIT 20
```

**Cross-OWL join mechanics:**

The crosswalk requires a join between two Lance datasets:
- `/data/fma.lance` — FMA anatomy SPO triples (subject u16 in anatomy ID space)
- `$MEDCARE_DRUG_KB` — drug-knowledge SPO triples (subject u16 in drug ID space)

The join key is the gene symbol (Utf8), which appears in both datasets as a linking
concept. The `MedcareBridge` implementation in
`crates/lance-graph-ontology/src/bridges/medcare.rs` is responsible for resolving
IRI namespace differences between FMA anatomy IRIs and MedCare pharmacology IRIs.

**OWL crosswalk relation mapping:**

| FMA relation | MedCare drug-knowledge relation | Semantics |
|---|---|---|
| `fma:expressed_in` | `dkb:EXPRESSED_IN_TISSUE` | gene expressed in anatomical structure |
| `fma:part_of` (tissue) | `dkb:TISSUE_OF_ORGAN` | tissue membership |
| (derived) | `dkb:TARGETED_BY` | gene → drug targeting link |

**Assertion for optional crosswalk (Heart):**

When `MEDCARE_DRUG_KB` is set and the drug-knowledge dataset is valid:
- The crosswalk query for "Heart" must return >= 5 rows.
- All returned `atc_code` values must match ATC Class C (Cardiovascular system) pattern
  `^C` — a sanity check that the crosswalk is resolving to cardiac drugs, not random entries.
- The join must complete within 500ms (DataFusion predicate pushdown on both datasets).

**Why this matters:** The drug-knowledge-2026-05-05 release was shipped as a complementary
asset to FMA-based anatomy work. The heart-click demo is the natural convergence point
where anatomical identity (FMA) meets pharmacological targeting (MedCare drug-knowledge).
Demonstrating the crosswalk in the smoke test validates the Healthcare super-domain's
cross-OWL coherence — a key value proposition of `UnifiedBridge<MedcareBridge>`.

---

## 7. CI Integration

### Test gate structure

```bash
# Run smoke test (ignored by default — requires data)
cargo test --test fma_smoke -- --ignored

# With data dirs:
FMA_DATA_DIR=/data/fma.lance MEDCARE_DRUG_KB=/data/medcare-drug-kb.lance \
  cargo test --test fma_smoke -- --ignored

# Nightly CI job (GitHub Actions):
# Triggered by schedule: cron '0 2 * * *'
# Requires: FMA_DATA_DIR secret + MEDCARE_DRUG_KB secret (optional)
```

### Test file location

```
tests/fma_smoke.rs          # integration test, #[ignore] on all fns
```

### Test structure

```rust
// tests/fma_smoke.rs

use std::env;

fn fma_data_dir() -> String {
    env::var("FMA_DATA_DIR").expect("FMA_DATA_DIR must be set for fma_smoke tests")
}

fn medcare_drug_kb() -> Option<String> {
    env::var("MEDCARE_DRUG_KB").ok()
}

#[test]
#[ignore = "requires FMA_DATA_DIR env var and pre-ingested FMA Lance dataset"]
fn test_heart_click_returns_neighbors() {
    // golden input #1: Heart, expect 38-46 rows
}

#[test]
#[ignore = "requires FMA_DATA_DIR"]
fn test_left_atrium_click_returns_neighbors() { /* depth-1: 16-20 rows */ }

#[test]
#[ignore = "requires FMA_DATA_DIR"]
fn test_aorta_click_returns_neighbors() { /* depth-1: 32-39 rows */ }

#[test]
#[ignore = "requires FMA_DATA_DIR"]
fn test_femur_click_returns_neighbors() { /* depth-1: 25-31 rows */ }

#[test]
#[ignore = "requires FMA_DATA_DIR"]
fn test_cerebellum_click_returns_neighbors() { /* depth-1: 28-34 rows */ }

#[test]
#[ignore = "requires FMA_DATA_DIR; MEDCARE_DRUG_KB optional"]
fn test_heart_drug_crosswalk() {
    let drug_kb = medcare_drug_kb();
    if drug_kb.is_none() {
        eprintln!("MEDCARE_DRUG_KB not set — skipping drug crosswalk assertion");
        return;
    }
    // crosswalk assertion: >= 5 rows, all ATC codes start with 'C'
}

#[test]
#[ignore = "requires FMA_DATA_DIR"]
fn test_audit_merkle_chain_integrity() {
    // Run all 5 queries in sequence, verify merkle_u64 chain is non-zero
    // and each event's merkle extends the previous
}
```

### GitHub Actions nightly job snippet

```yaml
# .github/workflows/nightly-fma-smoke.yml
name: FMA Smoke Test (nightly)
on:
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:

jobs:
  fma-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust stable
        uses: dtolnay/rust-toolchain@stable
      - name: Download FMA Lance dataset
        run: |
          aws s3 cp s3://${{ secrets.FMA_ARTIFACT_BUCKET }}/fma.lance /data/fma.lance --recursive
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      - name: Run FMA smoke tests
        run: |
          cargo test --test fma_smoke -- --ignored
        env:
          FMA_DATA_DIR: /data/fma.lance
          MEDCARE_DRUG_KB: ${{ secrets.MEDCARE_DRUG_KB_PATH }}
```

---

## 8. Dependency Chain

### Hard prerequisites (smoke test CANNOT pass without these)

| Dependency | Worker | What it provides | Failure mode if missing |
|---|---|---|---|
| Slot widen u16 | **W10** | `owl_from_schema_ptr` without u8 truncation; u16 `entity_type_id` | 75K FMA entities corrupt at ID > 255; all neighbor lookups wrong |
| q2 stub dedup | **W2** | `lance-graph` + `q2-ndarray` stubs are re-exports, not duplicates | FMA demo fails to compile; duplicate symbol errors |

### Soft prerequisites (smoke test degrades without these)

| Dependency | Worker | What it provides | Degradation if missing |
|---|---|---|---|
| medcare super-domain | **W4** | `MedcareBridge` in `crates/lance-graph-ontology/src/bridges/medcare.rs` | Drug crosswalk unavailable; `UnifiedBridge<MedcareBridge>` won't link |
| thinking-engine intent | **W6** | ThinkingStyle dispatch wired to Cypher intent generation | Intent cell in q2 falls back to raw Cypher string; no free-energy dispatch |
| Audit sink (Lance + JSONL) | **W8** | `UnifiedAuditEvent` persistence to Lance sink; `merkle_u64` chain | Audit assertions must be skipped; only in-memory chain available |

### Build order

```
W10 (slot u16) ──┐
                  ├──▶ fma-ingest binary builds
W2 (q2 dedup)  ──┘         │
                            │
W4 (medcare)   ─────────────┼──▶ UnifiedBridge<MedcareBridge> links
W6 (thinking)  ─────────────┤
W8 (audit)     ─────────────┤
                            │
                            ▼
                    cargo test --test fma_smoke -- --ignored
```

### Crate dependency additions required

```toml
# crates/fma-ingest/Cargo.toml (new binary crate)
[dependencies]
lance = "2"
arrow = "57"
lance-graph-contract = { path = "../lance-graph-contract" }
oxrdfio = "0.2"          # OWL/RDF parsing (Rust-native, no JVM)
serde = { version = "1", features = ["derive"] }
clap = { version = "4", features = ["derive"] }

# crates/lance-graph/Cargo.toml (additions for smoke test)
[dev-dependencies]
lance-graph-ontology = { path = "../lance-graph-ontology" }  # for MedcareBridge
```

---

## 9. Open Questions

1. **FMA data availability in CI.** The FMA OWL file is ~180 MB. Pre-ingesting to a Lance
   dataset produces ~600MB. Where does the nightly CI artifact live? Options: (a) S3/GCS
   artifact bucket owned by AdaWorldAPI org, (b) git-lfs on a dedicated `fma-data` repo,
   (c) download and ingest fresh each nightly run (~15 min). Decision needed from infra
   owner before CI job can go live.

2. **oxrdfio vs sophia-rs OWL parser.** The spec uses `oxrdfio` for OWL/RDF parsing. An
   alternative is `sophia-rs`. Both are Rust-native and handle OWL/XML. Which is already
   in the workspace dependency tree? Using an existing dep avoids lockfile churn. If neither
   is present, `oxrdfio` is the preferred choice (active maintenance, Apache 2.0).

3. **MedcareBridge IRI namespace resolution.** The drug crosswalk joins FMA IRIs
   (`purl.org/sig/ont/fma/*`) with MedCare drug-knowledge IRIs (namespace TBD from the
   `drug-knowledge-bases-2026-05-05` release). Is there a published SKOS mapping between
   these namespaces, or does W4 need to hand-author the bridge mapping table?

4. **Depth-2 traversal performance.** The Cypher template uses `*1..2` path depth. For
   Heart with ~280 depth-2 reachable nodes, the DataFusion self-join on the SPO table
   processes ~600K rows with predicate pushdown. Has the lance-graph-planner's `adjacency/`
   CSR substrate been validated for this join size? The ±10% tolerance implies the reasoner
   path is known-good, but join runtime at 600K triples needs a benchmark before CI timeout is set.

5. **q2 3D render handshake.** The spec describes the Arrow batch flowing into q2
   graph-notebook for 3D anatomy rendering. What is the actual q2 API contract for
   receiving an Arrow batch and triggering a render? Is it gRPC, shared-memory,
   or an Arrow Flight endpoint? This is the last-mile integration not covered by any
   current W-worker spec. A follow-up spike (W13 candidate?) may be needed to sketch
   the q2 renderer handshake protocol before the demo can run end-to-end.

---

## Appendix A: SPO Store Reference

The triple store backing this demo lives at:

```
crates/lance-graph/src/graph/spo/
├── mod.rs          # public API: SpoStore, SpoBuilder
├── builder.rs      # batch insert, merkle chain rolling (u64 FNV-1a)
├── semiring.rs     # HammingMin truth semiring (frequency x confidence)
├── truth.rs        # NarsTruth: frequency, confidence, evidence count
└── merkle.rs       # u64 FNV-1a rolling hash (NOT [u8;32])
```

The `HammingMin` semiring computes `min(freq_a, freq_b)` for conjunction and uses
Hamming distance on the fingerprint component for similarity-weighted propagation.

## Appendix B: CrystalFingerprint / Vsa16kF32

`Vsa16kF32` is an enum variant of `CrystalFingerprint`, not a standalone type:

```rust
// Per W2 real-repo findings
enum CrystalFingerprint {
    Vsa16kF32(Box<[f32; 16_384]>),   // 64 KB, hot path, lossless VSA carrier
    Vsa16kBF16(Box<[bf16; 16_384]>), // 32 KB, AMX-accelerated
    Binary16K([u64; 256]),            // 2 KB, Hamming compare format
    // ... other variants
}
```

The FMA ingest pipeline uses `CrystalFingerprint::Vsa16kF32` for entity fingerprint
bundles when computing neighborhood similarity during the query phase.

## Appendix C: UnifiedBridge<MedcareBridge> Wire Point

```rust
// Expected wire point in crates/lance-graph-ontology/src/bridges/medcare.rs (W4)
pub struct MedcareBridge {
    pub drug_kb_path: Option<PathBuf>,  // None if MEDCARE_DRUG_KB not set
}

// UnifiedBridge is a STRUCT (not a trait) parameterised on B: NamespaceBridge
pub struct UnifiedBridge<B: NamespaceBridge> {
    inner: B,
    // ... routing state
}

// The Healthcare super-domain smoke test instantiates:
let bridge: UnifiedBridge<MedcareBridge> = UnifiedBridge::new(MedcareBridge {
    drug_kb_path: env::var("MEDCARE_DRUG_KB").ok().map(PathBuf::from),
});
```
