# 3DGS Domain Adapter Strategy Plan — lance-graph

## Goal

Define how the wider 3DGS / HHTL / certified-field substrate should support many domains without collapsing into a single domain-spaghetti crate.

The rule:

```text
core substrate stays domain-neutral
adapters carry domain semantics
certificates cross the boundary through stable DTOs
```

## Core substrate

The core should know only about:

```text
hierarchical blocks
block metadata
kernel summaries
error certificates
skip/refine/hydrate actions
reason codes
storage references
query/render budgets
```

It should not know what a gene, neuron, ultrasound probe, asset, or customer ticket means.

## Adapter shape

Every domain adapter should implement the same conceptual flow:

```text
source payload
  -> domain metadata extraction
  -> block hierarchy
  -> kernel summary
  -> certificate request
  -> query/render/hydrate action
```

Candidate trait shape:

```rust
pub trait CertifiedFieldAdapter {
    type Source;
    type Block;
    type Query;
    type Output;

    fn describe_source(&self, source: &Self::Source) -> SourceDescriptor;
    fn build_blocks(&self, source: &Self::Source) -> Vec<Self::Block>;
    fn summarize_block(&self, block: &Self::Block) -> KernelSummary;
    fn plan(&self, query: &Self::Query, blocks: &[Self::Block]) -> Vec<BlockDecision>;
    fn execute(&self, decisions: &[BlockDecision]) -> Self::Output;
}
```

This should remain plan-level until at least two concrete adapters exist.

## Shared DTOs

Keep common DTOs narrow:

```text
SourceDescriptor
BlockDescriptor
KernelSummary
ErrorCertificateSummary
BlockDecision
DecisionReason
ExecutionBudget
ProvenanceRef
```

Do not put domain-specific fields directly in these DTOs. Use typed extension payloads or adapter-owned tables.

## Adapter candidates

### Datalake adapter

```text
source: Lance / Parquet / Iceberg dataset
block: fragment / row group / page
query: SQL / DataFusion logical plan / vector query
output: selected fragments / exact scan / approximate aggregate
```

### Geospatial adapter

```text
source: 3D Tiles / ArcGIS / Cesium / GeoJSON
block: tile / content / splat block
query: camera / spatial predicate / feature query
output: render schedule / selected features / tile decision report
```

### RAG memory adapter

```text
source: document corpus / embedding index
block: document / section / chunk family
query: user question / semantic vector / filters
output: grounded chunks + confidence report
```

### Observability adapter

```text
source: logs / traces / metrics / deploy events
block: service / time window / trace family
query: incident question / anomaly signature
output: root-cause candidates + hydrate traces
```

### Ultrasound adapter

```text
source: RF / IQ / Doppler / IMU frames
block: frame / scanline group / splat volume block
query: pose / atlas / anatomical region / time
output: fused volume / registration report
```

### Genetics adapter

```text
source: FASTA / FASTQ / expression matrix / variant set
block: sequence window / motif / gene / pathway
query: motif / variant / pathway / sample state
output: matched blocks + confidence envelope
```

### Neuronal adapter

```text
source: activation traces / connectome / model attention graph
block: neuron / synapse / microcircuit / region
query: state / activation / subgraph pattern
output: state comparison / selected subgraph
```

## Promotion criteria

A domain adapter should not become implementation work until it has:

```text
one small public or synthetic fixture
one concrete query type
one block hierarchy
one measurable certificate
one useful output
one acceptance test
```

## Repository placement

Keep Ring-1 adapters close:

```text
crates/adapter-datalake
crates/adapter-geospatial
```

Keep exploratory adapters either in docs or separate experimental crates:

```text
crates/adapter-ultrasound-experimental
crates/adapter-genetics-experimental
crates/adapter-neuronal-experimental
```

Names are placeholders.

## Certificate discipline

A certificate must affect behavior:

```text
skip
refine
hydrate exact
fallback
warn
reject
```

If it is only logged and never changes the decision, it is telemetry, not a certificate.

## Acceptance criteria

- Core DTOs remain domain-neutral.
- Two adapters can share the same decision/report machinery.
- Domain-specific tables do not pollute common schemas.
- Certificates are behavior-affecting.
- Exploratory domains cannot block Ring-1 datalake/geospatial progress.

## Implementation priority

```text
1. datalake adapter
2. geospatial adapter
3. RAG / observability adapter
4. ultrasound research adapter
5. genetics / neuronal exploratory adapters
```
