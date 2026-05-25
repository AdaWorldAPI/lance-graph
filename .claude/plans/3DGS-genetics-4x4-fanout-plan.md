# 3DGS Genetics 4x4 Fanout Plan — lance-graph

## Goal

Explore how the 3DGS raw-field and 4x4 cognitive-shader representation can fan out into genetics and molecular sequence analysis.

This is an exploratory plan. It should not block the geospatial or ultrasound paths.

## Core analogy

```text
3DGS geospatial field
  local anisotropic kernels over space

genetic field
  local anisotropic kernels over sequence, expression, structure, and time
```

A genetic block is not literally a visual splat. It is a compact kernel carrying local evidence and uncertainty.

## 4x4 carrier interpretation

```text
lane0: sequence coordinate / genomic locus / k-mer index
lane1: transition state / motif covariance / edit-distance neighborhood
lane2: expression / methylation / abundance / confidence
lane3: time / sample / lineage / provenance
```

## (4x4)^4 grammar

Use `(4x4)^4` as a four-level block hierarchy:

```text
level 0: local motif carrier
level 1: motif block
level 2: gene/regulatory region block
level 3: pathway / cell-state / population block
```

This mirrors:

```text
splat -> block -> tile -> region
motif -> gene -> pathway -> phenotype
```

## Data sources

Potential future sources:

```text
FASTA / FASTQ
variant calls
expression matrices
single-cell count matrices
methylation tracks
protein/domain annotations
pathway graphs
```

## Graph model

```text
Sample -> SequenceFrame -> MotifBlock -> Gene -> Pathway -> Phenotype
Sample -> ExpressionFrame -> ExpressionBlock -> CellType -> State
```

## Tables

Possible tables:

```text
genetic_sources
genetic_frames
motif_blocks
gene_features
variant_features
expression_blocks
genetic_certificates
```

## Kernel-fit idea

Convert local sequence/expression regions into kernels:

```text
sequence window
  -> k-mer / edit neighborhood / motif statistics
  -> 4x4 carrier
  -> certificate
  -> graph feature
```

## Certificates

Possible certificate types:

```text
motif sampling confidence
variant neighborhood stability
expression block variance
batch-effect residual
lineage transition confidence
quantization error
```

## HHTL relevance

Use HHTL-style cascade to avoid full dense scans:

```text
HEEL: k-mer / hash prefix candidate
HIP: edit-distance or motif score
TWIG: expression/covariance refinement
LEAF: exact alignment/statistical test
```

## Acceptance criteria for promotion

Promote this into implementation only when:

- a small public fixture is selected
- a domain-specific decoder is scoped
- the 4x4 carrier has a precise biological meaning
- certificates are meaningful and not decorative
- no medical/diagnostic claims are made

## Boundary

This plan is for representation and research tooling. It is not a clinical genomics product plan.
