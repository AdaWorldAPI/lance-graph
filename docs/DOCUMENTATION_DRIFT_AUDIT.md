# Documentation Drift Audit

> **Audit date**: 2026-03-17
> **Repositories**: ladybug-rs, lance-graph

---

## Methodology

For each documentation file, the "last modified" git commit date is compared against
the latest modification date of the source files it describes. When source files
have been modified after the documentation, drift is likely.

---

## lance-graph

### crates/lance-graph/README.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-03 | `34fb4f5` refactor: remove Simple executor |
| **hdr.rs (Cascade)** | 2026-03-14 | `0c2496c` clippy cleanup |
| **ndarray_bridge.rs (SIMD)** | 2026-03-16 | `229db76` wire ndarray bridge |
| **metadata.rs (cold path)** | 2026-03-16 | `4f16d59` add MetadataStore |
| **vector_ops.rs (hamming)** | 2026-03-16 | `229db76` wire ndarray bridge |
| **types.rs (BitVec)** | 2026-03-16 | `229db76` wire ndarray bridge |

**Drift**: 13 days. README does not mention:
- BlasGraph module (16,384-bit vectors, 7 semirings)
- HDR Cascade (three-stage exposure meter)
- SIMD dispatch (4-tier AVX-512/AVX2/scalar)
- MetadataStore cold-path skeleton
- Hamming distance/similarity UDFs
- `compute_hamming_distances()` / `compute_hamming_similarities()`
- SPO triple store

**Severity**: HIGH — the README describes a Cypher query engine but the crate now
includes a full binary vector search stack and graph algebra.

### AGENTS.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-02-13 | `c248af2` move crate to crates/ |
| **Latest source changes** | 2026-03-16 | `4f16d59` MetadataStore |

**Drift**: 31 days. Build commands and test instructions may be outdated. The
crate structure has changed significantly (BlasGraph, HDR, SIMD, metadata).

**Severity**: MEDIUM — developer onboarding instructions stale.

### docs/project_structure.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-02-13 | `c248af2` move crate to crates/ |
| **Latest source changes** | 2026-03-16 | Multiple PRs |

**Drift**: 31 days. Missing the entire `graph/blasgraph/` subtree, `graph/metadata.rs`,
and `graph/spo/` module.

**Severity**: HIGH — file map is incomplete.

### .claude/BELICHTUNGSMESSER.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-13 | `2f34f92` feat: add Belichtungsmesser |
| **hdr.rs** | 2026-03-14 | `0c2496c` clippy cleanup |

**Drift**: 1 day. Minor — only clippy cleanups, no semantic change.

**Severity**: LOW — design doc matches implementation.

### PR_DESCRIPTION.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-13 | `7cb77d0` docs: add PR description |
| **Latest source changes** | 2026-03-16 | `4f16d59` MetadataStore |

**Drift**: 3 days. Missing MetadataStore cold-path addition (PR #14).

**Severity**: LOW — PR description is a point-in-time snapshot, not living documentation.

---

## ladybug-rs

### docs/spo_3d/SCHEMA.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-01 | `e084042` PR #158 |
| **meta.rs (MetaView)** | 2026-03-01 | `e084042` PR #158 |
| **schema.rs (SchemaSidecar)** | 2026-03-01 | `e084042` PR #158 |

**Drift**: 0 days. Doc and source last modified in same PR.

**Note**: SCHEMA.md describes W8-11 as `DN parent/first_child/next_sibling/prev_sibling`
while `meta.rs` describes W8-11 as `DN rung + 7-Layer compact + collapse gate`.
This is a **spec vs implementation divergence**, not a temporal drift. SCHEMA.md
describes the SPO 3D variant; `meta.rs` describes the general-purpose Container 0
layout. Both are correct for their contexts, but the difference is not documented.

**Severity**: MEDIUM — readers may confuse which layout applies where.

### ARCHITECTURE.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-01 | `e084042` PR #158 |
| **CLAUDE.md (guardrails)** | 2026-03-12 | `b61d90f` overhaul |

**Drift**: 11 days. CLAUDE.md was overhauled with new guardrails (5 Cypher paths,
private spo.rs, hot/cold invariant). ARCHITECTURE.md may not reflect these.

**Severity**: LOW — CLAUDE.md is the authoritative guardrails doc now.

### HANDOVER.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-01 | `e084042` PR #158 |
| **Latest prompts added** | 2026-03-12+ | prompts 25, 26 |

**Drift**: ~11 days. Handover predates prompts 25 (Node/Plane/Mask) and 26
(entry point), which redefine the primary object model.

**Severity**: MEDIUM — the handover still references older architectural framing.

### docs/COGNITIVE_RECORD_192.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-01 | `e084042` PR #158 |
| **Status in doc** | SUPERSEDED | Header says so |

**Drift**: N/A — correctly marked as superseded. The 192-word layout was never
implemented. Current spec is 16K-bit (256 x u64 per container).

**Severity**: NONE — historical document with correct status header.

### docs/COGNITIVE_RECORD_256.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-01 | `e084042` PR #158 |

**Drift**: 16 days since PR #158, but no schema changes since then.

**Severity**: LOW — no source changes to invalidate it.

### CLAUDE.md

| Item | Date | Commit |
|------|------|--------|
| **Doc last modified** | 2026-03-12 | `b61d90f` overhaul |
| **Latest source changes** | 2026-03-12 | Same date |

**Drift**: 0 days. Most recently updated document. Contains accurate guardrails.

**Severity**: NONE — this is the freshest documentation.

---

## Summary

| Document | Repo | Drift (days) | Severity |
|----------|------|-------------|----------|
| crates/lance-graph/README.md | lance-graph | 13 | **HIGH** |
| docs/project_structure.md | lance-graph | 31 | **HIGH** |
| AGENTS.md | lance-graph | 31 | MEDIUM |
| docs/spo_3d/SCHEMA.md | ladybug-rs | 0 (spec divergence) | MEDIUM |
| HANDOVER.md | ladybug-rs | 11 | MEDIUM |
| PR_DESCRIPTION.md | lance-graph | 3 | LOW |
| ARCHITECTURE.md | ladybug-rs | 11 | LOW |
| BELICHTUNGSMESSER.md | lance-graph | 1 | LOW |
| COGNITIVE_RECORD_256.md | ladybug-rs | 16 | LOW |
| CLAUDE.md | ladybug-rs | 0 | NONE |
| COGNITIVE_RECORD_192.md | ladybug-rs | N/A | NONE |

### Recommendations

1. **lance-graph README.md** — needs rewrite to cover BlasGraph, HDR Cascade, SIMD dispatch, MetadataStore, and UDFs
2. **lance-graph project_structure.md** — needs complete file tree update including `blasgraph/`, `metadata.rs`, `spo/`
3. **ladybug-rs SCHEMA.md** — add a note clarifying that W8-11 layout differs between SPO 3D variant and general Container 0
4. **ladybug-rs HANDOVER.md** — add reference to prompts 25-26 (Node/Plane/Mask)
