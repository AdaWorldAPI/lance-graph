# Phase 2 — Adjacent crate landings (post-POC)

Captured per user direction ("don't forget to add SoA primary citizen as speed
lane in lance-graph-contract and lance-graph-ontology as OGIT/DOLCE inheritance
of CAM codebooks and lance-graph-callcenter for ractor/surrealdb").

These build **on** the 12-task SurrealDB-on-Lance container POC; they are NOT
part of the POC's 12 tasks. They land after the POC's clean-writer green-light
(task 12).

## 1. lance-graph-contract — SoA primary-citizen "speed lane"

- Expose the SoA container as a **first-class speed-lane** consumers can target.
- **Hard constraint:** `lance-graph-contract` is **ZERO-DEP**. So this is a
  zero-dep **trait** (e.g. `SoaSpeedLane` / `SoaContainerContract`) describing the
  container surface — **NOT** the concrete `ndarray::hpc::soa::SoaContainerHeader`
  type. Depending on ndarray would break the zero-dep invariant; mirroring the
  type would be the duplication-hallucination the surreal specs forbid.
- ndarray's `SoaContainerHeader` *implements* the contract trait; consumers
  depend on the trait. The concrete type stays the single source in ndarray.

## 2. lance-graph-ontology (new crate) — OGIT / OWL / DOLCE inheritance of CAM codebooks

- The immutable, append-only, **O(1) version-sealed** mapping table: label /
  concept / CAM-codebook entry → pointer into an immutable arena, with
  DOLCE / OGIT subclass inheritance **precomputed** (transitive closure
  materialised; minimal-perfect-hash per version).
- Append **downward only** (DOLCE/OGIT-upper fixed) → stable pointers, cheap
  delta-versioning. Makes the commit-time DOLCE constraint check O(1).
- **O(1) classification ≠ O(1) computation** — declarative inheritance only;
  computational business logic (Odoo computed fields) is not a lookup.

## 3. lance-graph-callcenter (new crate) — ractor / surrealdb application

- The application layer: ractor edge (cold path, tick/harvest) + SurrealDB-on-Lance
  (via `surreal_container`) + the ontology — the OSINT / call-center use case.
- Consumes the POC: `surreal_container` (write/read/fold) + `lance-graph-ontology`
  (DOLCE/CAM classification) + the contract speed-lane trait.
- Home for the `callcenter/role_keys.rs` domain role catalogue (per CLAUDE.md
  § VSA-switchboard three-layer architecture: Layer-2 domain catalogue).

## Ordering

POC (tasks 01–12) → contract speed-lane trait (1) → ontology (2) → callcenter app (3).
(1) can start now — `SoaContainer` is pinned (ndarray `b5d6b206`). (2)/(3) after the
POC writer is green (task 12). Each, when built, follows the same disjoint-file +
savant-meta-review discipline as the 12 POC tasks.
