# PhiSpiral256 SoA Cross-System Integration Plan — lance-graph

## Goal

Define how PhiSpiral256 integrates with lance-graph as a separate SoA lane for orthogonal local residual location.

PhiSpiral256 must sit beside existing lanes without mixing meanings:

```text
CAM_PQ
  meaning / semantic basin lane

PolarQuant
  magnitude / similarity lane

PhiSpiral256
  orthogonal local residual location lane

BGZ17
  golden offset/stride recoverable sampling skeleton

Fisher-z cosine
  optional scorer/gate after candidate ranking
```

## Cross-system role

`ndarray` owns:

```text
PhiSpiral256 center generation
distance / neighbor tables
atom16 pack/unpack
local encode/decode kernels
Fisher-z gate if used numerically
```

`lance-graph` owns:

```text
SoA schema placement
leaf packet graph relationships
HHTL / CAM_PQ / PolarQuant / PhiSpiral lane composition
candidate routing tables
provenance and certificates
integration fixtures
```

## Schema concept

Use a leaf-location sidecar table first. Do not force PhiSpiral256 into unrelated graph structures.

```text
leaf_planetarium_atoms
  leaf_id: UInt64
  atom_index: UInt8
  atom16: UInt16
  confidence_q: UInt8
  cam_pq_id: UInt16?
  polarquant_id: UInt8?
  replay_ref: UInt64?
  certificate_id: Utf8?
```

Optional expanded debug view:

```text
leaf_planetarium_atoms_debug
  leaf_id
  phi_spiral_id
  mag4
  bgz_offset_family
  bgz_stride_family
  confidence_q
  cam_pq_id
  polarquant_id
  plane
```

## Graph model

```text
Leaf
  -> HAS_PLANETARIUM_ATOM -> PhiSpiralAtom
  -> HAS_MEANING_CODE      -> CamPqCode
  -> HAS_MAGNITUDE_CODE    -> PolarQuantCode
  -> HAS_RECOVERY_SCHEDULE -> BgzRecoverySchedule
  -> HAS_CERTIFICATE       -> Certificate
```

Keep atom nodes optional. For performance, atoms should primarily live in SoA tables.

## Route key

Callers may compose a route/candidate key:

```rust
pub struct LeafPlanetariumRouteKey {
    pub cam_pq_id: u16,
    pub phi_spiral_id: u8,
    pub mag4: u8,
    pub bgz_offset_family: u8,
    pub bgz_stride_family: u8,
    pub plane: u8,
}
```

This key is for routing or candidate lookup only. It does not redefine CAM_PQ, PolarQuant, BGZ17, or Fisher-z.

## Candidate table

```text
leaf_planetarium_candidates
  route_key_hash: UInt64
  candidate_basin_id: UInt64
  priority_q: UInt16
  evidence_count: UInt32
  last_calibrated_epoch: UInt64
```

This enables:

```text
route key -> candidate basin set
```

without forcing exact replay.

## Calibration baselines

The integration must compare against:

```text
Mag4 only
BGZ17 L1 / weighted L1
BGZ17 sign agreement
PolarQuant only
PhiSpiral256 Euclidean
PhiSpiral256 Poincare/Mobius
PhiSpiral256 + Fisher-z gate
Hybrid: CAM_PQ + PolarQuant + PhiSpiral256 + BGZ17 schedule
```

## Integration metrics

```text
candidate fanout reduction
leaf replay reduction
route-key collision rate
wrong-basin high-confidence rate
SoA scan throughput
atom table byte size
candidate table byte size
query latency impact
calibration stability across datasets
```

## Fixtures

Start with synthetic fixtures:

```text
single meaning axis
orthogonal residual directions
controlled missing locations
known candidate basins
known noisy / ambiguous cases
```

Then add real-ish fixtures:

```text
CAM_PQ meaning codes
BGZ17 residual fingerprints
PolarQuant magnitude bands
multi-atom leaf packets
```

## Query examples

```sql
SELECT leaf_id, atom16, confidence_q
FROM leaf_planetarium_atoms
WHERE confidence_q >= 200;
```

```sql
SELECT candidate_basin_id
FROM leaf_planetarium_candidates
WHERE route_key_hash = ?
ORDER BY priority_q DESC
LIMIT 8;
```

## Distortion and safety gates

Reject or widen candidates when:

```text
spiral bin occupancy is too skewed
route-key collision rate is too high
Fisher-z margin is too small, if gate is enabled
wrong-high-confidence rate rises above threshold
multi-atom packets exceed configured K too often
```

## Implementation phases

```text
Phase 0: plan docs and term separation
Phase 1: ndarray PhiSpiral256 kernel + tests
Phase 2: Lance/Arrow SoA table shape
Phase 3: synthetic fixture and calibration report
Phase 4: integrate with CAM_PQ / PolarQuant / BGZ17 lanes
Phase 5: candidate table and route-key lookup
Phase 6: real dataset replay reduction benchmark
```

## Acceptance criteria

- PhiSpiral256 appears as its own lane in SoA schemas.
- CAM_PQ, PolarQuant, BGZ17, Palette256, Fisher-z terminology remains separated.
- Synthetic fixture proves better orthogonal location recall than Mag4-only.
- Hybrid route key reduces candidate fanout or exact replay in at least one fixture.
- Candidate table lookups are deterministic and explainable.

## Wall sentence

```text
CAM_PQ says what it means, PolarQuant says how much, PhiSpiral256 says where the unexplained part lives, and BGZ17 remembers how to recover it.
```
