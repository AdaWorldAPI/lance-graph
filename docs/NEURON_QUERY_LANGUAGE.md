# 6D SPO Query Language — Cypher/GQL Extension for NeuronPrint

> **Date**: 2026-03-31
> **Status**: Design — ready to implement when token budget refreshes
> **Depends on**: Cypher parser (done), DataFusion planner (done), NeuronPrint (done), hydrate partitions (done)

---

## The Idea

Extend lance-graph's existing Cypher/GQL parser to query the 6D NeuronPrint
structure natively. DataFusion executes the query over partitioned Lance datasets.
The 6 tensor roles become first-class graph relationships.

```
Today (string SPO):
  MATCH (s:Entity)-[r:KNOWS]->(o:Entity) RETURN s, r, o

Tomorrow (6D NeuronPrint SPO):
  MATCH (n:Neuron)-[:Q]->(target)
  WHERE n.layer = 15 AND distance(n.q, $query) < 100
  RETURN n.feature, n.v AS retrieval, n.trace.confidence AS conf
```

---

## Query Language Extension

### Node Type: Neuron

```cypher
-- A neuron is identified by (layer, feature)
MATCH (n:Neuron {layer: 15, feature: 42})
RETURN n.q, n.k, n.v, n.gate, n.up, n.down
```

Each property (q, k, v, gate, up, down) is a 17-dim Base17 vector.

### Relationship Types: The 6 Roles

```cypher
-- Attention: what does layer 15 attend to?
MATCH (n:Neuron {layer: 15})-[:ATTENDS]->(m:Neuron)
WHERE l1(n.q, m.k) < 50
RETURN n.feature, m.feature, m.v AS retrieved

-- Gating: which neurons fire at layer 10?
MATCH (n:Neuron {layer: 10})
WHERE magnitude(n.gate) > 0.8
RETURN n.feature, n.trace.frequency

-- MLP path: what does layer 5 amplify?
MATCH (n:Neuron {layer: 5})
WHERE magnitude(n.up) > magnitude(n.down) * 2
RETURN n.feature AS amplified, n.trace.confidence
```

### Role Masks (Pearl 2³ → Pearl 2⁶)

```cypher
-- Probe only Q+K (attention query)
MATCH (n:Neuron) USING ROLES(q, k)
WHERE l1(n.q, $probe) < 100
RETURN n.k, n.v

-- Probe only Gate+Up+Down (reasoning query)
MATCH (n:Neuron) USING ROLES(gate, up, down)
WHERE n.trace.expectation > 0.7
RETURN n.feature, n.layer, n.trace

-- Full 6D probe
MATCH (n:Neuron) USING ROLES(*)
WHERE bundle_distance(n, $query) < 200
RETURN n
```

### Cross-Layer Queries (Residual Stream Tracing)

```cypher
-- Trace a concept through the network:
-- which neurons activate at each layer for this query?
MATCH path = (n:Neuron)-[:ATTENDS*]->(m:Neuron)
WHERE n.layer = 0 AND m.layer = 27
  AND l1(n.q, $concept) < 50
RETURN nodes(path), [x IN nodes(path) | x.trace.frequency] AS activations
```

### NARS-Enriched Queries

```cypher
-- Find neurons with high confidence AND high frequency
-- (strong, reliable features)
MATCH (n:Neuron)
WHERE n.trace.frequency > 0.8 AND n.trace.confidence > 0.7
RETURN n.layer, n.feature, n.trace.expectation
ORDER BY n.trace.expectation DESC
LIMIT 100

-- Find contradictions: neurons where Q says one thing, Gate says another
MATCH (n:Neuron)
WHERE n.trace.attention > 0.8 AND n.trace.frequency < 0.2
RETURN n AS "attends but doesn't fire"

-- NARS revision across layers: combine evidence
MATCH (a:Neuron {layer: 10}), (b:Neuron {layer: 20})
WHERE a.feature = b.feature
RETURN a.feature,
       nars_revision(a.trace, b.trace) AS combined_truth
```

### Model Comparison (Diff)

```cypher
-- Compare Opus 4.5 vs 4.6: where do they diverge?
MATCH (a:Neuron:Opus45), (b:Neuron:Opus46)
WHERE a.layer = b.layer AND a.feature = b.feature
  AND l1(a.bundle, b.bundle) > 500
RETURN a.layer, a.feature,
       l1(a.q, b.q) AS q_diff,
       l1(a.gate, b.gate) AS gate_diff,
       CASE
         WHEN l1(a.q, b.q) > l1(a.gate, b.gate) THEN 'attention changed'
         ELSE 'gating changed'
       END AS change_type
ORDER BY l1(a.bundle, b.bundle) DESC
```

---

## DataFusion Execution Plan

The Cypher extension maps to DataFusion SQL over Lance datasets:

```
Cypher:
  MATCH (n:Neuron {layer: 15})-[:ATTENDS]->(m:Neuron)
  WHERE l1(n.q, m.k) < 50

DataFusion SQL:
  SELECT a.feature, b.feature, b.vector AS v_vector
  FROM weights a
  JOIN weights b ON l1_distance(a.vector, b.vector) < 50
  WHERE a.layer_idx = 15
    AND a.tensor_role = 0  -- Q
    AND b.tensor_role = 1  -- K

Lance execution:
  1. Partition prune: tensor_role=0 (Q) for a, tensor_role=1 (K) for b
  2. Layer filter: layer_idx=15 for a
  3. Vector search: RaBitQ ANN on a.vector against b.vector
  4. Join: matching features where L1 < 50
  5. Fetch: b's V-role vector for matched features
```

### UDFs Needed

```sql
-- L1 distance between two Base17 vectors (17 × i16)
CREATE FUNCTION l1(a FIXED_SIZE_LIST(FLOAT32, 17), b FIXED_SIZE_LIST(FLOAT32, 17))
  RETURNS UINT32 AS 'l1_distance';

-- Magnitude of a Base17 vector (sum of abs values, normalized)
CREATE FUNCTION magnitude(a FIXED_SIZE_LIST(FLOAT32, 17))
  RETURNS FLOAT32 AS 'base17_magnitude';

-- XOR bind two Base17 vectors
CREATE FUNCTION xor_bind(a FIXED_SIZE_LIST(FLOAT32, 17), b FIXED_SIZE_LIST(FLOAT32, 17))
  RETURNS FIXED_SIZE_LIST(FLOAT32, 17) AS 'base17_xor_bind';

-- Bundle (average) multiple vectors
CREATE AGGREGATE FUNCTION bundle(a FIXED_SIZE_LIST(FLOAT32, 17))
  RETURNS FIXED_SIZE_LIST(FLOAT32, 17) AS 'base17_bundle';

-- NeuronTrace from 6 role vectors
CREATE FUNCTION neuron_trace(q, k, v, gate, up, down)
  RETURNS STRUCT(frequency FLOAT32, confidence FLOAT32,
                 attention FLOAT32, coherence FLOAT32,
                 expectation FLOAT32) AS 'neuron_trace';

-- NARS revision of two truth values
CREATE FUNCTION nars_revision(a_f FLOAT32, a_c FLOAT32, b_f FLOAT32, b_c FLOAT32)
  RETURNS STRUCT(frequency FLOAT32, confidence FLOAT32) AS 'nars_revision';
```

---

## Implementation Plan

### Phase 1: UDFs (pure DataFusion, no Cypher changes)
Register `l1`, `magnitude`, `xor_bind`, `bundle`, `neuron_trace`, `nars_revision`
as DataFusion scalar/aggregate UDFs. Queryable via raw SQL immediately.

```sql
-- Already works after Phase 1:
SELECT tensor_name, row_idx,
       l1(vector, ARRAY[100,200,...]) AS dist
FROM weights
WHERE tensor_role = 0 AND layer_idx = 15
ORDER BY dist
LIMIT 10;
```

### Phase 2: Cypher Extension (parser + planner)
Add `Neuron` node type, role relationship types, `USING ROLES()` clause,
and `trace` property access to the existing nom-based Cypher parser.
Planner maps them to the Phase 1 UDFs.

### Phase 3: Cross-Layer Tracing
Add variable-length path patterns (`-[:ATTENDS*]->`) with layer progression
constraints. DataFusion recursive CTE or iterative join.

### Phase 4: Model Comparison
Multi-model queries with label selectors (`:Opus45`, `:Opus46`).
Multiple Lance datasets joined on (layer, feature).

---

## Why This Works

1. **DataFusion is already a dependency** (version 51, mandatory for Cypher stack)
2. **Lance datasets support partition pruning** (tensor_role, layer_idx columns)
3. **Cypher parser is nom-based and extensible** (44 tests, well-structured AST)
4. **Arrow RecordBatches carry the partition columns** (just added in hydrate.rs)
5. **UDFs are pure functions** (L1, magnitude, xor_bind — all deterministic, SIMD-friendly)
6. **NeuronTrace is derived, not stored** (computed at query time from 6 role vectors)

The query language makes the 6D NeuronPrint structure explorable without
writing Rust code. A researcher can interactively probe the model's knowledge
graph using familiar Cypher syntax.
