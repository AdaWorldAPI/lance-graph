# Refactor Notes — Sleeping Beauty Capabilities

> Items that compile and pass tests but are not yet wired to real data.
> Each has a clear activation path.

## 1. `with-engine` feature (EMPTY SHELL)

**Status**: Feature flag exists, thinking-engine is an optional dep, but zero
code runs behind it. The shader does bgz17 cascade only — no 4096² MatVec.

**Activation path**:
- Add `engine_cycle.rs` behind `#[cfg(feature = "with-engine")]`
- Import `ThinkingEngine` from thinking-engine
- In `ShaderDriver::run()`, after bgz17 cascade (step 3), insert:
  ```
  if cfg!(feature = "with-engine") {
      let engine_energy = engine.cycle(&top_k_indices);
      // XOR-fold the engine's top-k into cycle_fingerprint
  }
  ```
- The MatVec cycle produces a proper interference pattern instead of
  the current XOR-fold-of-content-fingerprints approximation.

**Risk**: thinking-engine depends on `bgz-tensor` + `highheelbgz` +
baked distance tables. Those 256×256 u8 tables must be available at
runtime. The shader currently works without any baked data.

---

## 2. Classification Distance Calibration (HAND-PICKED)

**Status**: `classification_distance()` uses 6 hand-picked emotion
archetypes in QPL space. The coordinates are plausible but not
calibrated against real convergence patterns.

**Activation path**:
- Run the thinking-engine on 100+ labeled texts (emotion corpus)
- Record `Qualia17D` for each
- Fit archetype centroids via k-means on the 17D space
- Replace the hardcoded `ARCHETYPES` array with calibrated centroids
- Optionally: expand beyond 6 basic emotions (Plutchik wheel = 8+)

**Risk**: Without calibration, "steelwind" (novel) vs "fear" (named)
boundary is approximate. The test verifies separation exists but
doesn't guarantee the boundary is meaningful.

---

## 3. Neural-Debug Health Endpoint (STUB)

**Status**: `/v1/shader/health` returns `neural_debug: null`. The
`WireNeuralDiag` struct exists but no scanner is wired.

**Activation path**:
- Add `neural-debug` as optional dep behind `serve` feature
- In `health_handler()`, call `neural_debug::scanner::scan_stack()`
- Map `StackDiagnosis` → `WireNeuralDiag`
- This gives live function health (dead/stub/NaN counts) via REST

**Risk**: `scan_stack()` walks the filesystem — could be slow on
first call. Cache the result and refresh on `/v1/shader/health?refresh=true`.

---

## 4. ThinkingStyle Unification (3 COPIES)

**Status**: ThinkingStyle exists in 3 places with 3 different enum sizes:
- `lance-graph-contract::thinking::ThinkingStyle` — 36 variants
- `thinking-engine::cognitive_stack::ThinkingStyle` — 12 variants
- `cognitive-shader-driver::UNIFIED_STYLES` — 12-element const array

**Activation path**:
- Adopt contract's 36-variant enum as canonical
- Map contract ordinals 0..35 → the existing 12-variant behavior
  (many contract styles cluster into the same shader parameters)
- The extra 24 styles are finer gradations — same layer_mask/combine/contra
  with different density_target and resonance_threshold

**Risk**: The planner uses contract's 36-variant enum. Changing the
driver to accept 36 would mean 24 new entries in UNIFIED_STYLES.
Better: keep the driver at 12 coarse styles, map 36→12 at the bridge.

---

## 5. A2A Blackboard Sweep (NOT IMPLEMENTED)

**Status**: BindSpace has `cycle_fingerprint` columns per row.
Multiple agents could write their cycle_fingerprints and sweep each
other's via Hamming distance. The column layout supports this but
no sweep function exists.

**Activation path**:
```rust
pub fn sweep_nearest(bs: &BindSpace, query: &[u64; 256], k: usize) -> Vec<(u32, u32)> {
    // For each row, compute hamming(query, bs.fingerprints.cycle_row(row))
    // Return top-k (row, hamming_distance) pairs
}
```
- Uses `ndarray::hpc::bitwise::hamming_distance_raw()`
- O(N × 256 words) per sweep — needs SIMD for N > 1000

**Risk**: No concurrency control. Two agents writing to the same
BindSpace need either a Mutex (slow) or a lock-free column append.

---

## 6. 5D Stream Cycle Loop (P2 — DESIGN ONLY)

**Status**: Documented in knowledge doc as P2 item 9. Not implemented.

**Activation path**:
```
for each cycle:
    1. topic     = dispatch(topic_column, CAUSES layer)
    2. angle     = dispatch(angle_column, SUPPORTS layer)
    3. causality = dispatch(edge_column, all layers)
    4. qualia    = compute Qualia17D from convergence snapshot
    5. exact     = dispatch(content_column, focused style)
    6. emit cycle_fingerprint = XOR-fold(topic, angle, causality, exact)
```
Each sub-dispatch uses a different BindSpace column plane and style.
The full cycle produces a richer fingerprint than the current single-pass.

---

## 7. BindSpace Persistence (EPHEMERAL)

**Status**: BindSpace is in-memory only. `EmitMode::Persist` sets
`persisted_row` in the crystal but doesn't write to disk.

**Activation path**:
- Use Lance (columnar storage) to persist BindSpace columns
- Each column maps to a Lance column: fingerprints → fixed-size-binary(2048),
  edges → uint64, qualia → fixed-size-list(float32, 18), meta → uint32
- Versioned snapshots via Lance's MVCC

**Risk**: Lance dependency is heavy (arrow, parquet). Keep behind
a feature flag. The in-memory BindSpace is fine for debugging.
