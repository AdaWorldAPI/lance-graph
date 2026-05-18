# lance-graph-tikv-provider

**Glue #2** in the four-crate integration plan: TiKV key ranges → Arrow `TableProvider`.

This crate allows lance-graph's Cypher executor (DataFusion-backed) to read
graph data stored in a distributed TiKV cluster as typed Arrow `RecordBatch`
output — with zero-copy handoff into the query engine.

## Role in the stack

```
lance-graph (Cypher executor)
        │
        │  Arc<dyn TableProvider>
        ▼
lance-graph-tikv-provider          ← THIS CRATE
  TikvNodeTableProvider / TikvEdgeTableProvider
        │
        │  key-range scan
        ▼
   TiKV (Raft + Percolator MVCC)
```

`lance-graph-catalog` declares the shape of each node and edge type (schema +
range key encoding). This crate consumes those shapes to translate DataFusion
scan requests — filters, projections, limits — into TiKV range bounds, then
decodes the raw bytes back into Arrow columns.

This replaces the JanusGraph-over-Cassandra pattern with TiKV's stronger
guarantees:
- **ACID** via Percolator 2PC instead of eventual consistency.
- **Native MVCC** snapshot reads aligned to the same HLC `u64` timestamp
  used by surrealdb-core and Lance dataset versions — **one clock, all
  storage targets** (plan §5 "Snapshot integration").

## What is stubbed vs implemented (as of Sprint 0)

| Item | Status |
|---|---|
| `TikvNodeTableProvider` struct + `TableProvider` impl | Stub — `scan()` is `unimplemented!()` |
| `TikvEdgeTableProvider` struct + `TableProvider` impl | Stub — `scan()` is `unimplemented!()` |
| `TikvScanExec` `ExecutionPlan` | Stub — `execute()` is `unimplemented!()` |
| `MvccProvider` + `TikvBackedProvider` marker impls | Wired — no logic needed |
| `Error` enum (`Tikv`, `Arrow`, `Decode`) | Defined — no conversion impls yet |
| Constructor `new()` + `with_snapshot()` | Wired — no client call yet |

The entire public API surface (struct fields, method signatures, trait bounds)
is locked in Sprint 0. Implementation lands in Sprint 1.

## Path to working implementation (Sprint 1, plan §7)

1. **Replace `Arc<()>` with `Arc<tikv_client::TransactionClient>`** in
   `TikvNodeTableProvider` and `TikvEdgeTableProvider`.  The `tikv-client`
   dep is already in `Cargo.toml` at `*`.

2. **Implement `TikvNodeTableProvider::scan`**:
   - Call `self.shape.range_from_filters(filters)` to derive key-range bounds.
   - Resolve snapshot: `self.snapshot_ts.unwrap_or_else(|| client.current_timestamp())`.
   - Construct `TikvScanExec::new(snapshot, key_range, schema, projection, limit)`.

3. **Implement `TikvScanExec::execute`**:
   - Open a `tikv_client::Snapshot` at the given HLC timestamp.
   - Call `snapshot.scan(key_range, limit)` to get an iterator of KV pairs.
   - Decode each value with the shape's column decoder into Arrow arrays.
   - Yield `RecordBatch` chunks to the DataFusion executor stream.

4. **Wire edge key encoding** in `TikvEdgeTableProvider::scan`:
   - Edge keys are `(src_node_id, edge_type, dst_node_id)` — outbound
     expansion is a single prefix range scan on `src_node_id`.

5. **Integration test** against local TiKV via `make tikv-up` (plan §7 Sprint 1).

## Cross-references

- Design intent: `integration-plan.md §5` (Glue #2, this repo)
- Contract traits: `crates/lance-graph-contract/src/provider.rs`
  (`TikvBackedProvider`, `MvccProvider`, `BackendId`)
- Sibling glue: `crates/cognitive-shader-actor/` (Glue #4, this repo)
- Consuming example: `integration-plan.md §8 Example 1`

## License

Apache-2.0
