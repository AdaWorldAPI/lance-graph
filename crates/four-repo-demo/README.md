# four-repo-demo

Minimal end-to-end integration demo for the lance-graph workspace.

Cross-reference: `.claude/plans/integration-plan.md` — the full integration
story for the four-repo stack (lance-graph + surrealdb + sea-orm + ndarray).

---

## What this crate demonstrates

Three of the four glue crates from the integration plan are exercised together
in a single runnable example and integration test:

| Glue | Crate / module | What is shown |
|------|----------------|---------------|
| **Glue #4** | `cognitive-shader-actor` | `SumShader` wrapped as a `ractor::Actor` via `CognitiveShaderActor<SumShader>` |
| **IR contract** | `lance-graph-contract::ir` | 3-node `OperatorTree` (RangeScan→Filter→CognitiveApply) built and inspected |
| **Actor contract** | `lance-graph-contract::actor` | `SupervisableShader` implemented on `SumShader` with `Arc<Mutex<i64>>` state |

### `SumShader` (real implementation)

`SumShader` is not a stub. It accepts `RecordBatch` payloads with a single
`value: Int64` column and maintains a running cumulative sum using
`Arc<Mutex<i64>>` interior mutability. Each `apply` call returns a
single-row `RecordBatch` containing the new running total.

```text
Apply(value=5)  → RecordBatch { value: [5] }    (running sum = 5)
Apply(value=7)  → RecordBatch { value: [12] }   (running sum = 12)
Drain           → actor stops cleanly
```

### Planner IR demo

`planner::build_demo_tree` constructs a 3-node operator tree that mirrors a
real query plan shape:

```text
CognitiveApply  [Cognitive engine,   ~1 row]
└── Filter      [LanceGraph engine,  ~500 rows]
    └── RangeScan [TiKV engine,      ~10 000 rows]
```

`OperatorTree::total_estimated_cardinality()` sums the estimates to
`10 501 rows`.

---

## Running the demo

```bash
cargo run --manifest-path crates/four-repo-demo/Cargo.toml --example run_demo
```

Expected output:

```
=== Cognitive Shader Actor Demo ===
After sending value=5  → running sum = 5
After sending value=7  → running sum = 12
Actor drained cleanly.

=== Demo Operator Tree ===
CognitiveApply [sum_shader] engine=Cognitive cardinality=1 rows
  Filter [filter_recent] engine=LanceGraph cardinality=500 rows
    RangeScan [scan_events] engine=Tikv cardinality=10000 rows
Total estimated cardinality: 10501 rows

=== Summary ===
Running sum after 5+7 = 12
Operator tree node count = 3
Total estimated cardinality = 10501 rows
```

## Running the tests

```bash
cargo test --manifest-path crates/four-repo-demo/Cargo.toml
```

---

## What is NOT demonstrated (and why)

### Glue #1 — `surrealdb-ractor` (live query → actor mailbox)

Described in integration-plan.md §1 (Contracts table) and cross-referenced
in §10. The crate lives in `AdaWorldAPI/surrealdb` and bridges SurrealDB
change feeds into `ractor` mailboxes. It requires a running SurrealDB
instance and the `surrealdb-ractor` crate to be wired in — neither is
available in the lance-graph standalone build.

### Glue #2 — `lance-graph-tikv-provider` (TiKV range scan → Arrow)

Described in integration-plan.md §5. The provider crate is scaffolded at
`crates/lance-graph-tikv-provider/` but requires a live TiKV cluster
(`make tikv-up`) and `tikv-client` which has native library dependencies.
This demo uses the `ir::EngineHint::Tikv` tag to represent where a
`RangeScan` would execute, but does not perform an actual TiKV scan.

### Glue #3 — `sea-orm-ractor` (Entity PK → actor process registry)

Described in integration-plan.md §10 cross-reference. The crate lives in
`AdaWorldAPI/sea-orm` and requires a running Postgres instance plus the
`sea-orm-ractor` crate. Not available in the lance-graph standalone build.

---

## Module layout

| Module | Contents |
|--------|----------|
| `src/cognitive.rs` | `SumShader` — `SupervisableShader` impl with running sum |
| `src/planner.rs` | `build_demo_tree` + `print_plan` + `node_count` |
| `examples/run_demo.rs` | Runnable `main()` exercising both parts end-to-end |
| `tests/end_to_end.rs` | Assertions for sum=5/12, tree 3 nodes, cardinality 10501 |

---

Cross-reference: `.claude/plans/integration-plan.md` — full integration story,
sprint sequence, and links to companion plans in surrealdb / sea-orm / ndarray repos.
