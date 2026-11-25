# Vector fork commits awaiting push access

These two commits exist only as patches here. This session has no push access to `AdaWorldAPI/vector`. Apply them to the fork's `master` at c879bd6:

```bash
git am .claude/handovers/vector-fork/*.patch
```

- **`0001` adds `lib/lance-graph-sink`.** It maps `vrl::value::Value` onto `lance_graph_vector::Event`. It predates #2205:
  - it uses `Cell::Number`, which no longer exists;
  - it should call `VectorRoute::receive_values(&[&Value])` directly, once per batch of `LogEvent::value()`.

  Rework it before pushing. Most of it becomes unnecessary, because the route now reads `vrl::value::Value` itself.
- **`0002` pins `rust-toolchain.toml` to 1.99.0 and raises `rust-version` to 1.99.** Only `vector-core` and `enrichment` were checked on 1.99. The full `vector` binary was not, for lack of disk. Full binary CI is mandatory before merge (operator, 2026-11-25).
