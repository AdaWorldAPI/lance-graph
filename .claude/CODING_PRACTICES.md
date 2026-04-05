# Coding Practices — Patterns for the Thinking Engine Stack

**Source:** Extracted from EmbedAnything (StarlightSearch, Apache 2.0, 1.1K stars),
validated against our architecture constraints. These are patterns, not dependencies.

**Audience:** Claude Code agents working on lance-graph / ndarray / ada-rs.

---

## Checklist for New Modules

Before merging a new module, verify:

```
[ ] Does it auto-detect model type, or hardcode model names?
[ ] Does commit() use the sink pattern, or return-and-pray?
[ ] Is there a builder, or does the caller assemble raw structs?
[ ] Are heavy deps behind feature gates?
[ ] Does it work with BOTH u8 and i8 tables (or is it type-generic)?
[ ] Are per-role scale factors preserved, or does it assume uniform range?
[ ] Is the boundary between calibration and runtime clean?
[ ] Does it avoid forward passes at runtime? (codebook lookup only)
```

## Anti-Patterns (what NOT to copy)

```
1. 48KB lib.rs → Keep lib.rs as module declarations only.
2. Clone-heavy structs → BusDto references codebook indices, not cloned content.
3. Python-first API design → Rust-first. Python binding later.
4. Forward pass at every query → Codebook lookup. Never go back.
5. f32 everywhere → Precision-aware types (BF16 calibration, i8 runtime).
```

## Implemented Patterns

```
Pattern                  Module                  Status
───────                  ──────                  ──────
Auto-detect config.json  auto_detect.rs          6 tests, routes by architecture
Builder (fluent API)     builder.rs              7 tests, Lens/TableType/Pooling/Sinks
Pooling strategies       pooling.rs              6 tests, ArgMax/Mean/TopK/Weighted
Commit sinks (adapter)   builder.rs on_commit()  callback-based, multiple sinks
Tensor bridge            tensor_bridge.rs        7 tests, F32/I8/U8/Tensor enum
Semantic chunking        semantic_chunker.rs     4 tests, convergence-jump = boundary

Feature gates:
  default      = runtime only (codebook lookup + MatVec)
  tokenizer    = HuggingFace tokenizers 0.22
  calibration  = candle 0.9 + hf-hub (forward pass + training)
```
