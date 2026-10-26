# pal256-admission-v1 — D-TFLP Amendment 6.5 lab probes

Lab only. No production file in ndarray or lance-graph is changed.

| path | role |
|---|---|
| `admit/` | P0 binding contract. `bind(artifact, expect)` admits 8-bit bytes for one consumer codec or refuses with a named `Refusal`, before any value is interpreted. Codec identity is a `CodecSignature` derived from `Codec8`'s own sealed traits (code range, NaN code, raw stored bytes, observed rounding) — no name table. `tests/admission.rs`: positive cases P1–P6 and negative controls N1–N10, each with an identical payload in the accepted and the refused arm. `examples/matrix.rs` → `matrix65.json`: 6 producers × 6 consumers. |

```sh
cd admit && cargo test --offline && cargo run --offline --example matrix ../matrix65.json
```
