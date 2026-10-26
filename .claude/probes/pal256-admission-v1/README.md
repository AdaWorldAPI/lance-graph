# pal256-admission-v1 (D-TFLP 6.5, lab)

These lab crates exist for the 8-bit code-space question: which consumer may interpret which bytes.
No production file in ndarray or lance-graph is changed, and neither crate is a workspace member.

| path | role |
|---|---|
| `admit/` | P0 semantic admission. `bind(artifact, expect)` admits 8-bit bytes for one consumer codec, or refuses with a named `Refusal`, before any value is interpreted |
| `replay/` | TFLM-free replay of the softmax reciprocal fixtures. It lands separately with the softmax certification work; see its own README |
| `matrix65.json` | 6 producers × 6 consumers; only the diagonal binds |
| `disable65_p0.txt` | disable runs: each admission check replaced by `if false && …`, and the test that must fail |
| `HANDOFF.md` | how this meets DT64, Quack and D-NAM, and what is deliberately not built |

## The defect it addresses

Amendment 6.4 finding F4 / N8.1: `Codec8<I8, Sym, TiesAway, NaNAtMin>` decodes −128 as NaN. That is
correct, because it is that codec's declared policy. The defect is that a payload is a bare `&[i8]`. The
producer's code space lives only in the producer's Rust type, so any consumer type can be applied to
the stored bytes. An asymmetric weight of −128 then silently becomes NaN.

The decoder is not changed. Admission happens once, at preparation time.

## Contract (`admit/src/lib.rs`)

- **`Receipt`**, carried by every artifact. It holds:
  - the codec signature;
  - the scale and zero-point bits;
  - the operator;
  - the codebook generation;
  - an optional source identity (revision, backend);
  - the length and the payload SHA-256.
- **`Expect<T, S, R, N, C>`**: what the consumer requires.
- **`bind`**: checks length, integrity, storage, code space, NaN policy, rounding, scale, zero point,
  operator, generation and source. It then range-scans the payload. Each mismatch has its own `Refusal`.
- **`Bound::decode`**: decodes with no further checks.

`CodecSignature` is derived from `Codec8`'s own sealed traits:
- the raw stored bytes for −128 and 0;
- `MIN` and `MAX`;
- `NAN_CODE`;
- the rounding of four half-ties.

There is no name table and no registry.

**Identity claim: TESTED_COMPATIBLE_FOR_PINNED_CODEC_SET.** The pinned set is every instantiation the
sealed traits admit on ndarray master `af4910a1`, 21 types in all. A test checks that their signatures
are pairwise distinct. A codec impl added to ndarray later is outside the claim until it is added to that test.

## Tests (`admit/tests/admission.rs`, 17)

| group | cases |
|---|---|
| positive | asymmetric −128 is an ordinary value (P1/P2); SymNaN −128 is the declared sentinel (P3); Sym −127..127 (P4); a TFLM symmetric weight still rejects −128 (P5); signed and biased indexing agree on every code (P6) |
| negative | F4 after the fix (N1); SymNaN→Asym (N2); biased U8→raw signed (N3); wrong generation (N4); rounding (N5); scale / zero point (N6); same bytes with an incompatible receipt (N7); one corrupted byte (N8); missing or mismatched source (N9); truncated payload (N10) |
| baseline | `f4_before_wrong_interpretation_accepted`: without admission, the wrong interpretation goes through |
| scope | `signatures_are_pairwise_distinct_on_the_pinned_codec_set` |

In every negative control, the payload bytes are identical in the accepted arm and the refused arm.
Only the consumer or a single receipt field changes.

```sh
# ndarray must sit five levels up (…/ndarray next to the lance-graph checkout)
cd admit && cargo test --release && cargo run --release --example matrix ../matrix65.json
```

Measured on ndarray master `af4910a1`: 17 passed, and `matrix65.json` regenerates byte-identical.
Collapsing the rounding signature to one half-tie makes the scope test fail
(`TiesEven` and `Floor` collide), so the test is not vacuous.
