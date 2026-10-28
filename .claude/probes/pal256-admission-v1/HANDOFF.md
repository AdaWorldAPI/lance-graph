# Handoff: 8-bit admission ↔ DT64 / Quack / D-NAM

**Status:** lab evidence plus integration notes. This document proposes no runtime, store, ABI or
registry, and it does not duplicate the DT64 or D-NAM documents. It says where the admission contract
in `admit/` would attach, and what must exist before that can happen.

## What exists

- **ndarray `Codec8<T, S, R, N>`** (`src/hpc/codec8.rs`, on master). Codec identity exists only in the
  type parameters; a payload is a bare `&[i8]`. There is no production consumer on main, only lab
  probes and `examples/codec8_asm_probe.rs`.
- **`admit/` (this directory).** It provides the receipt, the expectation and `bind`, with 17 tests and a
  6×6 matrix in which only the diagonal binds. Identity is derived from the sealed traits, and the
  claim is scoped to the 21 pinned instantiations (TESTED_COMPATIBLE_FOR_PINNED_CODEC_SET).
- **Softmax certification, #1800.** It is the same distinction one level up: a numerical cache key
  `(mult, shift)` against a semantic receipt `(scale, zp, β, operator, source)`. Two calibrations can share a
  table and still need separate receipts.

## Where it would attach (planned, not built)

| owner | what it needs from this | status there |
|---|---|---|
| D-NAM capstone §2 "Recipe receipt" (`.claude/plans/d-nam-palette-native-lut-capstone-v1.md`) | the field set of `Receipt` (codec signature, scale/zp, operator, generation, source, payload digest) as the 8-bit half of a recipe receipt | PROPOSAL; [EXO-152](https://linear.app/exo-red/issue/EXO-152) |
| PZL-4 (same plan): "two same-shape layers with different weights/calibration must refuse wrong table" | `bind` refusing N4 (generation), N6 (scale/zp) and N7 (same bytes, different receipt), and #1800's `table_identity_mismatch` | waits for owner-issued generation identity |
| D-NFR S5, **binding store, MISSING** (`.claude/plans/d-native-repr-contract-v1.md`) | the record a binding would carry. `admit` is not that store and must not become a second one | MISSING; D-NFR-1 is the probe |
| DT64-QB (#1799, with stages #1794–#1798) | DT64 supplies source-attested typed semantic bindings and borrowed lanes. For an 8-bit lane, the receipt is what makes "which code space" part of that binding instead of part of a Rust type that disappears at storage | docs only |
| Quack / RelAlg / R2IL / mask-risc | none directly. They execute relational algebra over already-bound lanes. A matching receipt is not permission to reuse an escaped Keep mask (#1792 merged; escaped authority still open) | unchanged |

ndarray owns numerics and D-NAM owns numerical methods; ONNX/TFLite stay leaf sources, and model
import is not a prerequisite for any of this.

## Open, with owners

1. **Generation identity:** who issues `codebook_generation` (EXO-152 / PZL-4). Until then the field is
   compared but not minted by anyone.
2. **Binding store:** where a bound receipt lives and when it is invalidated (D-NFR S5 × S6). This is not
   decided here.
3. **Contract placement:** whether the `Receipt` fields enter a contract type at all. That is an owner
   decision; this lab crate makes no claim.
4. **Accumulator wrapping:** biased VPDPBUSD/TDPBUSD wrap by design. A five-line `wrapping_*` proposal is
   on the research branch (`int8-semantics-v1/accum_wrapping65.diff`). It makes debug match release but
   removes the out-of-range signal. It is not in any landing PR; it needs an owner decision on
   overflow semantics first.
5. **Outside the pinned codec set:** a new `Rounding`, `NanPolicy`, `CodeSpace` or `Stores` impl needs the
   scope test extended before the identity claim covers it.

## Not claimed

- No source-system compatibility beyond the tested codec set and the softmax calibrations in #1800.
- No source permissions, no writes and no semantic identity from a digest.
- No change to Quack/DR architecture, Codec8 semantics or the decoder's handling of −128.
