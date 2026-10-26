# Softmax reciprocal fixture replay (D-TFLP 6.5, lab)

This crate replays committed int8 SOFTMAX fixtures without TensorFlow Lite Micro.
It shows that the fixtures, and the integer recipe they were recorded against, still agree.
It runs under plain `cargo test`. It is not a workspace member and has no production consumer.

## What is tested

The fixtures live in `../../source-behavior-v1/fixtures65/`. A SHA-256 manifest
(`manifest65.json`) covers every file.

| file | rows | content |
|---|---|---|
| `softmax_tables65.json` | 14 tables | prepared source weights per calibration: admitted flag, `A = exp_on_negative_values`, `B = Rescale<12>`, each with its SHA-256 |
| `witness65_full.jsonl` | 529 | every kernel-confirmed witness row. In each, the TFLM reciprocal and the correctly rounded one give different output bytes |
| `witness65_reduced.jsonl` | 27 | the regression subset (selection rule in each row's `why`) |
| `equivalence65.jsonl` | 7 | calibration 9 rows where both reciprocals give the kernel's bytes |
| `unreachable65.jsonl` | 10 | arithmetic candidates proved not to occur in any admitted row |

The runner works through three stages, in this order:
1. manifest integrity: any mismatch exits with code 2;
2. each table's digest;
3. every fixture row.

For each witness row it checks the following:
- the claimed calibration identity is the one the table was prepared for;
- the row maximum is code 127;
- every code is admitted and every count is positive;
- the recomputed sum equals the stored `S`;
- `S` lies in the supported domain;
- the source-reciprocal byte equals the stored kernel byte;
- the exact-reciprocal byte equals the stored alternative byte, which differs from the source byte.

All arithmetic is checked: an overflow the source would leave undefined becomes a named `Refusal`.

## Where the oracle came from

TFLM revision `5f64e4ad` (its `tensorflow/` tree is identical to `8f6596d`), reference int8 SOFTMAX
(`kernels/internal/reference/softmax.h`), built with g++ 13.3.0 on x86-64.
The source digest is sha256(`common.h` + `fixedpoint.h` + `reference/softmax.h`), equal to
`0566576b4fb0be4df914ce36e369de295d22e227d661c13a0cec17f5d7f2eb3c`.

The rows were produced by the certification gate (`../../source-behavior-v1/softcert65.cc`) and
confirmed by the real TFLM kernel (`softmax_oracle.cc`) before `fixtures65.py` wrote them.
Regenerating them with the current gate gives byte-identical files.

## Admitted domain

- int8 input, output scale 1/256, zero point −128;
- `x_integer_digits = 12`;
- row sums `S ∈ [2^19, 2^28)`.

The 14 calibrations are the known set of `softmax63.py`.

## How to run

```sh
cargo test --release                                                 # 11 tests, < 1 s
cargo run --release -- ../../source-behavior-v1/fixtures65 --full    # all 529 + 27 + 7 + 10
cargo run --release -- ../../source-behavior-v1/fixtures65 --reciprocal exact --full
```

The last command is the negative control. It must fail all 556 witness rows (529 full and 27 reduced) while
the 7 equivalence rows still pass. Measured: about 90 ms for the full tier.

## What PASS proves

- The committed fixture bytes are the ones the manifest names.
- On these 14 calibrations and these rows, the integer recipe in `src/lib.rs` reproduces the bytes the TFLM kernel produced.
- The correctly rounded reciprocal is *not* a drop-in replacement on the 13 calibrations that have witnesses.
- The tamper cases below are rejected, each for its own reason.

## What PASS does not prove

- It says nothing about calibrations outside the 14, rows outside the fixtures, other TFLM revisions, other compilers or other backends.
- It does not re-run TFLM. Kernel agreement was established when the fixtures were made; the replay checks the recorded result.
- It is not a softmax implementation, a Codec8 consumer or a production path.

## Tamper detection (`tests/tamper.rs`)

| tamper | expected result |
|---|---|
| changed byte, manifest not re-stamped | exit 2, `REFUSE integrity` |
| changed source byte, re-stamped | `source byte … != stored` |
| changed table weight, re-stamped | `table digest` |
| witness claims another calibration (scale) | `calibration identity` |
| witness points at another calibration's table | `table_sha256` |
| row without its maximum (code 127) | `no mandatory maximum` |
| stored `S` off by one | `!= stored S` |
| a count of −1 | `NonPositiveMultiplicity` |
| 600 maxima (`S > 2^28`) | `UnsupportedDomain` |
| exact reciprocal | 556 failures, 7 equivalence rows still pass |
