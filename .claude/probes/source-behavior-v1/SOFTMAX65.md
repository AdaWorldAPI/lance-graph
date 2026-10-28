# Certified softmax preparation (D-TFLP 6.5, lab)

This is a gate that answers one question for one int8 SOFTMAX calibration: can the correctly rounded
reciprocal replace TFLM's `GetReciprocal` without changing any output byte on any supported row?
It is lab evidence. Nothing here is linked into a production crate.

## Four things, kept apart

| artifact | depends on | file | meaning |
|---|---|---|---|
| source-global certificate | the TFLM reciprocal algorithm only | `softcert65_global.txt` | domain `[2^19, 2^28)`; exact reference `round(2^62 / (S << clz S))` monotone in every exponent band; exponent `35 − clz S` checked on every S; `E_h = max \|source − exact\| = 6` for every band `h = 4..12`. Measured once over every S in 5.4 s |
| prepared table | `(mult, shift)` from `PreprocessSoftmaxScaling(β, scale)` | trace63 `softexp` output | admitted flag, `A = exp_on_negative_values`, `B = Rescale<12>` for d = −255..0 |
| semantic receipt | `(scale, zero point, β)`, operator, source revision and digest | gate line `U receipt` | what the decision is issued *for* |
| per-calibration decision | certificate + table | gate line `D` | `PROVED_OUTPUT_EQUIVALENT`, `REQUIRES_SOURCE_RECIPROCAL` or `REFUSE_UNKNOWN` |

**A table digest is a numerical cache key, not a semantic identity.** The table is a function of
`(mult, shift)`, i.e. of β·scale. The input zero point never enters it. Known calibrations 1
(0.05, β 2) and 2 (0.1, β 1) prepare the identical table. Held-out calibrations 1 (0.2, β 1) and
2 (1/128, β 1) reuse the tables of known calibrations 3 and 11. Calibration indices are 0-based throughout. The gate therefore prints the two separately:
- `N numeric_key mult shift`: equal keys mean identical weights, and a decision may be cached on it;
- `U receipt scale_bits zp beta_bits`: the identity the result belongs to. It is never derived from the key.

## Table ↔ calibration correspondence (verified in the gate)

The gate no longer trusts that the supplied table belongs to the claimed calibration.
- It recomputes `mult`, `shift`, `diff_min` and every `(admitted, A, B)` from the claimed `(scale, β)`,
  using the pinned TFLM primitives.
- On any difference it refuses with `table_identity_mismatch`.
- The Python driver keeps its own independent table check as a second line.

Controls in the run record (see "Run record" below):

| control | result |
|---|---|
| calibration *i+1*'s table under calibration *i*'s identity (`input:wrong_calibration`) | 13/14 `typed_refusal:table_identity_mismatch`. The 14th pair (1→2) has identical tables and is correctly accepted |
| same, gate check disabled (`input:wrong_calibration+skip_table_identity`) | 13/14 caught only by the driver's validation. This shows the gate's check is load-bearing |
| calibration 1's table under calibration 2's identity (`table_identity_controls`) | accepted, same decision, **equal numeric key, different receipt** |
| calibration 3's table under calibration 2's identity | `REFUSE_UNKNOWN table_identity_mismatch` |

## Results (run record)

Every decision is validated without the gate's code. The table is recomputed, the candidate set is
compared with the exhaustive superset reference, every witness row is rebuilt and run through the real
TFLM kernel, and gap claims are re-checked by a coin DP. All 22 calibrations validate.

| group | candidates | reachable | unreachable | decisions |
|---|---|---|---|---|
| 14 known | 601 | 529 | 72 | calibration 9 `PROVED_OUTPUT_EQUIVALENT`; 13 `REQUIRES_SOURCE_RECIPROCAL` |
| 8 held-out | 514 | 446 | 68 | 8 `REQUIRES_SOURCE_RECIPROCAL` |

For calibration 9, 300 random admitted rows give identical bytes from the kernel, the source recipe
and the exact recipe.

Mutations (14 known calibrations each; every effect is either refused or detected):

| mutation | affected | how |
|---|---|---|
| `e_minus2` | 2 | candidate set |
| `one_sided_window` | 12 | candidate set |
| `wrong_band` | 13 | candidate set + decision |
| `wrong_weight_threshold` | 4 | candidate set |
| `no_mandatory_max` | 8 | witness |
| `no_reachability` | 13 | witness |
| `allow_negative_multiplicity` | 2 | witness / typed refusal |
| `witness_sum_off_by_max` | 13 | typed refusal |
| wrong revision / wrong digest | 14 / 14 | typed refusal |
| `e_minus1`, `exclude_lower_endpoint` | **0** | **ineffective** |

The two ineffective mutations are reported, not hidden. Shrinking the window by one, or dropping
its lower endpoint, does not reach any differing pair on these 14 calibrations, so they show nothing
about the window's tightness there. `e_minus2` is the smallest window cut that is detected.

## Economics (`perf65.txt`, `perf65_runs23.txt`; single thread, x86-64, 3 runs)

| cost | value |
|---|---|
| global certificate | 5.4 s, once per source revision |
| gate per calibration | median 291 ms, max 493 ms (calibration 9: 3 ms). This is mostly the reachability graph |
| reciprocal alone, per row | 28–31 ns source vs 3.5 ns exact: about 26 ns saved |
| whole recipe, per row | difference resolvable only at n ≤ 64. At n ≥ 256 it is inside run-to-run noise (−279 to +1,416 ns) |
| break-even | about 98k rows per calibration for the gate; about 2·10^8 rows including the certificate |
| calibration 10 at n = 8 | the exact recipe mismatches on 1 random row in all 3 runs. This is why it is not admitted there |

The saving is real but small. The gate's value is the proof, not the speed.

## Build and run

You need a tflite-micro checkout at `5f64e4ad`, with the reference library built
(`gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a`). Compile `softcert65.cc`, `trace63.cc`,
`recipall63.cc`, `perf65.cc` and `../fisherz-softmax-v1/softmax_oracle.cc` from the checkout root:

```sh
g++ -std=c++17 -O2 -fno-rtti -fno-exceptions -fno-threadsafe-statics \
  -DTFLM_REMOVE_VIRTUAL_DELETE -DTF_LITE_DISABLE_X86_NEON -DKERNELS_OPTIMIZED_FOR_SPEED \
  -DTF_LITE_USE_CTIME -Werror -Wall -Wextra -Wno-unused-parameter \
  -I. -Itensorflow/lite/micro/tools/make/downloads \
  -Itensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -Itensorflow/lite/micro/tools/make/downloads/kissfft \
  -Igen/linux_x86_64_default_gcc/genfiles/ \
  <probe>.cc gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a -Wl,--gc-sections -lm -o <probe>
```

Then run from this directory:

```sh
./softcert65 global 5f64e4ad $(cat $TFLM_DIR/tensorflow/lite/micro/kernels/internal/{common.h,fixedpoint.h,reference/softmax.h} | sha256sum | cut -c1-64)
uv run heldout65.py ./recipall63 ./trace63 recipall65_heldout.json 3
TFLM_DIR=... uv run softcert65.py ./softcert65 ./trace63 ./softmax_oracle /tmp/run.json --mutations   # ~50 s
uv run fixtures65.py /tmp/run.json ./trace63 ./softmax_oracle fixtures65   # byte-identical today
./perf65 > perf65.txt
```

The fixtures are replayed without TFLM by `../pal256-admission-v1/replay`.

## Run record (not committed)

The gate's full validation record (every candidate, witness, mutation and timing; about 290 KB) is an
output, not an input: neither the replay nor any script reads it. It is therefore not kept on `main`.
Regenerate it with the `softcert65.py` command above. The copy produced while landing is
`git show 26666aa0f:.claude/probes/source-behavior-v1/softcert65_run.json` (#1800). A re-run gives
identical decisions, witnesses, unreachable pairs and mutation verdicts; only timings differ.

JSON data files (`recipall63.json`, `recipall65_heldout.json`, `fixtures65/softmax_tables65.json`) are
written compactly, one line each; their generators write the same form, so regeneration stays
byte-identical.

## What this does not establish

- It covers one source revision, one compiler and x86-64. Another revision needs a new certificate,
  since the certificate is keyed by source digest.
- It does not admit the exact reciprocal anywhere except calibration 9, and it changes no kernel.
- It makes no claim for int16 SOFTMAX, other output quantizations or sums outside `[2^19, 2^28)`.
- It does not import ONNX or TFLite models; calibrations are given as numbers.

## Provenance

The work comes from research branch `claude/bpe-tokenization-architecture-3xd4eh`:
- `4b657dd25` (P1 gate);
- `3fbd44792` (P2 fixtures);
- the 6.3 helpers `trace63`, `recipall63` and `softmax63.py` (`7c8898274`);
- `softmax_oracle.cc` (`74804aab0`).

The audit narrative is Amendment 6.5 of `.claude/board/entries/2026-10-26-d-tflp-parity-audit.md`
on that branch.

Changes made while landing:
- table-identity verification in the gate, with `N`/`U` outputs and the `skip_table_identity` switch;
- `TFLM_DIR` instead of a hard-coded path;
- the identity controls;
- the run record regenerated with them. Decisions, witnesses and unreachable pairs are identical to the research run.
