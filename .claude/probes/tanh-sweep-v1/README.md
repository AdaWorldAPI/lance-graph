# tanh-sweep-v1

The P0 TANH calibration sweep (D-TFLP audit, Amendment 6.1). It compares TFLite Micro's int8 TANH
against a correctly rounded reference over many input calibrations. The output contract is scale
1/128, zero point 0.

- `tanh_int8_sweep.cc`: a calibration-sweep variant of ndarray
  `tools/tflm_oracle/tanh_int8_oracle.cc`. `RunTanh` is copied verbatim; calibrations are read
  from stdin. Before the sweep was trusted, it was checked to reproduce all 7 calibrations of
  ndarray `tests/data/tflm/tanh_int8_8f6596d.json` byte for byte (output and prepare parameters).
- `sweep.py`: the calibration set (7 fixture + 17 power-of-two + 2,000 random + 200 zero-point-0,
  seed 61) and these arms:
  - **ref_real**: tanh of the exact real input;
  - **ref_f32**: tanh of Codec8's f32 decode;
  - **lut**: the interval-certified code→code table;
  - **libm**: numpy float32 `tanhf`;
  - **vstanh**: ndarray `vstanh`, via `../fisherz-softmax-v1/precision` `vstanh_dump`.

  It writes `sweep.json` (inventory, per-calibration diffs, mismatch samples).

Build, in a TFLM checkout whose `tensorflow/` tree is identical to `8f6596d`:

```sh
g++ -std=c++17 -Os -fno-rtti -fno-exceptions -fno-threadsafe-statics \
  -DTFLM_REMOVE_VIRTUAL_DELETE -DTF_LITE_DISABLE_X86_NEON \
  -DKERNELS_OPTIMIZED_FOR_SPEED -DTF_LITE_USE_CTIME -Werror -Wall -Wextra \
  -Wno-unused-parameter -I. -Itensorflow/lite/micro/tools/make/downloads \
  -Itensorflow/lite/micro/tools/make/downloads/flatbuffers/include \
  -Itensorflow/lite/micro/tools/make/downloads/kissfft \
  -Igen/linux_x86_64_default_gcc/genfiles/ \
  <lance-graph>/.claude/probes/tanh-sweep-v1/tanh_int8_sweep.cc \
  gen/linux_x86_64_default_gcc/lib/libtensorflow-microlite.a \
  -Wl,--gc-sections -lm -o tanh_sweep
uv run sweep.py ./tanh_sweep <precision>/target/release/vstanh_dump sweep.json 2000
```

Not covered: ONNX (no source oracle was run), U8 `UnitFull` tables (no source oracle), and the
int16 TANH kernel.
