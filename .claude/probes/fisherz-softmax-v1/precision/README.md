# fisherz-softmax-v1 / precision

Probe for audit Amendment 5: softmax over a Palette256 codebook with four
precision arms, scored against an independent 40-digit mpmath reference.

- `src/main.rs`: the arms, using ndarray's own `simd::mul_add_f64`
  (path dependency on a local ndarray checkout; edit `Cargo.toml`).
  A = F24 table, exact u64/i128 sums. B = F24 table as f64, f64 sum,
  `mul_add_f64` dot. M = B with separate multiply and add. U = unary f64
  table `exp(beta*(x[q]-x_max_codebook))`, no row maximum. C = per-element
  f64 `exp`, Neumaier sums. `--bench` prints ns/element.
- `driver.py`: generates 261 cases, runs the binary, scores every arm.
- `deskew.py`: per-code representative corrections against true values.
- `results_*.json`, `deskew.json`: the runs recorded in the amendment.
- `cert.py` (Amendment 6): the certified F24 bound. Same 261 cases and seed as
  `driver.py`; table entries certified by 96-bit interval arithmetic; exact
  rational arm A; per-output bound; per-row escalation (exact rational, then
  outward-rounded intervals 64-512 bits, fail closed); negative controls.
  `cert.json` is the recorded run. Costs it records are Python, not kernel, costs.

```sh
cargo build --release                                   # baseline x86-64: software FMA
RUSTFLAGS="-C target-cpu=native" cargo build --release  # hardware FMA
uv run driver.py target/release/precision-probe results.json
uv run deskew.py deskew.json
uv run cert.py cert.json
```

## Amendment 6.1

- `cert2.py`: candidate arms over the SAME 261 cases. It loads the generator,
  reference and resolvers from `cert.py`'s own text (asserting a marker), so
  `cert.py` and `cert.json` are untouched. Arms: `base` (= Amendment 6,
  re-verified per case against `cert.json`), `zero` (exact-zero error for
  entries at the row maximum), `pe` (certified per-entry error, G = 8),
  `first` / `first_pe` (exact-uniform shortcut first), `unary` / `unary_pe`
  (table referenced to the codebook maximum). Negative controls consume masks
  and refusals inside the candidate pipeline. Writes `cert2.json`, plus an
  optional canonical fixture for the Rust lab (35 MB, NOT committed;
  regenerate it, then check the sha256 recorded in the amendment).
- `src/bin/certlab.rs`: the Rust lab. It reads the fixture (no Rust-side
  regeneration), recomputes the F24 table in f64 and compares it with the
  certified table, then runs every arm in exact integers. Arithmetic is
  checked, with 256-bit cross-multiplication. It compares every per-output
  certification flag, row class and code with Python. Interval resolution is
  not implemented in Rust: those outputs are DEFERRED to Python's result and
  counted. `--bench` prints per-stage ns/element; this is unoptimized lab code.
- `src/bin/vstanh_dump.rs`: ndarray `hpc::vml::vstanh` over hex f32 input,
  used by `../../tanh-sweep-v1/sweep.py`.

```sh
uv run cert2.py cert2.json fixture.txt
cargo build --release --bins
./target/release/certlab fixture.txt [--bench]
```
