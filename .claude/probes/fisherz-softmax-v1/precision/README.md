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

```sh
cargo build --release                                   # baseline x86-64: software FMA
RUSTFLAGS="-C target-cpu=native" cargo build --release  # hardware FMA
uv run driver.py target/release/precision-probe results.json
uv run deskew.py deskew.json
```
