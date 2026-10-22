# BNN modes are not interchangeable

Status: **test-only vocabulary guard**, no neural engine.
- **Binary/Binarized Neural Network (BNN):** bipolar values allow an exact XNOR+PopCount inner product, provided masking, bit polarity and group width agree. In the existing substrate, reuse `ndarray::bitwise` / `mask-risc`, not a new SIMD backend.
- **Bayesian Neural Network (BNN):** distribution/uncertainty over parameters or predictions. A quantized mean/score does not retain posterior variance, calibration, or sample provenance.

Two tests separately falsify arithmetic and uncertainty conflation. Extending either to 256x256 palette relation tables requires model-specific typing, held-out oracle and BNN kind in the law/cost catalogue.

Run: `cargo test --manifest-path crates/jc/Cargo.toml --test palette256_bnn_modes_probe`.
