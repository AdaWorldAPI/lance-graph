# Immaterial needle

A foldable optimization in `cognitive-shader-driver`. The name is the claim: the cross table is not built.

Kernel: `crates/cognitive-shader-driver/src/immaterial_needle.rs`.

`[a, b][c, d]` is two `(8:8)` needles. Each needle is `morton(byte, byte)`, a `u16` slot. The fold mins the two bins in register and drops them. `cross_table` returns `CrossRefused`. Not wired into the shader cycle. `CausalEdge64` is not touched.

A later wiring may call `fold` from a tile kernel. It may not allocate a `256²` buffer to hold the result.
