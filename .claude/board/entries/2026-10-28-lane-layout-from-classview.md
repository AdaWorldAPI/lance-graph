# Lane layout per class: chosen at compile time, resolved once through the ClassView

**Status:** TEST-PINNED. Code: `crates/lance-graph-contract/src/lane_codec.rs`, `ClassView::epi_lanes`, `crates/cognitive-shader-driver/examples/thought_layers_probe.rs` (`bind_lanes::<L>`, `write_epi`). Closes the open point of `2026-10-28-thought-layers-nan-skip.md` (the fixed lab layout).

## Decision

- The consumer fixes a class's lane layout at compile time as `EpiLaneClass::SPEC`, built with `checked()`: an invalid layout fails to compile. Fields: width (8 or 9 bits), code space (Sym / Asym), value offset, presence offset, lane count.
- The presence plane is mandatory. Absent (NaN, `$null`, UNKNOWN) is always presence 0 and value 0, at 8 and 9 bits alike. At 9 bits the presence bit is also the sign, so presence 0 with a nonzero magnitude is a negative value. A 9-bit lane must be Sym.
- ONNX `QuantizeLinear` / TFLite annotations pick the code space through `CodeSpace::from_zero_point` (0 = Sym).
- `ClassView::epi_lanes(class)` names the same constant for a classid. Default `None` = undeclared, and binding refuses, like `epistemic_reading`.
- A class is bound once (`bind_lanes::<L>` returns a `LaneBinding<L>`); writes take the binding and do not look the class up again. A binding shared across threads belongs in a `OnceLock` set when the registry loads; `LazyLock` does not fit, because its initializer cannot take the view.

## Tests

Contract: every validation rule refuses, adjacent regions pass, valid layouts pass, absent is exactly one of the 512 (value, presence) pairs, presence bits beyond the first byte, zero point to code space, a `compile_fail` doctest. Probe: binding refuses an undeclared or different layout; absent not written (DR512 and ONNX uint8); own value kept; 9-bit negative written; computed −0.499 written as +0; absent beside a change; asym code 0 written; sym int8 code −128 with presence 1 is a value; the contract rule agrees with `codec9` on all 512 codes. Eight disables, each red: bind check, keep step, presence-bit copy, presence-only test, value-only test, presence byte offset, overlap check, 9-bit Asym.

## Open

The classid → `ClassView::ClassId` (u16) mapping is the caller's; the probe takes `ClassId` as an argument rather than deriving it from the row key.
