# Lane layout per class: chosen at compile time, resolved through the ClassView

**Status:** TEST-PINNED. Code: `crates/lance-graph-contract/src/lane_codec.rs`, `ClassView::epi_lanes`, `crates/cognitive-shader-driver/examples/thought_layers_probe.rs` (`write_epi::<L>`). Closes the open point of `2026-10-28-thought-layers-nan-skip.md` (the fixed lab layout).

## Decision

- The consumer fixes a class's lane layout at compile time as `EpiLaneClass::SPEC`, built with `checked()`: an invalid layout fails to compile. Fields: width (8 or 9 bits), code space (Sym / Asym), never-touched code (None / PresenceZero / SymMin), value and presence offsets, lane count.
- ONNX `QuantizeLinear` / TFLite annotations pick the code space through `CodeSpace::from_zero_point` (0 = Sym). The never-touched codes mirror `ndarray` `codec8` (`NoNaN`, `NaNAtMin`) and `codec9` (presence 0 and value 0).
- A 9-bit lane is DR512 only: Sym, PresenceZero, with a presence plane, matching `codec9`'s non-configurable contract.
- `ClassView::epi_lanes(class)` names the same constant for a classid. Default `None` = undeclared, and the writer refuses, like `epistemic_reading`.

## Tests

Contract: every validation rule refuses, valid layouts pass, presence bits beyond the first byte, never-touched per policy, zero point to code space, a `compile_fail` doctest. Probe: undeclared or different layout refused (nothing stored, no look), NaN not written (DR512 and ONNX uint8), own value kept, 9-bit negative written, computed −0.499 as +0, NaN beside a change, asym code 0 written, TFLite code −128 not written and 0 written, and the contract rule agrees with `codec9` on all 512 codes. Six disables, each red: view check, keep step, presence-bit copy, SymMin code, presence-only test, presence byte offset.

## Open

The classid → `ClassView::ClassId` (u16) mapping is the caller's; the probe takes `ClassId` as an argument rather than deriving it from the row key.
