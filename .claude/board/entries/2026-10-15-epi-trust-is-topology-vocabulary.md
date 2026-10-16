# 2026-10-15 — `TrustTexture` is the MUL vocabulary for the topology coordinate, not a second reading

STATUS: fixed | OUTCOME: doc drift corrected in `causal-edge/src/layout.rs`,
`causal-edge/src/lib.rs`, `lance-graph-contract/src/epistemic_state5.rs`,
`lance-graph-contract/src/band_reading.rs`; executable contract unchanged |
OPEN: none.

**What the drift was.** Four sites described bits 59..60 as carrying TWO
readings — an "epistemic-trust texture" (`TrustTexture`: how much to trust the
edge; `0b11 = Contradiction`) and a "factual topology" (`CausalTopology`) —
"correlated in practice but not the same fact", with "historical factual
provenance NOT guaranteed for old rows" and "no mapping into the canonical
state" for the Trust lens. `epistemic_state5.rs` and `band_reading.rs` then
listed `TruthLens::Trust` beside `ReasoningBand` as having written "OTHER
meanings" into the merged field.

**The current contract.** Bits 59..63 are the epistemic ladder
(`EpistemicState5 = Topology2 × Certification3`; README front-page diagram and
`docs/architecture/ce64-semantic-upper-half.md`). Its lower coordinate is the
Faktorortung — Direct / IndirectKnownIntermediates /
IndirectUnknownIntermediates / Unknown — and `TrustTexture`'s Crystalline /
Solid / Fuzzy / Murky are the Meta-Uncertainty-Layer's names for those SAME
four cells (`Murky` = `Unknown`). The mapping is the identity; the Proceed /
Sandbox / Compass column is the MUL's reaction to a cell, not a second
meaning of the bits. A contradiction that needs revision is an **epistemic
pothole** the MUL reacts to, not a cell of the field. Only the historical
`ReasoningBand` on 61..63 wrote other meanings. Historical provenance is the
`EdgeProvenance` gate (were the bits written at all), not a semantic gap.

**Why no code moved.** No class in the tree declares either `TruthLens`
variant (`.band_reading(` has no production caller; `project_band` is reached
only by the obsolete `reasoning_band_probe`), so the over-refusal the old prose
implied for Trust-declared classes was never live. Gates: `causal-edge` and
`lance-graph-contract` tests + clippy `-D warnings` + fmt unchanged-green.

Source of the ruling: this session, reading the README diagram against the
crate prose. Related: `2026-10-07-epi-legacy-deprojection.md` (kept; its
"other meanings" line is regraded by this entry, not edited).
