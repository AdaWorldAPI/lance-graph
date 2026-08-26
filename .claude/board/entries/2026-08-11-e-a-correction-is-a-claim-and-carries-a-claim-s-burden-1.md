## 2026-08-11 — E-A-CORRECTION-IS-A-CLAIM-AND-CARRIES-A-CLAIM-S-BURDEN-1

**Status:** FINDING `[G]` on the code facts and on its own rule; **its `Pair48`
successor recommendation is WITHDRAWN and several of its INFERENCES are
regraded** — see `E-THE-DOCTRINE-DOC-EXISTED-AND-I-NEVER-READ-IT-1` (top of
file) and `weather-normalized-substrate.md` §12.12. Read that first.

**A correction written in the same breath as the error it replaces has had no
independent gate — and one of ours was wrong.** The helix arc's ledger entry C2
recorded rejecting a 12-byte `FacetSchema::Pair48` for wind from/to, on the
ground that "the 6 B `HelixResidue` lane is already 2×24 in/out by
construction", citing `canonical_node.rs:963`. That line is a **comment**
stating a **width identity (48 = 2×24) plus coverage**, never a structural
in/out claim. **The rejected 12-byte `Pair48` was the structurally correct
answer** — C2 replaced a right answer with a wrong one and banked the wrong one
as a lesson, i.e. the failure mode compounding inside the ledger built to catch
it.

The audit falsified the whole premise: `Signed360` is ONE signed orientation
(`rim(3)+polar(1)+azimuth(2)`, `residue.rs:76-116`); `ResidueEdge` **cannot carry
a hemisphere sign at all** (`sprite_replay.rs:56-63` — `Sign::Neg` reconstructs
as `Sign::Pos`, *"a real, measurable structural error"*); it has **no azimuth**,
the only angular field in the lane; and the crate states twice, unprompted, that
**no 2-DOF direction codec exists** (`sprite_replay.rs:47`,
`continuous_field.rs:31-43`) and that inventing one is the "invented round-trip
API" it warns against. Also: `end_idx` is monotone in `n` (`residue.rs:258-266`, the
`end_idx_monotonic_in_n` test) ⇒ **non-circular**, so a wrapped bearing puts
359° and 1° at maximum L1 distance; `DistanceLut`'s triangle-inequality
guarantee (`distance.rs:22-33`, regression `:87-105`) is about a **linear**
index order and does not survive being re-purposed as an angular one
(`residue.rs:53-55` names this failure mode for `distance_heuristic`).

**Structural finding worth keeping regardless of weather:** there is **no
per-value-lane reading selector in the contract at all** — `ReadMode` has three
axes (tail / value_schema / edge_codec), none of which selects a *reading* of a
value lane; `EdgeCodecFlavor` covers `EdgeBlock`, a different region. Exhaustive
grep (`HelixResidue`, all `**/*.rs`, unbounded): 15 hits, **zero writers, zero
decoders**. Correct successor shape if a pair is needed: a NEW 16-byte facet
lane read as `Pair48 = [Signed360; 2]`, codec in `crates/helix`, lane in the
zero-dep contract, discriminant **16** — `15` is reserved for `BoardAggregates`.

**Pre-existing defect found in passing:** `Signed360::sign()` decodes an
all-zero, never-written lane as a definite `Sign::Neg` (`residue.rs:109-115`),
where sibling lanes `Tekamolo` / `CausalWitness` explicitly define all-zero as
*unaddressed* / *unbound*. Violates the CANON zero-fallback ladder. Filed, not
fixed.

**Rule:** route corrections through the same gate as originals. "I was wrong
before" is not evidence the new version is right.

