
## 2026-08-04 — [Opus filigree / edit-only] D-BLW-2 REBUILD — `examples/blw_binding.rs`

**Branch:** `claude/x265-x266-plans-review-h9osnl`. **Scope:** ONE new file,
`crates/lance-graph-planner/examples/blw_binding.rs` (~830 lines). Nothing else
touched — `blw_tenant.rs`, every `src/` file, and `Cargo.toml` are unmodified
(the example needs no new dependency: `lance-graph-contract` +
`lance-graph-planner` only).

**NOT COMPILED, NOT RUN.** Edit-only agent; the orchestrator gates
`cargo fmt` / `clippy -D warnings` / `test`. No claim of green/passing/measured.

### The KILL being repaired (plan §12.7)
Predecessor `blw_texture.rs` (deleted at `cbca9e6`) wrote 3 loci of 24, only
`Antecedent` shared ⇒ `agreement_count` capped at **1** before any verse was
read; 21 loci `0.0000` always; means 0.0015–0.0825.

### Write-site count (stated before anything else)
**9 write sites of 24 ⇒ agreement ceiling 9.** All nine are in ONE function
(`mint`) and ALL are shared by all four stances — there are **no private loci**.
Menu: `SMeaning`(4) `PMeaning`(5) `OMeaning`(6) `SupportedBy`(9) `Supports`(10)
`QualiaReference`(12) `MeaningLevel`(13) `Quorum`(14) `Contradiction`(15).
What differs per stance is **focus selection**, not which loci get written.

15 slots are `0` BY CONSTRUCTION and each is justified in-source: 0–3 TEKAMOLO
(no TEKAMOLO parse exists), **7 `Antecedent` deliberately retired** (`stream`
collapses every personal pronoun to one referent — binding it fabricates
coreference, and it was the predecessor's entire capped axis), 8 `BasinAnchor`,
11 `RunbookEvidence`, 16–23 reserved. Every other zero is a measurement, and the
report separates `n/a` (construction) from `none` / `oow` (measurement).

### Two quantities, never averaged (§12.3c A4)
- **(a) LEVER** — `Lever::{Both,BelowOnly,AboveOnly,Collapsed}` over loci 9/10.
  `Collapsed` = the nihilism signature. Counted over FOCUSED verses only.
- **(b) TORQUE** — `max |at(12) − at(l)|`, `l ∈ {4,5,6}` bound. `None` when
  undefined; **never reported as 0** (0 would mean said==meant).
Printed in two separate tables with an explicit "never combined" line.

### Corrections of predecessor defects carried into the design
- **No saturation.** Out-of-window targets read UNBOUND (`Silence::OutOfWindow`)
  instead of clamping a 5-digit delta to `+7` — clamping manufactures agreement.
- Offset 0 never written (register reads 0 as unbound) — disclosed in advance.
- `QualiaReference` is backward-only (`pick_backward`), which is both the right
  reading of "the event that SET my texture" and what stops locus 12 from
  degenerating into a duplicate of `Quorum`.
- Near-constant loci (>0.90 bind rate) are flagged and a second,
  discriminating-loci-only agreement mean is printed beside the raw one.
- Degeneracy by prevalence (>0.90 fire rate) excludes a stance from the pairwise
  table and prints it.
- Anchor-index guard: asserts verses 55/60/62/77 carry their pre-registered
  texts, so a differently-generated TSV cannot silently shift every verdict.

### Verdict rule, fixed in source before the run
An anchor pair SEPARATES iff its 9-locus structural distance strictly exceeds
the LARGEST control-pair distance — the threshold is the corpus's own baseline
churn, not a chosen constant. Anchors: A1 55↔62, A2 60↔77, A3 60↔62. Controls:
(96,97) (98,99) (100,101). Corpus indices verified against `/tmp/kjv_verses.tsv`
(31,102 rows) before writing. Corpus bounded to 2,000 by default, argument
overridable (`all` for whole book).

### Not verified without a compiler (orchestrator gates)
Type-check of every call site (`stance_panel` tuple order, `Copula` variants,
`CausalWitnessFacet` method receivers), clippy under `-D warnings`, `cargo fmt`,
and the runtime cost of the two `stance::stream` passes (main + horizon).

### Seam stopped at / NOT done
No cross-language arm (no morphological parser for Latin/Greek/Syriac/Hebrew —
hand-writing an `in quo` matcher would fit the pre-registered A3′ answer). No
substrate surface (tenant / kanban / batch-writer) — that is D-BLW-1. No
validity claim, no p-values, no fusion claim.
