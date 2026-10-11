# QUESTION BRIEF — Mexican-hat popcount in service of deforestation (2026-10-10)

## 1. The question
How can a centre-surround ("Mexican hat") response be computed as POPCOUNTS so that it
serves DEFORESTATION — answering a fold directly from resident masks, with no intermediate
population materialised — using ONLY the workspace's yardsticks (lance-graph-mask-risc,
lance-graph-quack, jc), ClassView + WideFieldMask for masking, and palette256:palette256 as a LUT?
Open now because the session's earlier attempt (D-MHB-1/2, R-MHB-1, tile64k) was ruled wrong by the
operator: signed pos/neg planes, float-DoG weights rounded to integers, geometric windows, new kernels.
The operator states the idea itself is sound "IF you know what you are doing" and points at `.archiv/*`
(location not found in any cloned repo or GitHub code search — scouts must look for it).

## 2. Anchors (frozen; never re-opened)
- A1-A4 of lance-graph-mask-risc (crates/lance-graph-mask-risc/src/lib.rs:62-94): plan describes,
  executor borrows, ndarray computes, caller owns memory; **A2: masks choose admissibility; magnitude is
  a separate reduction over survivors; a mask bit is never a weight**; ternlog is semantics; independent
  reference oracle.
- Quack (crates/lance-graph-quack/src/lib.rs:1-25): operators ARE masking ops; a new operator is a new
  LOWERING, never a new kernel; quack builds a Program and never evaluates one.
- jc is the home of all calibrated math (crates/jc/src/lib.rs, stats.rs, reliability.rs, jirak.rs).
- Operator 2026-10-10: no signed masks; deterministic with not one bit of rounding error; masking =
  ClassView + WideFieldMask (crates/lance-graph-contract/src/class_view.rs:243, 1065);
  palette256:palette256 is a LUT, not a semiring, not geometry (crates/helix: DistanceLut, ResidueEdge,
  curve_ruler; helix/KNOWLEDGE.md).
- CausalEdge64 layout (crates/causal-edge/src/layout.rs); I-LEGACY-API-FEATURE-GATED.
- Zero-copy law; data-flow rules (no &mut during compute; write-back gated XOR / BUNDLE).

## 3. Internal surface
lance-graph: crates/lance-graph-mask-risc (ir.rs Pred/MaskOp/Terminal, exec.rs, value.rs PowerSums),
crates/lance-graph-quack (lib.rs lowerings, bind.rs, examples/fold_join_probe.rs = D-RPF-9),
crates/jc, crates/helix, crates/bgz17 (palette, distance_matrix), crates/lance-graph-contract
(class_view.rs, facet.rs, morton8x8.rs, alpha*.rs), crates/causal-edge.
Board: .claude/plans/2026-10-08-resident-projection-fold-mask-v1.md (D-RPF), entries mexhat-*,
fold-join-deforestation-probe, D-RPF-*. ndarray: src/simd_masking_ops.rs (masked_group_sum, power sums,
ternlog popcount), src/hpc/ (pillar/mexican_hat.rs).
Unknown location: `.archiv/*`.

## 4. External domains + seeds
- Literature: deforestation / fusion (Wadler 1988 deforestation; stream fusion; shortcut fusion),
  bit-sliced / bit-serial arithmetic (SIMDRAM, bit-sliced indexes O'Neil-Quass, BitWeaving),
  integer/binary DoG & centre-surround (difference of boxes, integral images, binary LBP, census
  transform), popcount-based convolutions (XNOR-net, binarized conv).
- Systems: bitmap indexes (Roaring, BSI bit-sliced index aggregation), DuckDB aggregates/filter
  pushdown, FastLanes, Lance.
- Concepts: formal concept analysis, lattice/ Boolean algebra of sets, rough sets (lower/upper
  approximation as centre/surround), set differences as unsigned decomposition.
Existing knowledge docs to read first: .claude/knowledge/research-council-semantics-papers-2026-06.md,
encoding-ecosystem.md, folding-doctrine.md, FIRST-HAND-SOURCE-LAW.md.

## 5. What counts as an answer
A mechanism (or several) by which a centre-surround response is expressed with unsigned masks
(admissibility) + separate magnitude reductions over survivors, lowered onto EXISTING mask-risc
ops / quack lowerings, with palette relations as LUT bands, and exact integer results; plus how that
removes an intermediate population (the deforestation gain); plus falsifiable probes. Out of scope:
any new kernel, any signed mask, any float weight on the path.

## 6. Budget
≤10 items per scout; literature any date; read grades mandatory.

## 7. Premise gate result (premise-auditor, 2026-10-10): PREMISE-SPLIT — the question is re-asked as three
- Q1 ADMISSIBILITY (row masks, motion): which EXISTING mask-risc preds/ops (Range, Pred…under, AndNot,
  Ternlog, EqU32Via, MatchU32/MatchU64, strided) lower "row is in the centre band" / "row is in the surround
  band" to UNSIGNED row planes, a band being a palette256 LUT distance on the row address (V3 64K: u8:u8 IS
  the row address, crates/lance-graph-mask-risc/src/ir.rs:218-223)? ClassView+WideFieldMask selects which
  FIELD/LANE is read (one bit per field), it is NOT the row mask — do not conflate the two "masks".
- Q2 MAGNITUDE (weight, frame): weights constant PER BAND ⇒ two UNSIGNED fused Counts, arithmetic after the
  counts (jc / caller), compared without subtraction (unsigned cross-multiplication, e.g. a·Cc ≥ b·Cs in
  u64/u128). A smooth per-offset weight is a VALUE-LANE magnitude reduction over survivors (mask-risc A2),
  not a popcount — and then not deforested per query. "The premise is wrong" is an allowed answer.
- Q3 DEFORESTATION: which population-sized intermediate does the chosen form remove, against D-RPF-9's stage
  table (.claude/plans/2026-10-08-resident-projection-fold-mask-v1.md :65-74)?
Prior wrong attempts (do NOT re-propose): signed Pos_b/Neg_b planes (R-MHB-1), f64 DoG rounded to integers,
geometric 2-D windows / new window operand, new kernels, per-query weight lanes (arm B).
