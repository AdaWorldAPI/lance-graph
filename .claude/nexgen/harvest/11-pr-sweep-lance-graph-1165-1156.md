# PR sweep — lance-graph #1165–#1156 (verbatim, 2026-09-05)

- #1165 — board correction of #1163 verdict. NONE.
- #1164 — (closed unmerged) four superseded readings. NONE.
- #1163 — 58.2% inversion + stranded-branch method fix. WEAK: `atlas.rs:465` agreement between two DAG-depth ancestor mechanisms.
- #1162 — (closed unmerged) Storno: HHTL 58.2% inversion, 352/2,048 non-zero key bytes in `all-lanes.soa`, rail-head agreement 314/314. WEAK.
- #1161 — `plan_dids.py` CI gate. NONE.
- #1160 — D-DCR-4 Σ-transport gate: `jc::ewa_sandwich` vs `lance_graph_contract::sigma_propagation::ewa_sandwich` byte-identity (`Spd2`, `0.5*(r01+r10)` symmetrization). STRONG. Ruling: "lance-graph-contract is zero-dependency by design; its manifest forbids even optional path deps" → forced-copy + gate pattern.
- #1159 — regrade dangling refs to retracted `epistemic_bassin`/`basin_lanes`/loco band 0x87..0x8B. WEAK landmine: retracted design had `sweep_ternlog`/`eval_ternlog`, `ASKED_CONTESTED 0x80`…`ASKED_SILENT 0x02`. Ruling `E-SIX-SEMANTIC-FAMILIES-MUST-NOT-IMPERSONATE-EACH-OTHER-1` — only `TERNLOG = FnIndex(0x86)` survives.
- #1158 — D-PRLR-1..5 for R2IL regfile probe. NONE.
- #1157 — R2IL session coordination; TERNLOG-as-address vs palette-vocabulary. WEAK: reaffirms `ogar_loco::TERNLOG = FnIndex(0x86)`, one FnIndex / 8-bit truth table addressing 256 combinators; minted-only, unconsumed.
- #1156 — untracked-plan census 53/208. NONE.
