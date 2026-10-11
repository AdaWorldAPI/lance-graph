# 5+3 draft v2 — change ledger vs v1 (consolidated from S1-S5; reviewers not yet cast)

R1 signed Register128 readings: unchanged — clean revert (96304b88, then 14831543); plus a new fail-closed test for from_tag 2..=7. CausalWitness/Qualia signed i4 are out of scope (pre-existing, not masks); this is stated.

R2 Moore tenants 18-20: CHANGED
- hand-edit, not git revert (MooreSlot lives in the same file; CI rust-test.yml:484-495 depends on it)
- OPERATOR DECISION A: tombstone (18-20 RETIRED, slots reserved, BoardAggregates stays at 21) vs delete (BoardAggregates back to 18). Delete is lawful only if nothing was ever persisted under the 21-layout (canonical_node.rs:957-961). Fail-safe default: tombstone.
- OPERATOR DECISION B: MoorePalettePairs=19 is a clean LUT tenant but unratified (no batched mint, no envelope-auditor verdict). Keep it and ratify, or retire it with the others?
- inventory added: moore_tenant.rs in-file tests :401-470, canonical_node.rs :1316-1330/:1449-1451/:1488/~:2727, band_reading.rs:150; tenant_bytes 252→300 (Full row ends at 332, not 300)
- moore_nars16_probe.rs: kept (it compiles after R2; N1)
- G4 added: diff check that MooreSlot/MOORE_LANES/ALL/index/offset are byte-identical
- .arxiv/2607.21273 weighs against Energy4/NARS readings on Moore

R3 strict decode: CHANGED
- forward_checked dropped (admission has no opcode step; it would be new wiring, AP6)
- forward returns to total pre-1bcb15b6 semantics (v2 from_mantissa: 0→Deduction, ±6→Synthesis-average with the type restamped; v1 3-bit path; one cfg split)
- isa.rs + Opcode::decode kept as a free strict decoder (mask-risc resident_ce64_predicate.rs uses isa::Field)
- ~15 files incl. post-strict dependents (d_rpf_conf_0.rs:384, d_rpf_table_1.rs, d_rpf_table_1_bench.rs, chain_counterfactual.rs); revert order bad7b157,9fcfb8cb,1bcb15b6, or a hand-edit
- docs/architecture/ce64-semantic-upper-half.md; chain_replay.rs:134-142 doc
- G2: new loader test — 126 legacy vectors replay without refusal
- OPERATOR DECISION C: confirm strict decoding was never ordered (S1 found no request)

R4 board storno: CHANGED — ~12 docs + 2 indexes, all appends; added d-rpf-conf-0 :54-55, STATUS_BOARD :53 (D-RPF-5), :22 D-ART-1, :39-40, :57, LATEST_STATE :18-24; wording depends on Decision A.
