# 5+3 savant findings banked (S1,S2,S3,S5; S4 pending)

## Load-bearing deltas for v2
- PERSISTENCE GATE (S2#3, S5#1-2): canonical_node.rs:957-961 "never move once instances persist, retired tenants never reused". R2 delete+rebase lawful ONLY if no Full-schema row / FieldMask bit 18-20 ever persisted. Unproven -> fail-safe alternative: tombstone 18-20 (RETIRED, slot reserved), BoardAggregates stays 21. OPERATOR DECISION.
- MoorePalettePairs=19 (S5#3): clean LUT tenant; removal only bundled. Unratified (F9, S1#8-9: no batched mint, no envelope-auditor verdict). Ask operator separately.
- R2 inventory additions (S2#4, S3#4-5,7): band_reading.rs:150 "reserved at 21"; lib.rs:128 doc; canonical_node.rs:1449-1451 test list, :1099-1125 docs; Full row end 332 (not 300) -- tenant_bytes 252->300; moore_tenant.rs imports :30-31, doc :1-28, cfg(test) :401+ must go; moore_nars16_probe.rs compiles after R2 -> keep per N1 (not delete).
- R2 G4 add: diff check MooreSlot/MOORE_LANES/ALL/index/offset byte-identical before/after (S5#10).
- R3 (S1#1-4, S3#9-10, S5#5-6, S2#1-2,9): F5 is already an EPIPHANY (archive :9306, :9339-9346). No operator request for strict decode found (S1#4). chain_admission has NO opcode step -> "forward_checked used by admission" is new wiring (AP6). Simpler: drop forward_checked; keep isa::Opcode::decode as free strict decoder, forward total only. Pre-1bcb15b6 semantics: v2 InferenceType::from_mantissa, 0->Deduction, +-6 -> Synthesis-average arm with type restamped; v1 3-bit path; keep cfg split in one place. Affected callers: dcr_w0_replay_budget.rs:127, d_rpf_table_1.rs:212, chain_confidence.rs:132, moore_nars16_probe.rs:365, stage26_v3_parity.rs:95-103. chain_replay.rs:134-142 doc must revert.
- R4 additions (S1#5-6): storno entries/2026-10-09-d-rpf-conf-0-confidence-surface.md:54-55; STATUS_BOARD.md:53 D-RPF-5. R4 wording conditional on tombstone vs delete (S5#9).
- R1 (S2#6-8, S5#8): no consumer (S3#2 all repos zero). Add G1 fail-closed test for tags 2..=7. Scope F1: CausalWitness/Qualia signed i4 are pre-existing, non-mask -> out of scope, state it. Persisted from_tag 2..=7 activation: none evidenced; state the check.
- MooreSlot::offset (i8,i8) (S5#4, N6): signed coordinate helper survives; keep as N6, flag to operator.

## S4 cascade (origin/main 65f45706)
- R1 clean revert (96304b88 then 14831543); 3 code files; new from_tag test follow-on.
- R2 NOT git-revertable (28ec7c07 created moore_tenant.rs incl. MooreSlot; perturbation-sim examples + CI rust-test.yml:484-495 need it) -> hand-edit. In-file tests moore_tenant.rs:401-470 (asserts len==21 :437). canonical_node: :1316-1330, 252/300 literals, :1100-1125, :1449-1451, const assert :1488, tests ~:2727; band_reading.rs:150.
- R3 ~15 files, revert NOT clean; post-1bcb15b6 dependents: d_rpf_conf_0.rs:384-385 (asserts Err mantissa 0), d_rpf_table_1.rs (e9ac316e, today), d_rpf_table_1_bench.rs:181,184, chain_counterfactual.rs:52,225-247; isa.rs must stay (mask-risc resident_ce64_predicate.rs:12,19 uses Field::span); revert order bad7b157,9fcfb8cb,1bcb15b6 or hand-edit; docs/architecture/ce64-semantic-upper-half.md +16 lines; G2 needs new loader test.
- R4 ~12 docs + 2 generated indexes; all appends (other sessions edited today).
