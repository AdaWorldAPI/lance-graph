# 5+3 SPEC v1 — remediation of this session's architecture violations on lance-graph `main` (2026-10-10)

Scope: production code and canon docs on `lance-graph` `origin/main` that this session
(`session_01EFw2WdKr1oxvaKCJC2ua2R`) merged and that the read-only audit found in violation.
The council VERIFIES this spec; it does not redesign. Implementation (Phase 5) happens only after
the operator approves v3 (operator instruction 2026-10-10: "audit everything first, change nothing").

## 1. FROZEN DECISIONS (cited; flag only as VIOLATES with file:line)

F1. Operator 2026-10-10: no signed masks; deterministic, no rounding error; masking = ClassView +
    WideFieldMask; palette256:palette256 is a LUT; "you have no permission to ignore the main architecture".
F2. mask-risc A1–A4, esp. A2 "a mask bit is never a weight" (crates/lance-graph-mask-risc/src/lib.rs:74-94).
F3. Quack: a new operator is a new lowering, never a new kernel; quack never evaluates
    (crates/lance-graph-quack/src/lib.rs:1-25).
F4. jc is the home of calibrated math (crates/jc/src/lib.rs:49-74).
F5. Replay keystone: "Replay must not refuse history … replay stays total over everything already
    admitted" (crates/lance-graph-planner/src/chain_admission.rs:105-121); "byte-identical trace forever"
    (chain_replay.rs module doc).
F6. I-LEGACY-API-FEATURE-GATED (lance-graph CLAUDE.md): the same function name must not silently
    change semantics; layout reclaim needs a version gate.
F7. CE64 layout is fixed by crates/causal-edge/src/layout.rs; bits 46..49 = signed inference mantissa;
    a different reading of 43..52 "needs a version gate … not done here"
    (docs/architecture/ce64-semantic-upper-half.md "Open").
F8. crates/lance-graph-contract/src/hotplug.rs module doc :43 "No per-concept permission table"; :47
    "The declaration carries NO semantic identity: SPOG supplies it."
F9. V3 canon / tenant mint: append-only value tenants; a mint is an architecture change requiring the
    operator (OGAR/lance-graph CLAUDE.md "no permission to touch the main architecture"; F1).
F10. Board append-only: corrections are appended storno lines citing what they correct; never delete
    entries; regenerate `entries_index.py --write` and `supersession_index.py` LAST.
F11. Other sessions' work is not ours to break: crates/perturbation-sim/examples/{MoorePertubationsfeld,
    moore_observable_first}.rs (commits 1006a852, 56d678e1, b03c7e4a — not this session) use
    `lance_graph_contract::moore_tenant::MooreSlot` incl. `index()`, `ALL`, `offset()`.

## 2. INPUT INVENTORY (file:line, shape)

I1 Signed Register128 readings — commits 14831543, 96304b88 (PR #1410):
   - crates/lance-graph-contract/src/register128.rs: `RegisterCarving{I4x32,I8x16}`, `RegisterLaw{RelativeOffset,
     AxisPosition,Support}` (~:224-236), `SignedRegisterRefusal`, `SignedRegisterLanes` (+read_/write_i4x32/i8x16
     ~:394-431); module doc :36-38 still claims `words` is the only interpretation.
   - crates/lance-graph-contract/src/hotplug.rs: `SlabReading::{RegisterI4x32(RegisterLaw), RegisterI8x16(RegisterLaw)}`
     (~:253-257), tags 2..=7 in `from_tag`/`to_tag` (~:268-284), `Activation.register_laws`, `with_register_laws`,
     `register_law_for` (~:130-181), `bind_signed_register`, `ActivationDrift::{NotSignedRegister, NoRegisterLawFor,
     RegisterLawMismatch}`; tests `signed_rails_need_a_signed_carving_and_an_agreeing_law`, tag tests.
   - crates/lance-graph-contract/src/atoms.rs: `I4x32::{from_bytes, as_bytes}` (added only for the carving).
   - Consumers: NONE outside the contract crate and docs (grep over lance-graph/crates, OGAR, a2ui-rs,
     stockfish-rs, lance-graph-java, MedCare-rs, tesseract-rs). No OGAR authority populates `with_register_laws`.
   - Docs: .claude/board/entries/2026-10-08-register128-signed-readings.md, LATEST_STATE.md lines,
     .claude/v3/soa_layout/tenants.md row 17 note, plan 2026-10-08-resident-projection-fold-mask-v1.md,
     entry 2026-10-08-fold-join-deforestation-probe.md (mentions).
I2 Moore value tenants + CE64 re-reading — commit 28ec7c07 (+75c83c45 docs, PR #1408):
   - crates/lance-graph-contract/src/moore_tenant.rs: `MOORE_LANES`, `MooreSlot` (KEEP, F11), `MooreNars16`
     (Energy4 = CE64 46..49, :105-165), `DirectionReading`, `require_sign_triple`, `DirectionRefusal`, `MooreRefusal`,
     `MooreTenantView`, `MooreTenantMut`, `ce64_upper` (:209-219, fabricates direction 000), `from_ce64`, `lift_ce64`
     (:370-395).
   - crates/lance-graph-contract/src/canonical_node.rs: `ValueTenant::{Nars16x8=18, MoorePalettePairs=19,
     MooreNars16=20}` (~:1096-1125), `VALUE_TENANTS` entries, BoardAggregates reservation re-based to 21 (~:1300-1330),
     Full schema bytes 252→300, doc "The physical layout is RATIFIED" (:1101, :1123).
   - crates/lance-graph-contract/src/lib.rs: `pub mod moore_tenant`.
   - Tests: crates/cognitive-shader-driver/tests/moore_palette_pairs_lut.rs; crates/lance-graph-planner/tests/
     moore_tenant_isa_equivalence.rs; crates/lance-graph-planner/tests/moore_nars16_probe.rs (probe-local Energy4,
     signed SLOTS).
   - Docs: .claude/v3/soa_layout/le-contract.md §3c; tenants.md rows 18-20 + Full list; entries
     2026-10-08-hhtl-nars-moore-value-tenants.md, -moore-nars16-isa-visible-representation.md; LATEST_STATE.
I3 Strict ISA decode in forward/replay — commits 1bcb15b6, bad7b157 (+e60f2fff/32a7ddf0 test follow-ups):
   - crates/causal-edge/src/edge.rs `forward` (~:667-690): `Opcode::decode(weight.inference_mantissa())?` replaced
     `InferenceType::from_mantissa` fallback; v1 path `Opcode::decode_v1(...)?`; signature `-> Result<Self, IsaFault>`.
   - crates/causal-edge/src/network.rs `forward_chain -> Result`; crates/lance-graph-planner/src/cache/nars_engine.rs:603
     `forward_edge -> Result`; cache/stage26_v3_parity.rs fields `Result`; chain_replay.rs `replay_step` ~:189-205
     (`?`), `replay_chain` ~:266, `ReplayError::Isa{step,fault}` ~:136-143.
   - crates/causal-edge/src/isa.rs (Opcode, IsaFault, Compose, truth, Field/Operand/Contract) — truth delegation was
     measured bit-identical on 126 legacy vectors (crates/causal-edge/tests/ce64_legacy_capture.txt).
   - Measured: of 126 raw-word vectors 71 reproduce, 55 now refuse (entry 2026-10-08-ce64-isa-register-contract.md).
I4 Board/plan steering (doc only): entries 2026-10-08-mexhat-bucket-cascade-probe.md (Recommendations: F "ADOPT as
   dense path", R-MHB-1), 2026-10-08-mexhat-df-choice.md (Q ADOPT with D and F), STATUS rows D-MHB-1/D-MHB-2;
   plan 2026-10-08-resident-projection-fold-mask-v1.md :289 (`shift(grid, −d)`), :308-309 (LUT as bilinear form,
   eigen/SVD), :362 (Pos_b/Neg_b weight bits, mis-cites A2); entry 2026-10-10-d-rpf-g2-law-a-boundary.md DECISION
   ("exact revision (law A)" = f32 ISA law with measured one-code deviations); STATUS row D-RPF-TABLE-1.

## 3. THE PROPOSED RESOLUTION (committed)

R1 (I1) REMOVE the signed Register128 readings entirely: delete RegisterCarving, RegisterLaw, SignedRegisterRefusal,
   SignedRegisterLanes, bind_signed_register, SlabReading::RegisterI4x32/I8x16, Activation.register_laws +
   with_register_laws + register_law_for, ActivationDrift::{NotSignedRegister, NoRegisterLawFor, RegisterLawMismatch},
   I4x32::{from_bytes, as_bytes}. `from_tag` refuses tags 2..=7 again (fail-closed, as before #1410). Restore
   register128.rs module doc truth (`words` is the only interpretation) and hotplug.rs doc consistency (F8).
   Safe: no writer, no reader, binding fails closed in production today. Implement as `git revert` of 96304b88 then
   14831543 where it applies cleanly; otherwise hand-revert to the pre-#1410 file state of these items only.
R2 (I2) REMOVE the CE64 re-reading and the unratified tenants; KEEP `MooreSlot` and `MOORE_LANES` unchanged (F11):
   delete MooreNars16, DirectionReading, require_sign_triple, DirectionRefusal, MooreRefusal, MooreTenantView,
   MooreTenantMut, ce64_upper, from_ce64, lift_ce64; delete ValueTenant 18-20 and their VALUE_TENANTS rows;
   BoardAggregates reservation returns to 18; Full schema bytes back to 252. Delete the three tests listed in I2.
   `MooreSlot::offset() -> (i8, i8)` stays (used by another session); filed as a CONCERN for the operator (signed
   coordinate helper), not changed here.
R3 (I3) RESTORE replay totality (F5) — ESCALATION ITEM: whether strict decoding was operator-ordered is not
   recorded in the repo. Proposed resolution: `CausalEdge64::forward` returns to `-> Self` with the pre-1bcb15b6
   total dispatch, byte-identical on all 126 legacy vectors; the strict decoder stays available ONLY as a separate,
   differently named method `forward_checked(...) -> Result<Self, IsaFault>` (F6: no same-name semantic change),
   used by admission, never by replay. `forward_chain`, `forward_edge`, `replay_step`, `replay_chain` return to
   their pre-1bcb15b6 signatures; `ReplayError::Isa` removed. isa.rs (Opcode, truth, contracts) is kept; its truth
   delegation is bit-identical. If the operator confirms strict decoding was ordered, R3 is replaced by: keep strict
   `forward`, but replay must call a total path (F5 is not optional either way).
R4 (I4) BOARD STORNO (append-only, F10): append a dated SUPERSEDED/RETRACTED line to each listed entry, plan row and
   STATUS row, citing this remediation and the violated anchor (F1/F2/F3/F7/F9); mark tenants 18-20 and the
   signed-register readings RETRACTED in tenants.md / le-contract.md §3c / LATEST_STATE; mark the D-RPF-G2 DECISION
   WITHDRAWN (law A is the f32 ISA law, not exact). One new entry
   `.claude/board/entries/2026-10-10-remediation-session-violations.md` lists every retraction. Regenerate
   entries_index and supersession_index last.

## 4. NON-GOALS
N1 The mexhat / fold-join / phasor / recipe PROBE FILES are not deleted (a probe that measures is not a production
   violation; the separate /coresearch decides the correct Mexican-hat form). Only their ADOPT recommendations are
   retracted (R4).
N2 examples/shared/lab.rs → jc migration: follow-up row, not this pass (example code, not production).
N3 ndarray hpc::phase re-export through ndarray::simd: CONCERN only; separate repo; no consumer on a mask path yet.
N4 ndarray checked_affine: code is exact and additive; only its D-ART-1 recommendation is noted (R4 storno line).
N5 stockfish-rs docs naming the opcode codebook "palette": separate repo; filed as follow-up.
N6 MooreSlot::offset signed helper: other session's dependency (F11); operator decision.
N7 ISA golden ±1 LSB tolerance, duplicate planner commits, model identifier in stockfish-rs trailers: follow-ups.

## 5. PRE-REGISTERED GATES (all must pass before commit)
G1 `cargo test -p lance-graph-contract` green; `from_tag(t)` is `None` for t in 2..=7 (new test); VALUE_TENANTS has
   18 entries and BoardAggregates sits at 18 (existing field-isolation/layout tests updated, not deleted).
G2 causal-edge tests green under default (v2) AND v1 layouts; the 126-vector legacy capture reproduces bit-for-bit
   through `forward`; a new test: a weight with mantissa 0 and −6 runs `forward` without fault (total), and
   `forward_checked` refuses the same codes (strict path preserved).
G3 planner tests green; new test: `replay_chain` over a chain containing codes 0 and −6 returns Ok and a
   byte-identical trace on two runs (F5); disable run: making replay call `forward_checked` turns it red.
G4 cognitive-shader-driver and perturbation-sim (`--features moore-probe`) build; perturbation-sim examples unchanged.
G5 `cargo clippy -D warnings` on every touched crate; `cargo fmt --check`.
G6 Board: entries_index.py --write and supersession_index.py regenerated last, CI index check green; no entry
   deleted (`wc -l` never shrinks on append-only files).
G7 No model identifier in any committed artifact other than the mandated attribution trailer.

## 6. PER-SAVANT QUESTION SETS

S1 PRIOR ART (Opus): 1) Is any of R1/R2/R3 already decided differently by an operator ruling or EPIPHANY (cite id)?
   2) Does any board entry or plan from another session depend on tenants 18-20 or the signed readings? 3) Is there
   an existing "forward_checked"/strict-vs-total split precedent (e.g. chain_admission) to reuse instead of a new name?
   4) Was strict decoding requested by the operator anywhere recorded (search board, plans, AGENT_LOG)? 5) Is the
   "RATIFIED" label on tenants 18-20 backed by any ruling?
S2 IRON RULES: 1) Does R3's split (total `forward`, strict `forward_checked`) satisfy I-LEGACY-API-FEATURE-GATED?
   2) Does R2's tenant removal violate append-only value-tenant canon (is removal of a never-written tenant
   permitted, or must it be tombstoned)? 3) Does R1 leave any signed reading in the contract? 4) Any AP1-AP9 hit?
S3 CODE TRUTH: verify every file:line and every "no consumer" claim in §2 (CODED/CLAIMED/ABSENT); confirm the legacy
   capture file exists and what `forward` produced before 1bcb15b6 for codes 0 and ±6; confirm MooreSlot is
   independent of the deleted items (compiles alone).
S4 CASCADE IMPACT: list every file/test/doc/board row that must change for R1–R4, mandatory vs follow-up; does
   reverting 14831543/96304b88/28ec7c07 conflict with later commits on main (e.g. #1457, perturbation-sim)?
S5 DIFFERENT VIEWS: strongest alternative reading of each Ri without redesigning (e.g. tombstone instead of delete;
   keep MoorePalettePairs as the one clean LUT tenant); second-order consequences.

## Output contract (each savant)
≤10 findings, each `(question #, CONFIRMS | VIOLATES | GAP | PRIOR-ART-AT | RISK, file:line evidence, ≤2 sentences)`.
Read-only. No designs.
