# CE64 reasoning substrate — the plateau

**READ BY:** any session touching `CausalEdge64`, the Alpha overlay or
tunnel, hydration ("global sudoku"), NARS truth/revision, the 34 recipes,
Narsese oracles, the Moore/NARS value tenants, or the split-tunnel write path.
Also `truth-architect`, `v3-envelope-auditor`, `fold-carrier-scientist`,
`premise-auditor`, `trajectory-cartographer`.

**Status:** WORKING-MODEL. This document connects material that already
exists. It decides nothing. Every row cites the file or board entry it rests
on. Where the operator's design and the code disagree, the disagreement is
listed as OPEN with the decision it needs (§4).

**Plan:** `.claude/plans/ce64-hypothesis-layers-v1.md` (D-HL-0..9).
**Research map:** `.arxiv/2026-10_ce64_hypothesis_layers_research_map.md`.

---

## §1 — The design in one picture (operator design intent, 2026-10-12)

The operator's words, condensed. The intent is stated here, but this
document does not adopt it as a code change.

1. **The edge is opaque bytes under a little-endian contract**, like a
   TCP/IP payload. Agreement on what is transferred costs precision once,
   at entry (the "entry tax"). After entry the bytes are never converted;
   operations are bit-exact on the representation, the way SIMD on BF16 is
   exact on BF16's bits. The meaning lives in the reading contract, not in
   the bytes.
2. **Narsese vs the LE contract.** With Narsese, two parties agree on the
   notation and the semantics stay in the scientist's head. With the LE
   contract, the mapping lives "in the veins of the thinking": the bytes the
   system computes on are the meaning.
3. **The ontology or text is the immutable address.**
   `Graph(classid) × NodeGuid / row` never changes. Thinking is an
   accumulating overlay on that address.
4. **64 parallel tables, like Photoshop alpha-channel layers.** Each layer is
   a copy-on-write view of the base table. Only the dirty bytes (the delta)
   are written: sparse append, merged on read, keyed by the immutable row
   id. Each layer is one distinct, testable hypothesis. Writes are saccades
   into a layer; every saccade is recorded, so the reasoning across layers is
   auditable.
5. **Two masks make it economic.** The read selection (Lance row-address
   mask) × the thinking mask (which layers touched which rows). Anything
   outside both is never decoded.
6. **The 64 is a budget, not a mapping.** Fewer than 10⁹ accumulations do
   "basic" hydration: deduction, setting bits 50..52 and 59..63, frequency
   and confidence. After that, reasoning about those results starts. The cap
   of 64 objects per address keeps meta-awareness O(1) per node however many
   operations were folded in.
7. **Hydration is a global sudoku.** On first execution it marks bits 50..52
   (Staunen; the witness ladder, a "cognitive Maslow pyramid") and sets bits
   59..63 deterministically: direct/indirect × known/unknown, plus the CARE
   relation (relates, causes, …): correlation vs causality, direct vs
   intermediate learning. Bits 50..52 also leave a signal for a later
   free-energy / entropy pass.
8. **F and C are 8 bits each (a 256 scale).** Candidate readings: linear
   NARS, log or Shannon-entropy scaled, or the 8× Moore degrees of freedom
   (palette256² tenant + NARS16 tenant) for local adjacency, attention and
   energy plus NARS reasoning.
9. **Writes amortize after a validation ladder.** The ladder is to be
   determined. A fold starts at 0.6 ns when amortized.
10. **Validate by running literal Narsese in parallel with the binary
    encoding** and comparing them with the `jc` statistics (ICC, Spearman).
    Free value-tenant slots can carry the parallel encodings.

---

## §2 — Where each idea already lives

| idea | home in the workspace | status |
|---|---|---|
| CE64 word, bit layout | `causal-edge/src/layout.rs`; `entries/2026-10-06-ce64-nextstate-register.md` §2 (S 0..7, P 8..15, O 16..23, F 24..31, C 32..39, Pearl 40..42, direction 43..45, mantissa 46..49, plasticity 50..52, W 53..58, topology 59..60, band 61..63; one const assert covers all 64 bits) | VERIFIED-IN-CODE |
| LE contract for the edge | D-CE64-NEXTSTATE-0: no `to_le_bytes` existed for the v2 edge before that change; `ce64_nextstate_probe.rs` pins an LE persist/restore and disable runs with big-endian go red | TEST-PINNED |
| The edge as an instruction word | `causal-edge/src/isa.rs` (strict decoder, register methods, per-field compute/pass/constant contracts); `entries/2026-10-08-ce64-isa-register-contract.md` | TEST-PINNED |
| F/C as the BF16-like part | `entries/2026-10-08-coresearch-ce64-bf16-for-thinking.md` (premise split: BF16 vocabulary applies to bits 24..39 only); `.arxiv/1905.12322_narrow_storage_wide_accumulate_round_once.md` | EXPLORATION MAP |
| Bits 59..63 as topology × certification | `EpistemicState5` (`contract/src/epistemic_state5.rs`); D-GSO-7a relational certification `0 Open … 5 Causes` (`relational_certification_probe.rs`); topology lens D-GSO-4 | TEST-PINNED (probe level) |
| Evidence writer for 59..63 | D-CTX-7 `GadamerRevision` → declared versioned transition | In PR (probe) |
| Pearl in and out of the edge | D-PEARL-IO-0 `pearl::{reason, revise, hydrate}` (`pearl_ladder_probe.rs`) | TEST-PINNED (probe) |
| Bits 40..42 are a plane lattice, not a ladder | coresearch X13: `SO = 0b101` and `PO = 0b011` are incomparable subsets | VERIFIED-IN-CODE |
| Alpha overlay ("allocate ≠ claim") | `contract/src/alpha.rs`: an ephemeral, discardable table at the same addresses as the baked spine; key copied verbatim; `AlphaStamp { cycle, seq, rung, visits }`; `AlphaMask` with and/or/xor/and_not over `u64` words | VERIFIED-IN-CODE |
| Split tunnel, parallel lanes | `contract/src/alpha_tunnel.rs`: **10** lanes (`rung_schedule::LEVELS = 10`), one writer per lane, `run_wave` and `run_wave_parallel` required byte-identical | VERIFIED-IN-CODE |
| World G from the route | `contract/src/spog_tenants.rs` (`graph_of(addr) = classid >> 16`, `merge_in_claim_order`); D-ALPHA-G-0 | TEST-PINNED |
| Sparse append, fold on read | `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md` | MEASURED in one consumer (HubSPO-rs) |
| Lance read-selection mask | `lance-graph-benches/examples/lance_source_probe.rs` builds `RowAddrMask::from_allowed(RowAddrTreeMap)` | used in a probe |
| Stable row ids | `entries/2026-09-23-stable-row-ids-hold-tombstones-exact-seal-still-owns-cohort.md` | MEASURED |
| Global sudoku orchestration | `plans/2026-10-06-global-sudoku-replayable-orchestration-v1.md`, D-GSO-0..8 | P0..P7 shipped as probes |
| Staunen / surprise scheduler | D-CE64-STAUNEN-0, D-CE64-CYCLE-0 (`staunen_scheduler_probe.rs`) | TEST-PINNED, MEASURED |
| Self-orchestration from local feedback | D-CE64-LOOP-0 (`self_orchestration_probe.rs`) | TEST-PINNED, MEASURED |
| Moore + NARS value tenants | `ValueTenant` 18 `Nars16x8`, 19 `MoorePalettePairs`, 20 `MooreNars16` (`canonical_node.rs`); `entries/2026-10-08-hhtl-nars-moore-value-tenants.md` | layout TEST-PINNED, MooreNars16 semantics CANDIDATE |
| Resident CE64 predicates (no extraction) | D-RPF-0 `Pred::MatchFacet16Strided` over `MaterializedEdges`; `raw5 >= t` = union of ≤ 6 patterns | TEST-PINNED |
| Confidence surface, stall, duplicates | D-RPF-CONF-0, D-RPF-P5 (`causal-edge/tests/d_rpf_conf_0.rs`, `d_rpf_p5_beta.rs`) | MEASURED |
| NAL against OpenNARS / ONA | `.claude/knowledge/nal-recipe-audit/` (oracles pinned in `oracles.md`); D-NAL-AUDIT-0/MCP-0/REV-0 | MEASURED, merged #1495 |
| Reliability statistics | `jc/src/reliability.rs`: `spearman` (:180), `cronbach_alpha` (:225), `icc` (:292); `jc/src/quorum.rs` `cronbach_report` | VERIFIED-IN-CODE |
| Thinking palette | `contract/src/cognitive_palette.rs`: 256 codes, full (144 verbs, 34 recipes 145..178, 36 personas 179..214, …); growth goes to the `ogar-loco` domain range | VERIFIED-IN-CODE |
| Rubicon kanban | `contract/src/kanban.rs` (Planning → CognitiveWork → Evaluation → Commit / Plan / Prune) | VERIFIED-IN-CODE |
| Potholes / reasoning gaps | `lance-graph-ogar/src/recipe_vocab.rs` `Refusal::Ungrounded`; planner `nars::tactics` `ReasoningGap` | VERIFIED-IN-CODE, both dropped after reporting |

---

## §3 — Numbers to cite, and not confuse

| quantity | value | source |
|---|---|---|
| fold over one 64-bit word | 0.6 ns | `.grok/board/CLAUDE_LANE_FOLD_CAPSTONE.md` ("0.6 ns is a word-fold … 605 ns is the plane … Citing one as another is a defect") |
| plane execute | 605 ns | same |
| fold over one stored record | ≈ 1.7 ns | `entries/2026-09-24-cost-scales-with-dirty-rows-replay-budget-before-materialization.md` |
| one Lance write (floor) | ≈ 1.7 ms ⇒ one write pays for ≈ 10⁶ folds | same |
| history read: resident fold vs Lance scan | 0.72 ms vs 336 ms | split-tunnel knowledge doc |
| one CE64 self-orchestration fold, including selection, no SIMD | ≈ 830 ns | D-CE64-LOOP-0 |
| unit-evidence revision stall | c = 244 (w ≈ 22.2) after 20 observations, all 32 runs | D-RPF-P5 E2 |
| duplicated evidence | 4 copies ⇒ posterior sd 2.04× too small; 8 copies 3.48× | D-RPF-P5 E3 |
| naive "surprise creates work" | resolved 0.649 vs round-robin 0.964; noise share 0.918 | D-CE64-STAUNEN-0 |
| what the register must carry | forgetting the evidence table → 0.000 resolved; the posterior H → no change (an echo) | D-CE64-CYCLE-0 |

**Budget arithmetic from these numbers.** 10⁹ word-folds ≈ 0.6 s.
10⁹ plane executes ≈ 10 min. 10⁹ CE64 loop folds at today's 830 ns ≈ 14 min.
"Basic hydration in < 10⁹ accumulations" is cheap only if an accumulation is
a word- or record-fold, not a loop step with selection.

---

## §4 — Where the design and the code disagree (OPEN)

Each row needs a decision before code changes a stored meaning. Changing what
stored bits mean requires a version gate (`I-LEGACY-API-FEATURE-GATED`).

| # | field / thing | design intent (§1) | code today | evidence | decision needed |
|---|---|---|---|---|---|
| O1 | bits 46..49 | signed activation / residue (+7 excitation … −8 inhibition) | signed opcode: sign = chain direction, magnitude = NARS rule; strict decoder executes only +1, +2, −1, +4, +5 | `isa.rs`; ISA register contract entry. The transition-loop design says the operation lives in the running thought ("do not whisper the operation back into the result edge"), which contradicts the opcode reading | Which reading, under which ClassView. D-CE64-LOOP-0 kept activation in a transient trace instead |
| O2 | bits 50..52 | Staunen marking on first execution; follow-up signal for free energy | three per-plane hot/frozen plasticity flags; `CausalEdge64::learn` gates on them | D-CE64-STAUNEN-0 pinned that writing surprisal there would decide whether `learn` adopts an archetype | Move plasticity, move Staunen, or give Staunen a separate home |
| O3 | bits 53..58 | W breadcrumb: handle to the thought that changed the state | corpus root handle; `forward` and `syllogize` return W = 0 | PROBE-W-PRESERVE (`ce64_op_contract.rs`) pins W lost through `forward`; D-CE64-CYCLE-0: W on Epi5 changes would cover ≈ 1.5 % of belief updates; G6: three competing readings of W | W ownership (G6) |
| O4 | bits 40..42 | "Pearl ladder" | plane-subset lattice; not ordinal | coresearch X13 | none for the code; doc wording must not threshold it |
| O5 | number of layers | 64 hypothesis layers (a budget) | `AlphaTunnel`: 10 lanes, one per rung | `alpha_tunnel.rs`, `rung_schedule::LEVELS = 10` | Generalize the tunnel from rung lanes to a hypothesis budget, or add a second tunnel kind |
| O6 | layer storage | dirty-byte delta, persisted, merged on read | overlay row = a full 512-byte `NodeRow`, ephemeral and discardable whole ("ephemer daneben, verwerfbar") | `alpha.rs` module doc | Persist validated layers only (through the split-tunnel append), keep unvalidated layers ephemeral |
| O7 | F/C encoding | 8 bits: linear, log/Shannon, or Moore DoF | linear u8 (`/255`, round half away from zero) | D-RPF-CONF-0, D-RPF-P5 | Chosen by the calibration (D-HL-6), not by taste |
| O8 | "Activation", "novelty", "belief update" meanings | stored in bits | no home; G8 of the coresearch entry | | Resolved if readings are per-classid ClassView declarations (§5 U1) |

---

## §5 — The proposed unification (HYPOTHESIS until its probe runs)

**U1 — Opaque bytes, readings declared per class.** The design's "opaque
bytes, meaning in the contract" and the code's "instruction word with
per-field contracts" are compatible if the reading of bits 40..63 is a
declaration resolved through the classid's ClassView, the same way
V3 facets work. The current ISA reading becomes one declared reading
("isa-v2"). The design's reading (activation, Staunen, breadcrumb) becomes
another. A class picks one. A word read under an undeclared reading is
refused. This turns O1, O2, O3 and O8 from conflicts into two declarations,
and keeps `learn`'s plasticity gate correct under its own reading.
Precedent: D-GSO-7a declares the band reading per class and refuses the
historical reading on the same bits.

**U2 — Merge-on-read is the wide accumulator.** D-RPF-P5 E2 measured that
writing each revision back into 8 bits stalls at c = 244 after about
20 unit observations. BF16 practice (1905.12322) and the design's own
"sparse write, merge on read" give the same remedy: deltas accumulate in a
wide fold (`Σw`, `Σw·f` in integers) and the result is rounded into the
byte once, at commit. The stall is then a property of in-place revision only,
and disappears by construction. This is also what D-RPF-4 needs; its blocker
is the missing LUT-weighted strided sum (coresearch G3).

**U3 — A layer is a hypothesis; a validated layer is a commit.** Generalize
`AlphaTunnel` from 10 rung lanes to a budget of up to 64 hypothesis lanes over
one `AlphaAllocation`. An unvalidated layer stays ephemeral (as today). A
layer that clears the validation ladder is composited by one fold and written
through the split-tunnel append: one Lance version per batch, never updated
in place. A failed hypothesis is a dropped layer and has touched nothing.

**U4 — Delta, not row.** A layer record is `(NodeGuid or row id, field mask,
dirty bytes)`, not a 512-byte row. This is the open "SparseDelta" of the Quack
storage-portability entry.

**U5 — Two masks before any decode.** `RowAddrMask` (Lance read selection) ∧
the per-layer `AlphaMask` (thinking selection) decide which rows are read at
all. Both are already word-level bitmaps.

**U6 — The layer id is the independence stamp.** D-RPF-P5 E3 (duplicates
inflate confidence) and the AriGraph no-stamp gap (R16) are the same defect:
revision cannot tell independent from repeated evidence. A delta carries its
layer and its saccade sequence (`AlphaStamp { cycle, seq, rung, visits }`
already exists). Cumulative fusion applies across independent layers;
repeated evidence within one layer is averaged or ignored. This answers
coresearch G5 ("where the stamp lives") without a CE64 field.

**U7 — Meta-awareness is mask arithmetic over layers.** Per node, the set of
layers that wrote is a 64-bit word. "How many hypotheses touch this",
"where do hypotheses 3 and 7 both claim", "what changed between two
snapshots" are popcount, AND, XOR: word-folds at ≈ 0.6 ns, independent of how
many operations a layer folded in.

**U8 — One truth quantiser, declared.** Coresearch T5/G2 and D-RPF-CONF-0
(24 definitions → 7 laws → 4 behaviours) ask for one declared encode/decode/
rounding/saturation contract. U2 makes the rounding happen once, so the
declaration has one place to apply.

**U9 — One belief arena.** `planner::nars::belief` and `deepnsm-v2/src/belief.rs`
duplicate the arena; AriGraph's `TripletGraph` is a fourth NARS
implementation with three semantic gaps (NAL ledger R8, R16, R17).

---

## §6 — How to validate: Narsese statistics

The rule: the binary encoding is right if, on the same inputs, it agrees
with literal Narsese run by an independent engine, to a measured tolerance.

1. **Oracles.** OpenNARS for Research and ONA, pinned and built per
   `.claude/knowledge/nal-recipe-audit/oracles.md`. `RunNal.java` runs a
   `.nal` file and exits non-zero on any failed expectation.
2. **Two arms, same inputs.** Arm L: the Narsese sentences through the
   oracle. Arm B: the same evidence through CE64 bytes, deltas merged on read
   (U2). Both arms log one row per derived statement: `(statement, f, c)`.
3. **Statistics** (`jc::reliability`): ICC over `f` and over `c`;
   Spearman over the ranking of derived statements by expectation;
   Cronbach α across encodings. Record the absolute code error alongside;
   a high ICC with a systematic one-code bias is still a finding.
4. **What is swept:** F/C encoding (linear, log-evidence, Moore pair); the
   hydration budget split (how many of 64 lanes "basic" hydration may use);
   the validation-ladder rung at which a layer may commit; in-place revision
   vs merge-on-read.
5. **Kill conditions** (pre-registered; full list in the plan):
   merge-on-read does not beat in-place revision on long chains ⇒ U2 is
   cosmetic; log encoding does not beat linear ICC by a declared margin ⇒
   stay linear; layer stamping does not remove D-RPF-P5 E3's inflation ⇒ U6
   is wrong.
6. **Falsifiability.** Every guard written for this gets a can-fire and a
   can-stay-silent test on non-trivial input (lance-graph `CLAUDE.md`
   § falsifiability rule).

---

## §7 — How to expand the substrate

- **Value-tenant room.** `ValueTenant` runs 0..20 today. Ordinals 21 and up
  are unassigned. The cap (32, per the design) is not confirmed in code: check
  `canonical_node.rs` and `tenant_counter.rs` before assuming it. Parallel
  encodings for the calibration belong in probe-local storage first, a tenant
  only after the calibration picks a winner (D-RPF-8 forbids a new tenant
  inside the D-RPF plan).
- **NAL-3..6 need compound terms.** Terms are atomic u16 ids and a statement
  cannot be a term (NAL ledger R10). Everything above NAL-2 waits on that.
- **NAL rules as loco ops.** The 256 palette is full; NAL rules go to the
  `ogar-loco` domain range (ledger R18).
- **Receipts give NAL-9.** `run_recipes` returns executed steps and
  `wave_dispatch` drops them (ledger R7). With receipts, MCP/SSR/ASC can judge
  their own past, which is NAL-9's "wonder/doubt" substrate.
- **Gaps start searches.** A pothole (`Refusal::Ungrounded`) or a
  `ReasoningGap` is a known unknown; today both are dropped (ledger R19).

---

## §8 — Loose ends, everywhere (index; details in the cited source)

**NAL ledger** (session artifact; reproduced in the plan §Loose ends):
R1 RCR abduction attaches results to the reversed statement · R2 commit the
tactic oracle test · R3 exemplification, conversion, comparison, resemblance
missing · R4 two non-NARS revision laws still live (NarsTables B1,
`grammar::revise_truth`) · R5 `TruthValue::new` accepts c = 1 · R6 dispatch
recipes 4, 6, 7, 8, 11 to the statement tactics · R7 receipt contract · R8 one
belief arena · R9 deriving rule in bits 46..49 (blocked on O1) · R10–R15
NAL-3..9 · R16 AriGraph stamp · R17 AriGraph deduction/contradiction semantics
· R18 NAL rules as loco ops · R19 gaps start a search.

**Coresearch (BF16 for thinking):** T1–T7, R1–R9, G1–G9 in
`entries/2026-10-08-coresearch-ce64-bf16-for-thinking.md`. Settled since:
T1/T3/R1/R3 by D-RPF-CONF-0; R2 (stagnation) by D-RPF-P5 E2. Still open:
G1 (c = 255 policy), G2 (normative quantiser), G3 (LUT-weighted strided sum),
G5 (stamp; see U6), G6 (W ownership), G8 (infographic meanings; see U1).

**ISA:** 11 of 16 inference codes refuse (`IsaFault::Unsupported`);
`counterfactual`/`intervention` always refuse; `forward` zeroes W and 59..63.

**Wiring:** `MailboxSoA::apply_edges` has no production caller; no path
connects `SpogTenants` / `AlphaTunnel` to a mailbox (D-ALPHA-G-0 §3);
multi-mailbox routing is the unbuilt W5.

**Scheduling:** round-robin is hard to beat under scarcity (0.964);
any Staunen-driven scheduler must clear it, with habituation and learning
progress (D-CE64-STAUNEN-0).

---

## §9 — Traps (each has cost a session before)

- Reading CE64 as a number format. Only F/C (bits 24..39) take numeric-format
  vocabulary; the rest are codes, masks, handles and tags.
- Thresholding bits 40..42 as a rung ordinal.
- Writing anything into bits 50..52 while `learn` gates on them as plasticity.
- Citing 0.6 ns as a join, a plane or a loop step.
- Revising F/C in place per observation and calling the result accumulated
  evidence (it stalls at c ≈ 244).
- Counting repeated evidence as independent (inflation 2–3.5×).
- Persisting an unvalidated layer: the overlay is discardable by design.
- Adding a tenant, a CE64 field or an opcode before the calibration measures
  that the existing homes are insufficient (`isa-anti-lasagne-warden`).
- Using grep absence as proof: list the search space or say "not found in X".
