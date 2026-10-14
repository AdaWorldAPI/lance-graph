# CE64 hypothesis layers — v1

**Status:** PROPOSAL (D-HL-0..9). D-HL-0 is this documentation. Nothing else
is authorized: each later item needs its decision (D-HL-1) or its probe to
pass first. No CE64 bit, tenant, opcode or ABI change is authorized by this
plan.
**Date:** 2026-10-12.
**Plateau (read first):** `.claude/knowledge/ce64-reasoning-substrate-plateau.md`.
**Research map:** `.arxiv/2026-10_ce64_hypothesis_layers_research_map.md`.
**Parents:** D-GSO (global sudoku), D-RPF (resident projection fold),
D-CE64-STAUNEN/CYCLE/LOOP, D-ALPHA-G-0, D-NAL-AUDIT-0, coresearch
"CE64 as BF16 for thinking" (2026-10-08).

## Overview

The operator's design (plateau §1) treats reasoning as up to 64 parallel
hypothesis layers over an immutable address space. Each layer is a sparse
delta of opaque little-endian bytes keyed by `NodeGuid` / row id, merged on
read. A layer that clears a validation ladder is composited into the base
by one fold and one Lance append. Literal Narsese, run by an independent
engine, is the yardstick the binary encoding must agree with.

Most pieces exist (plateau §2). This plan connects them, in an order where
every step is a measurement that can kill the next.

## Checklist

- [x] **D-HL-0** Plateau, plan, research map, board entry (this PR).
- [ ] **D-HL-1** Decisions O1, O2, O3, O5, O6 (plateau §4). Operator.
- [ ] **D-HL-2** Merge-on-read vs in-place revision (U2).
- [ ] **D-HL-3** A per-delta evidential base as the independence test (U6).
- [ ] **D-HL-4** Narsese parallel calibration harness.
- [ ] **D-HL-5** Hypothesis lanes: `AlphaTunnel` with a lane budget.
- [ ] **D-HL-6** Two masks before decode (U5).
- [ ] **D-HL-7** SparseDelta probe record, persisted through the split tunnel (U4).
- [ ] **D-HL-8** Validation ladder: when a layer may commit.
- [ ] **D-HL-9** Hydration budget split sweep.
- [ ] **D-HL-10** NAL inference ISA: an `ogar-nal` loco vocabulary plus a mask-risc dialect.

## Order

```text
D-HL-2 ─┐
D-HL-3 ─┼─► D-HL-4 ─► D-HL-9
        │       ▲
D-HL-5 ─┴─► D-HL-6 ─► D-HL-7 ─► D-HL-8
D-HL-1 gates any stored-meaning change; D-HL-2..6 are probes and need no decision.
```

## Details

### D-HL-1 — decisions (operator)

Plateau §4, rows O1 (bits 46..49), O2 (bits 50..52), O3 (W ownership, the
coresearch G6), O5 (10 rung lanes vs a 64-lane budget), O6 (what persists).
The recommendation in plateau §5 U1 is to declare readings per classid
through the ClassView, so the ISA reading and the design reading coexist
behind a refusal for undeclared readings. Any stored-meaning change ships
behind a version gate (`I-LEGACY-API-FEATURE-GATED`).

### D-HL-2 — merge-on-read vs in-place revision (S)

- **Where:** `crates/causal-edge/tests/`, next to `d_rpf_p5_beta.rs`.
- **Arms:** (a) revise into the u8 byte per observation (today);
  (b) accumulate `Σw`, `Σw·f` in integers, round once; (c) exact rational.
- **Grid:** unit-evidence chains N = 1..1000 at observation strengths
  c_obs ∈ {1, 64, 128, 240, 250} (the E2b table).
- **Measure:** final code error vs (c); the N at which (a) stalls.
- **Kill:** (b) within 1 code of (c) on every cell and (a) not ⇒ U2 holds.
  (b) as far from (c) as (a) ⇒ U2 is cosmetic; record and stop.
- **Anti-vacuity:** at least one cell where (a) and (c) differ by ≥ 2 codes.

### D-HL-3 — a per-delta evidential base as the independence test (S)

- **Where:** same test file family.
- **Setup:** D-RPF-P5 E3's duplicate fixture (4 and 8 copies). Each delta
  carries `(layer, seq, evidential base)`, the base being a source bitset
  (`nars::belief::Stamp` shape). Fusion is cumulative only between disjoint
  bases; overlapping bases are not counted again. Layer membership plays no
  part in the decision.
- **The fixture must separate layer from provenance**, or it encodes the
  answer: it includes (a) one observation copied into two layers, which must
  count once, and (b) two independent observations inside one layer, which
  must count twice; plus the plain cases (duplicates in one layer,
  independents in distinct layers).
- **Kill:** the sd ratio does not return to within 10 % of 1.0 on all four
  cases ⇒ U6 is wrong. A rule keyed on the layer id instead of the base must
  fail cases (a) and (b); that arm is run as the disable.
- **Silence twin:** independent observations must still narrow the posterior
  exactly as today.

### D-HL-4 — Narsese parallel calibration harness (M)

- **Arm L:** `.nal` files run by the pinned OpenNARS for Research build
  (`.claude/knowledge/nal-recipe-audit/oracles.md`, `RunNal.java`).
- **Arm B:** the same evidence as CE64 deltas, merged on read (D-HL-2 (b)),
  computed by the scalar CE64 register methods.
- **Arm C (added by D-HL-10 / D-NALP-7):** the same Narsese text compiled to
  an `ogar-nal` program and run through the NAL dialect over mask-risc. It is
  a separate arm, not a replacement for B.
- **What the gate evaluates:** each of B and C against arm L (the oracle).
  B and C must also agree with each other, within the same tolerance, on
  every statement both derive.
- **Output:** one row per derived statement `(statement, f_L, c_L, f_B, c_B)`.
- **Statistics:** `jc::reliability::{icc, spearman, cronbach_alpha}`, plus
  absolute code error and its sign (bias).
- **Encodings swept:** linear u8 (today); log-evidence u8; the Moore pair
  reading of tenants 19/20, if D-HL-1 allows it.
- **Kill:** an encoding that does not beat linear ICC by the declared margin
  (set before the run, in the test file) is dropped.
- **Fixtures:** start from the upstream NAL-1/2 fixtures the audit already
  ran (nal1.*, nal2.*); add `revision_points.nal`.

### D-HL-5 — hypothesis lanes (M)

- **Where:** probe first (`cognitive-shader-driver/examples/`), not
  `contract/src/alpha_tunnel.rs`.
- **Shape:** a tunnel whose lane count is a parameter up to 64 over one
  `AlphaAllocation`; one writer per lane; `run_wave` and `run_wave_parallel`
  byte-identical (the existing tunnel's falsifier, kept).
- **Measure:** reserve cost per lane (must stay an empty `Vec`), claim cost,
  merge cost per touched row.
- **Kill:** reserve cost grows with lane count ⇒ the "64 for nothing" premise
  fails; report the slope.

### D-HL-6 — two masks before decode (M)

- **Where:** `lance-graph-benches/examples/` beside `lance_source_probe.rs`.
- **Projection first:** `AlphaMask` bits are `AlphaAllocation` ordinals and
  `RowAddrMask` holds Lance row addresses (fragment id << 32 | offset). Build
  an explicit `AlphaAddr` / `NodeGuid` → row-address map for the fixture and
  project the thinking mask through it before intersecting. A test pins the
  projection: every projected address resolves to the row whose key equals
  the alpha address, and a fragment boundary inside the selection is covered.
- **Arms:** unmasked scan; `RowAddrMask` only; `RowAddrMask ∧ project(AlphaMask)`.
- **Measure:** rows decoded, bytes read, wall time; answers must be equal
  across arms.
- **Kill:** the joint mask decodes no fewer rows than the read mask alone on
  the fixture ⇒ the thinking mask adds nothing at the storage layer.

### D-HL-7 — SparseDelta, persisted (M)

- **Record:** `(NodeGuid or row id, layer, seq, field mask, dirty bytes)`;
  the open vocabulary item in `entries/2026-10-05-quack-storage-portability.md`.
- **Path:** the split-tunnel append (one Lance version per batch, never
  updated in place), replay once, fold resident.
- **Falsifiers:** replay composite == in-memory composite, byte for byte;
  dropping one delta changes exactly the rows it touched; a delta with a
  field mask outside the declared reading is refused.

### D-HL-8 — validation ladder (M)

- **Candidate rungs:** hydrated (59..63 set) → witnessed (W breadcrumb) →
  settled (Staunen below a threshold, under whatever home D-HL-1 gives it) →
  committed.
- **Measure:** for each commit rung, agreement with the oracle (D-HL-4) and
  Lance writes per 10⁶ deltas.
- **Kill:** no rung beats "commit everything" on agreement ⇒ the ladder is
  decoration.

### D-HL-9 — hydration budget split (M)

- **Sweep:** k = lanes "basic" hydration may use, k ∈ {4, 8, 16, 32, 48, 60}.
- **Measure:** oracle agreement (D-HL-4) and lanes left for reasoning.
- **Output:** the curve; the knee is the recommended split.
- **Kill:** agreement is flat in k ⇒ the budget is not the binding constraint.

### D-HL-10 — NAL inference ISA (M, design first)

**Integration plan:** `.claude/plans/nal-polyfill-v1.md` (D-NALP-0..11) maps
this item across `lance-graph-r2il`, quack, mask-risc, `ogar-nal` and
`lance-graph-nal`. Where the two differ, that plan governs.

**Idea (operator, 2026-10-12):** make NAL's inference rules a small RISC-style
instruction palette with a compile-time polyfill. The abstraction gives
reusable patterns, execution costs what masking costs, and serialization
costs almost nothing, because a loco program is already a 512-byte node of
call bytes.

**This is a vocabulary on the existing IR, not a new IR.** The shape exists
three times:
- `ogar-ro` (relation predicates as a loco `Vocabulary`);
- `ogar-r2il` (an opcode arity table, pinned to its source enum by a drift
  test, minting nothing);
- `lance-graph-r2il::FoldDialect` (calls build mask ops; each fold runs one
  mask-risc `Program`).

`ogar-loco/src/nars.rs` already carries the 34 recipes as a vocabulary with
their Datapath / Control / Gate tiers. D-HL-10 adds the layer below them:
- `ogar-nal`: the NAL rules (deduction, induction, abduction, exemplification,
  conversion, comparison, analogy, resemblance, revision, choice, …) as an
  arity table in the loco domain range, selected by its own classid;
- a NAL dialect in lance-graph that lowers each call to mask-risc.

Ledger item R18 is this vocabulary.

**Naming.** "Thinking atoms" already names the 144 verbs at rung 2
(`.claude/v3/knowledge/persona-vs-rung-ladder.md`). Call these
**inference atoms**, the truth-ALU instruction set: verbs say *what* is
thought; inference atoms say *how truth moves*. A recipe (rung 3) is a
program over both.

**Relation to the CE64 strict decoder.** `isa.rs` already executes five NAL
codes per edge (+1 deduction, +2 induction, −1 abduction, +4 revision,
+5 synthesis). `ogar-nal` is the population form of the same rules. A
parity test pins each `ogar-nal` op, applied to one row, against the CE64
register method of the same name.

**What is cheap and what is not:**
- choice and threshold predicates: already mask patterns (D-RPF-0);
- revision over a population: one fold, `Σw` and `Σw·f`, which needs the
  LUT-weighted strided group-sum (coresearch G3);
- deduction, induction, abduction over a population pair statements on a
  shared middle term. That is a hop (mask-risc `Gather` over a foreign
  plane), not an element-wise mask. Measure it before calling it cheap.

**Blocked regardless of IR:** NAL-3..6 need compound terms (ledger R10).

**Gate:** D-HL-4's oracle comparison, with Narsese text compiled to an
`ogar-nal` program as arm C, a separate arm beside B (the scalar CE64
deltas). Both are scored against arm L, the pinned OpenNARS. **Kill:** a NAL op whose dialect lowering
disagrees with the OpenNARS output beyond the D-HL-4 tolerance is not
admitted. **Warden:** `isa-anti-lasagne-warden` must find no second IR and
no lowering into a representation with no second reader.

## Pre-registration — thought lanes as hypothesis tests (2026-10-14)

**Status:** PROPOSAL, written before any lane code or run. Decision O5 taken
by the operator on 2026-10-14: up to 64 thought lanes beside the 10 rung lanes.
Changing a hypothesis, metric, threshold or decision rule after a run is not an
edit of this section: it is a new dated section that says what changed and why.

**Unit of observation.** One fixture: a premise pair (or set) with its kind of
fact declared (math, episodic, agreement/disagreement) and its premises' CE64
bytes set, including bits 50..52, 53..58 and 59..63. The fixture set is
committed before the first run and is not filtered afterwards.

**Lane = treatment.** Each lane runs one thought on the same fixtures against
the same read-only base. A lane records its saccade (`AlphaOverlay`: where it
looked, in which order, revisits) and its conclusions as sparse
`delta = conclusion XOR base`. Lane identity is its own type, not a rung and
not a recipe id. External engines (ONA, OpenNARS) fill their lanes from
committed golden outputs; their derivation order is their saccade.

**Controls, run in every comparison:**
- Negative control: an identity lane that concludes the base. "Differs"
  against base must report 0 rows. If not, the comparison is broken and no
  other result of that run is read.
- Positive control: a lane that changes exactly one known field on a known
  set of rows. "Differs" must report exactly that field on exactly those rows.
- Permutation null for every agreement rate: the same lane's conclusions
  reassigned across fixtures at random (fixed seed); the observed agreement is
  reported beside the null distribution.

**Confirmatory hypotheses.** Deterministic ones are falsified by one
counterexample and need no statistics.

| H | Claim | Measure | Falsified when |
|---|---|---|---|
| H1 | No lane changes the base | hash of the base bytes before and after every run | any hash differs |
| H2 | Reserving a lane costs nothing | reserve cost per lane over 1..64 lanes | cost grows with lane count (report the slope) |
| H3 | Lanes are independent | parallel run vs sequential run | any byte differs |
| H4 | The current law loses hydration marks | conclusions' bits 59..63 from `execute`/`syllogize` vs `pearl::hydrate` on the same chain | (stated null: they agree) agreement on every fixture, i.e. no loss |
| H5 | NAL evidential-base overlap and the Gadamer gate agree on whether a conclusion may be revised into its own premise | admit / refuse per fixture, paired | any fixture where they differ (reported as a finding, not a failure) |
| H6 | The exact codes (lane 1) agree with OpenNARS better than the f32 path (lane 2) | per-statement code error, paired | lane 1 not better on the paired exact McNemar test |

**Candidate laws for a conclusion's marks** (lanes 9..12): (a) 0, today;
(b) `IndirectKnown × Open`; (c) weakest premise topology made indirect,
× `Open`; (d) `Unknown × Open`, hydrated afterwards. **Decision rule, declared
now:** among laws that never raise certification above what a Pearl operator
measured, choose the one with the highest topology agreement with the
`pearl::hydrate` lane; on a tie, the one that writes fewer fields. The
operator makes the final call from the table; this rule is the
recommendation.

**Statistics.** Exact counts with n for every rate. Paired comparisons (same
fixtures through two lanes) use the exact McNemar test. Significance level
0.01, Bonferroni-corrected over the number of paired comparisons, fixed here
before the run (H6 plus the six pairwise comparisons between laws (a)..(d)).
No confidence claim on dependent rows without citing the dependence
(`I-NOISE-FLOOR-JIRAK`).

**Exploratory, no decision attached:** where ONA's and OpenNARS's saccades
diverge (first differing `seq`), which lanes revisit and where, and what each
lane attended that no other lane did. Exploratory findings may motivate a new
pre-registration; they never confirm a hypothesis in this one.

## Placement and the first probe (2026-10-14)

**DECISION (operator, 2026-10-14).** Saccades go into up to 64 tables; each
table writes only its delta (sparse write) and is merged on read. Any change
inside a 512-byte row causes a write. A thought's delta is placed either
**globally**, in a table it owns alone, or **locally**, in one 16-byte slot
(a tenant-sized lane) of a table shared with other thoughts, to save tables
when a thought only needs one slot.

**Probe:** `cognitive-shader-driver/examples/thought_layers_probe.rs`
(`cargo test --example thought_layers_probe`). In memory only.
- Table = `AlphaOverlay` (saccades) + per-row `RowDelta`: the XOR of the
  slots that differ from the base, nothing else. Read = base XOR delta.
- A slot-placed thought may write and read only its slot; the key slot never
  changes; the 65th table is refused.
- MEASURED (`cargo run --example thought_layers_probe`): 30 slot-placed
  thoughts × 256 rows, one byte each, fit in 1 table and store 122,880 bytes
  against 3,932,160 for full rows.
- TEST-PINNED: 9 tests. Disable runs red: the key guard, the slot guard,
  storing every slot instead of the changed ones, and reading another
  thought's slots. Dropping the empty-delta check in `RowDelta::between`
  stays green because `restrict` already drops empty deltas (redundant, not
  a missing test).
- Not yet: oracles in lanes (H4..H6), persistence (D-HL-7), the comparison
  report over all lanes.

## Related fix-now items outside this plan

From the NAL ledger (`D-NAL-AUDIT-0` follow-ups), each S and independent of
D-HL-*:
R1 RCR abduction direction; R2 commit the tactic oracle test; R4 retire the
two non-NARS revision laws (needs coresearch G2); R16 AriGraph evidence stamp
(U6's evidential base is the general form); R17 AriGraph deduction and contradiction semantics.

## Boundaries

- mask-risc never executes CE64 instructions (D-RPF-8).
- No VSA on this path.
- No new tenant until D-HL-4 picks an encoding.
- No opcode added to the strict decoder by this plan.
- No `.0.0` pins or new dependencies; probes use existing crates.
