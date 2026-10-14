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

## Delta granularity: 16-byte slot or 512-byte row (pre-registered 2026-10-14)

**Question (operator, 2026-10-14).** Should a thought's delta store the whole
512-byte row image or only the changed 16-byte slots, when up to 10^9 folds
land in up to 64k rows of 512 bytes? Written before the probe exists; a change
after the run is a new dated section.

**Arms.**
- R (row image): per dirty row, the full 512-byte image, as `cycle_sink`
  writes today.
- S (slot delta): per dirty row, a 32-bit slot mask plus the changed 16-byte
  slots, packed. S-known: the writer knows which slots it wrote. S-diff: the
  mask is found by comparing the new row with the base.

Both keyed by the same row id and merged on read against the same base.
Folds accumulate wide in a resident accumulator per (row, slot) and are
rounded once at seal (U2); the fold path is identical in both arms and is
measured once to show it.

**Grid.** 65,536 rows; slots written per dirty row k in {1, 2, 4, 8, 16, 31};
dirty fraction d in {1 %, 10 %, 100 %}; 10^9 folds.

**Metrics.** v4 arm only, tier printed by the probe, ns per element, median
of 5 runs.
- M1 bytes stored per seal (counted).
- M2 ns per fold.
- M3 seal ns per dirty row (R, S-known, S-diff).
- M4 merge-on-read ns per row: full row and one slot.
- M5 resident bytes for 64 tables (computed, not allocated).

**Decision rule.**
- S is the default if at k = 1, for every d: M1(S) <= M1(R) / 16,
  M3(S-known) <= M3(R), and M4 one-slot(S) <= 2 x M4 one-slot(R).
- Crossover k*: the smallest k where S's M3 + M4(full row) exceeds R's. A
  table whose thought writes k >= k* slots per row stores row images. No
  crossover up to k = 31 means S throughout.
- Not measured here: Lance append bytes and time (in-memory probe). Both arms
  make one Lance write per seal by construction.

**Result (MEASURED 2026-10-14).** `delta_granularity_probe`, built with
`rustc -O -C target-cpu=native`, printed tier `v4 (avx512f)`; 4-core VM, two
full runs. 5 tests; 5 disable runs red (read without overlay, store every
slot, diff ignoring the base, slot offset from 0, store an empty mask).
- M1, counted: S/R = (8 + 16k) / 516, from 0.047 at k = 1 to 0.977 at k = 31.
- M2: 2.93 ns/fold over 1 MiB of accumulators (k = 1), 13.0 and 15.2 ns/fold
  over 31 MiB in random order, 4.64 ns/fold over the same 31 MiB sorted by
  accumulator. 10^9 folds take 3 to 15 s; sealing 64k rows takes 3 to 35 ms.
  The fold cost does not depend on the arm. It depends on accumulator
  footprint and order.
- M5, computed: 64 tables with every row dirty hold 2.016 GiB as row images,
  0.094 GiB as slot deltas at k = 1 and 1.969 GiB at k = 31.
- Rule at k = 1: met in both runs. M1 0.047 <= 0.0625; M3 S-known below R in
  all 6 cells; M4 one-slot S/R between 0.66 and 1.97 (closest: 1.97 at
  d = 10 %, run 2).
- k*: not stable. Run 1: 31, 31, 16; run 2: 16, 8, 16 (d = 1, 10, 100 %).
- Full-row reads cost 1.1 to 4.3 times more with S in every cell; one-slot
  reads are equal within noise at d <= 10 %.
- Caveats: M3 at d = 1 % includes allocating the 256 KiB row index per seal
  for 655 rows, so it is not a per-row cost, and it varies up to 6 times
  between runs. The k* sum adds one seal to one read, which is the cost only
  when each dirty row is read once per seal. Lance append not measured.

## Lane-tables addressed by the NodeGuid trie (pre-registered 2026-10-14)

**DECISION (operator, 2026-10-14).** One thought owns up to 32 lane-tables,
table position = lane/tenant position: 0 identity (the thought's coverage,
key immutable), 1 edge, 2..31 value. Up to 64 thoughts, so up to 2048
lane-tables. Addressing is the NodeGuid trie, not a row index. A read is a
merge on read over a selection (prefix aperture x thought mask x lane mask),
keyed by the immutable NodeGuid: no join, no fold.

**Question (operator: "measure both").** Per-lane tries, or one trie per
thought with a lane mask at each leaf? Written before the probe exists; a
change after the run is a new dated section.

**Arms.**
- P (per-lane): each (thought, lane) has its own trie over the NodeGuids it
  wrote and its own packed payload in leaf order.
- S (shared): each thought has one trie over every NodeGuid it wrote; each
  leaf carries a 30-bit lane mask; payload is packed per leaf in lane order,
  located by a per-leaf start plus popcount of the lane mask below the lane.

**Trie, identical in both arms.** Level-ordered, fan-out 16, one 16-bit child
mask per node; child index = first child of the node + popcount(mask below
the nibble). Path = classid (8 nibbles) + HEEL (4) + HIP (4) = 16 levels.
Pointer-based layouts are out of scope for this probe.

**Reads, both lockstep over the selected tries.**
- Point: one NodeGuid; descend all selected tries level by level, drop a trie
  when its child bit is clear; overlay the selected lanes of the surviving
  tries in thought order (higher thought index wins a shared lane) onto the
  base value.
- Range: every NodeGuid under a 9-nibble prefix (classid + top HEEL nibble,
  1/16 of the universe); depth-first lockstep enumeration, child set at each
  node = OR of the alive tries' masks.

**Grid.** Universe 65,536 NodeGuids, one classid. Thoughts T in {1, 8, 64};
each covers a fraction d in {1 %, 10 %, 100 %} of the universe and writes
w in {1, 4, 30} lanes per covered node, the lanes drawn per node. Lane mask
selects 1 lane or all 30. Cells with more than 32 M written lane-cells are
skipped (only T = 64, d = 100 %, w = 30).

**Metrics.** v4 arm only, tier printed by the probe, median of 5.
- Q1 bytes: trie structure + payload.
- Q2 point read: ns per read and ns per trie-level step.
- Q3 range read: ns per NodeGuid emitted.
- Correctness, not timed: P, S and a brute-force oracle return identical
  bytes for every point and range read in the tests.

**Decision rule.** Reference cell T = 64, d = 10 %, w = 4, all 30 lanes
selected. Choose P if, at the reference cell, its Q2 and Q3 are each within
1.5 times S and its Q1 within 2 times S; otherwise S. Every other cell is
reported and does not change the choice. If neither arm reads a point in
under 1 microsecond at the reference cell, report that as a failure of the
nanosecond target, not as a win for either arm.

**Result (MEASURED 2026-10-14).** `lane_trie_probe`, built with
`rustc -O -C target-cpu=native`, printed tier `v4 (avx512f)`; 4-core VM, one
run. The run reached the 30-minute limit before its last two rows (T = 64,
d = 100 %, w = 4, all lanes; w = 30 was skipped by the cap); every other cell,
the reference cell included, completed. 5 tests (both arms against a
brute-force oracle for point and range reads, precedence, drop on an
uncovered key, rank order, empty lane-tables); 5 disable runs red.

| reference cell T = 64, d = 10 %, w = 4, all lanes | P (per-lane) | S (shared) |
|---|---|---|
| Q1 bytes | 79,270,480 | 41,514,192 |
| Q2 point read | 569,240 ns | 6,363 ns |
| Q2 per trie-level step | 26.4 ns | 8.1 ns |
| Q3 range, per emitted NodeGuid | 11,068 ns | 1,078 ns |

- Rule: P is 89 times slower on Q2, 10 times on Q3 and 1.9 times larger, so
  **S is chosen**. Neither arm reads a point under 1 microsecond at the
  reference cell: **the nanosecond target fails for both**.
- P is never cheaper with all lanes selected. With one lane selected it reads
  points at about S's cost (3.3 to 7.5 microseconds at T = 64) and is faster
  on ranges, because S walks every covered node to emit the few that carry
  the lane.
- Cost is set by tries descended times levels: at T = 64 a point read takes
  up to 64 x 16 steps at 5 to 10 ns. The first 8 levels are the classid
  chain, identical in every trie, so about half the steps carry no
  information.
- Not decided here: a single trie over all thoughts with a 64-bit thought
  mask per node (one descent instead of 64), and taking the classid levels
  out of the trie. Both change the arms and need a new pre-registration.

## Identity table and thought sessions (pre-registered 2026-10-14)

**DECISION (operator, 2026-10-14).** Measure the two follow-ups from the
lane-trie result (one shared trie; classid out of the trie), and add: the
identity is a dense table of 64k rows x 32 lanes (lane 0 key, 1 edge, 2..31
value; 64k x 512 B), addressed by the NodeGuid's rank in its class trie; the
64 thoughts are session addresses for temporary materialization, not
storage. Written before the probe exists; a change after the run is a new
dated section.

**Arms.**
- S16: the previous winner, one trie per thought, 16 levels (classid 8 +
  HEEL 4 + HIP 4). Re-run as the baseline.
- S8: S16 with the classid taken out (one trie per class): 8 levels.
- M16, M8: one identity trie per class (16 or 8 levels) resolving a NodeGuid
  to its identity row; thought overlays hang off the row: one 64-bit
  written-by thought mask per row, then per (row, thought) entry a 30-bit
  lane mask and the packed cells, entries sorted by thought. One descent per
  read whatever the number of thoughts. Internal trie nodes carry no thought
  mask: a prefix maps to a contiguous row range, so the range read scans
  rows and skips each row whose written-by AND selection is 0.

All arms return the same merged delta (30 value lanes plus the lane mask)
and are checked against the brute-force oracle; a higher thought index wins
a shared lane.

**Session materialization (M8 only, reported, no decision attached).**
Materialize a session's view into a temporary 512-byte-row buffer over a
prefix range (1/16 of the class): identity row copied, then the session's
overlays applied. Session = one thought (s in 0..63), and the merged view of
all 64. Metric: ns per materialized row; bytes for all 64 sessions over the
full class are computed, not allocated.

**Grid, metrics.** As in the lane-trie pre-registration: universe 65,536,
T in {1, 8, 64}, d in {1 %, 10 %, 100 %}, w in {1, 4, 30}, lanes 1 or all,
32 M lane-cell cap; Q1 bytes (the 32 MiB identity table is shared by every
arm and reported once), Q2 point read ns and ns per descent step, Q3 range
ns per emitted NodeGuid; v4 only, tier printed, median of 5.

**Decision rule.** Reference cell T = 64, d = 10 %, w = 4, all lanes. Among
the arms whose Q3 is within 1.5 times the best Q3 and whose Q1 is within
2 times the best Q1, choose the lowest Q2. Nanosecond target: a point read
under 1 microsecond at the reference cell, reported pass or fail per arm.

## Dense tile: the Morton code is the row (pre-registered 2026-10-14)

**Input (operator, 2026-10-14):** the Morton tenant already has Cartesian
addressing (`lance-graph-contract::morton8x8`, D-MORTON-0: checked 8:8
arithmetic on the Z-order code, one nibble = one 4 x 4 refinement, fan-out
16). 64k identity rows = 256 x 256 = one Morton tile, so for a class whose
HEEL tile is dense the row is the HEEL code: no trie, no rank. A prefix
aperture (top nibble p) is the contiguous row range [p x 4096, (p+1) x 4096).
Written before the arm exists; it runs after the identity-session run, on
its own, so the two do not share the CPU.

**Arms, on a dense universe** (one classid, all 65,536 HEEL codes, HIP 0):
- M8: the identity trie of the previous section (8 levels, HEEL + HIP).
- D: row = the key's HEEL code, after checking the classid; no descent.
Overlays, merge, materialization and the oracle are the ones of M.

**Grid, metrics.** T in {1, 8, 64}, d in {1 %, 10 %, 100 %}, w in {1, 4, 30},
lanes 1 or all, same cap; Q2 point read ns, Q3 range ns per emitted
NodeGuid, and session materialization ns per row at the reference cell;
v4 only, tier printed, median of 5.

**Decision rule.** If D reads a point under 1 microsecond at the reference
cell (T = 64, d = 10 %, w = 4, all lanes), dense class tiles use D and the
trie is kept for sparse classes only. Otherwise both stay open and the
overlay cost is the next thing to measure.

**Results (MEASURED 2026-10-14)** for the two sections above.
`identity_session_probe`, built with `rustc -O -C target-cpu=native`, printed
tier `v4 (avx512f)`; 4-core VM, one run each, both complete (only the
pre-declared T = 64, d = 100 %, w = 30 cell skipped). 6 tests (every arm and
the dense arm against the brute-force oracle, prefix ranges, rank = row,
classid-out depth, sessions); 6 disable runs red (classid check, prefix row
range, thought selection in the merge, thought precedence, trie range end,
session writing the key lane).

Identity table and thought sessions, reference cell (T = 64, d = 10 %, w = 4,
all lanes), random sparse universe:

| arm | bytes | point read | range, per NodeGuid |
|---|---|---|---|
| S16 | 41.5 MB | 6,725 ns | 1,121 ns |
| S8 | 41.5 MB | 3,844 ns | 1,082 ns |
| M16 | 31.4 MB | 776 ns | 112 ns |
| M8 | 31.4 MB | 783 ns | 116 ns |

- Rule: M16 and M8 are the arms within 1.5 times the best range read; M16
  has the lower point read, by 7 ns, which is within run-to-run noise.
  **Chosen: the identity trie with overlays per row (M).** Nanosecond target
  (point read under 1 µs): **met by M16 and M8**, not by S16 or S8.
- Classid out of the trie: S8 is 1.75 times faster than S16 on points; M8
  and M16 are equal, because in M the merge of the overlays costs more than
  the descent.
- Sessions (M8): 105 ns per materialized row for a one-thought session,
  138 ns for the all-64 session. All 64 sessions over the whole class at
  once would take 2 GiB, which is why sessions materialize a prefix range
  and are temporary.

Dense tile, reference cell, dense universe (all 65,536 HEEL codes):

| arm | point read | range, per NodeGuid |
|---|---|---|
| M8 (identity trie) | 1,256 ns | 134 ns |
| D (Morton code is the row) | 558 ns | 82 ns |

- Rule: D reads a point under 1 µs at the reference cell, so **dense class
  tiles use D; the trie stays for sparse classes.**
- With few thoughts D reads a point in 16 to 60 ns (T = 1 and 8, d up to
  10 %). At T = 64 the merge over up to 64 entries per row is what remains;
  the overlay layout is the next thing to measure.
- Sessions on the dense tile: D 106 ns per row (one thought), 147 ns per row
  (all 64); M8 144 ns per row (all 64).

## Coverage orientation: per-thought masks or per-row words (pre-registered 2026-10-14)

**DECISION (operator, 2026-10-14).** Counting is popcount under an explicit
equivalence key, intersected with the aperture. The thought x identity
coverage is one 64k x 64 bit matrix (512 KiB per class) held in one of two
orientations; measure both on the four count queries before choosing.
Written before the probe exists; a change after the run is a new dated
section.

**Orientations, one dense class tile of 65,536 rows, 64 thoughts.**
- C (per thought): 64 masks of 65,536 bits, 1,024 u64 words each; the
  formula's M_t.
- R (per row): one 64-bit written-by word per row; what the identity-session
  and dense-tile arms already store.

**Queries,** for a thought selection T (u64) and an aperture (a contiguous
row range, as a Morton prefix gives):
- Q-ID, distinct NodeGuid: C = popcount(OR of M_t over T, in the aperture);
  R = number of rows with w AND T != 0.
- Q-TH, distinct ThoughtId: C = thoughts in T whose M_t has a set bit in the
  aperture; R = popcount(OR of w AND T over the aperture).
- Q-PAIR, (NodeGuid, ThoughtId) observations: C = sum of popcount(M_t) in
  the aperture; R = sum of popcount(w AND T).
- Q-PERSP, perspectives on one identity: C = |T| bit reads; R =
  popcount(w AND T).
EvidenceRoot is not a coverage query and is not measured.

**Also reported, no decision attached:** the cost of deriving one
orientation from the other at seal (transpose, ns per row).

**Grid.** Coverage independent per (thought, row) with density d in {1 %,
10 %, 50 %}; |T| in {1, 8, 64} (T a fixed random subset, all 64 for 64);
aperture 65,536, 4,096 or 256 rows. Every answer checked against a
brute-force oracle over the bit matrix.

**Metrics.** ns per query; for Q-ID, Q-TH, Q-PAIR also ns per aperture row
(the per-element figure); v4 only, tier printed by the program, median of 5.

**Decision rule.** Reference cells: d = 10 %, |T| in {8, 64}, aperture
65,536 and 4,096. If one orientation is within 2 times the best on every
query at every reference cell, only that one is stored. Otherwise both are
kept, each query routed to its faster orientation, and the second
orientation derived at seal. Nanosecond target: Q-PERSP under 10 ns, and
the aperture queries under 1 ns per aperture row, in the better
orientation; reported pass or fail.

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
