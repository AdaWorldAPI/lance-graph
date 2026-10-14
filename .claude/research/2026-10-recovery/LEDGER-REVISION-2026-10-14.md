# Ledger revision — after #1495 … #1553 (2026-10-14)

Revises `CONSOLIDATED-LEDGER.md` (2026-10-11) in place of a new list. Every
O-id keeps its number; new items get N-ids, merged dots get K-ids. The
hypothesis-layer plateau (`ce64-reasoning-substrate-plateau.md` §4) has its
own O1..O8, which collide with this ledger's ids; this file calls them
**HL-O1..HL-O8**.

Method: lance-graph, OGAR and ndarray fast-forwarded to `main` (49 lance-graph
PRs merged after #1494, OGAR #357, ndarray +16; nothing rewritten). Read in full: PR arc
#1505–#1552, `revision-layer-invariants`, `isa-truth-exact-codes`,
`ce64-hypothesis-layers-plateau`, plateau §4/§5/§8, `nal-polyfill-v1` §D-NALP-3,
`inverted-temporal-ablation` entry and plan §0–§1. Haiku harvest (search and
maps) located candidates; every claim below that relies on a hit was read at
the cited line.

## 1. What merged since #1494, grouped by the thread it touches

| PRs | what | touches |
|---|---|---|
| #1495, #1496 | NAL-1..9 × 34-recipe audit with OpenNARS/ONA oracles; CE64 hypothesis-layer plateau and plan (D-HL-0..10, D-NALP-0..11) | O35 (schedule depth measured), K1, K2 |
| #1525, #1538 | `lance-graph-nal`: NAL queries as NodeGuid apertures; **revision as one grouped fold keyed by evidential base** | **K2/D2 core landed**, O8 (C3 arm), O25 |
| #1548 | exact `u8` truth codes for deduction / induction / abduction; Gadamer gate invariants G1–G7; Rubikon R1–R4 specs | O22, O23, O26, O33, **N1** |
| #1546 | `lance-graph-relalg`: relational IR normalized to a fixpoint before it runs | O32 (closure home), D-RA-4 |
| #1550, #1551, #1553 | D-HOLO: address-indexed algebra falsifiers (H6 interval closure holds) | O28 |
| #1549 | D-ORACLE-0 oracle fidelity matrix | N13 |
| #1497–#1500, #1543, #1545 | research: PLN×Shannon, Hyperon/MeTTa, NARS self-evidence, Rubicon volition, flow/cog, semantic Morton | O33, K4 |
| #1509, #1523, #1504–#1520, #1524, #1532 | SAG; class-method layer built then moved to backup (D-CMR reset) | D-SAG-3 builds on the pyramid (§5 K5) |
| #1505, #1526, #1529–#1531 | native target-cpu default, debug 0, production v4 tiers | build only |
| #1534, #1537, #1540, #1552 | r2il workspace member; core depends on the fold path; supervisor backoff; Haiku cards | infrastructure |

## 2. Closed ends

Kept from 2026-10-11: C1–C14 unchanged. New:

| id | item | how it closed | evidence |
|---|---|---|---|
| C15 | O25 "pin first" | already pinned before the ledger was written | `planner/tests/chain_confidence.rs:88-100` (`self_revision_raises_confidence_on_the_replay_path`, FAILS IF `replay_step` gains a stamp guard), commit `1bcb15b6`, 2026-10-08 |
| C16 | D2 core: a revision that refuses dependent evidence | `nal::Revision` classes each piece as independent / duplicate (counted once) / overlapping / no stamp; refuses the last two | `nal-polyfill-v1.md` §D-NALP-3; `lance-graph-nal/src/lib.rs:292-404`; 7 disable runs red |
| C17 | exact NARS truth codes (part of the quantiser debt in coresearch G2 / plateau U8; not this ledger's O7) | deduction / induction / abduction exact over all 2³² inputs | #1548 entry, `truth_exact_probe` |
| C18 | Gadamer invariants as contract | G1–G7 TEST-PINNED, G6 makes `is_acceptable()` the commit gate | `revision-layer-invariants.md` §1 |
| C20 | CodeRabbit "Staunen in the codebook table" | no change: board entries are append-only; the same-day addendum records the rename to Astonishment | `2026-10-11-maturation3-ladder-probe.md:76-80` |
| C21 | O9 resident-cycle operand design | unchanged, stays settled | — |

## 3. Excluded

X1–X7 unchanged. New:

| id | item | why |
|---|---|---|
| X8 | NAL as a core module (D-FOLD-CORE-c) | dropped in #1537; `lance-graph-nal` is the authoritative NAL crate |
| X9 | a second substrate for the exact truth law | #1548 DECISION: it lives in `causal_edge::isa::truth` |
| X10 | ONA as a runtime | pinned oracle only (#1536) |

## 4. Open ends — revised, deduplicated

Status: OPEN / DECISION / MERGED-INTO. Fruit: L hours, H one PR with a probe,
Y heavy. **Agent** = the cheapest tier that may do the grindwork (Haiku =
harvest-scout / quote-harvester / census-counter / substrate-mapper /
spec-editor / guarded-executor under orchestrator supervision; S = Sonnet
worker; O = Opus main thread).

### Thread 1 — audit and hygiene

| id | item | status now | fruit | agent |
|---|---|---|---|---|
| O1 | board supersession of tenants 18–20 claims, #1427 label | OPEN, unchanged | L | Haiku spec-editor on an O-written spec |
| O2a | Moore tenant hygiene (`project_state5`, `tenant_update`, Energy4 name) | OPEN, unchanged (0 doc hits since) | L | S |
| O3 | `RelativeOffset` in `RegisterLaw` | OPEN, still at `register128.rs:228`, `hotplug.rs:274` | H | O decides, S edits |
| O7 | model-identifier trailer on commits | DECISION, unchanged: OGAR, tesseract-rs and stockfish-rs forbid model identifiers in committed artifacts; lance-graph's CLAUDE.md does not, and `LATEST_STATE.md:1748` leaves the workspace-wide policy to the operator. This session omits the trailer, which is a practice, not a rule | — | O |
| O4 | CE64 replay compatibility | DECISION, now part of **K1** (one version gate for every bits-40..63 reading change) | Y | O |
| O5→O6 | `lab.rs` into jc, then its five defects | OPEN, unchanged | H | S + guarded-executor |
| N3 | PR arc has no entries for #1494–#1504 (11 merged PRs) | OPEN | L | Haiku quote-harvester (PR bodies) → O writes |
| N4 | STATUS_BOARD has no rows for D-RESIDUE-FP-0, D-MATURATION3-0 | OPEN | L | Haiku spec-editor |
| N10 | `causal-edge` fails `clippy -D warnings` on six pre-existing lints; CI runs only `cargo test` there | OPEN | L | S, guarded-executor gate |

### Thread 2 — learning seam and evidence pooling

| id | item | status now | fruit | agent |
|---|---|---|---|---|
| O8 | F.2: HashMap learner vs quack `GroupAddr::Pair` + SumI32 | OPEN, **narrowed**: arm C3 (echo with and without dedup) is now answerable by `nal::Revision`'s duplicate class instead of a new dedup | L | S |
| O25 | `replay_step` revises every hop with no evidential base | pin DONE (C15); **fix OPEN**, with two shipped patterns to reuse: refuse (NAL) or choice (`BeliefArena::revise_at`) | H | O picks refuse vs choice, S implements, the pinned test flips deliberately |
| N6 | exact revision / synthesis: both-at-255 packs `(0, 0)` (pinned known defect) | OPEN (#1548) | H | O |
| N8 | `nal::Choice` returns the maximum confidence, not the belief (no argmax fold) | OPEN (#1525) | H | S |
| N7 | R2IL `RANGE` takes `u8` immediates: no range past row 255 | OPEN, needs an `ogar-r2il` change | Y (cross-repo) | O |

### Thread 3 — resident Moore × CE64 cycle, and the upper half of the word

| id | item | status now | fruit | agent |
|---|---|---|---|---|
| O10 | op read from bits 46..49 vs op from the recipe | MERGED-INTO **K1** (= HL-O1) | — | — |
| O11 | a `SlabReading` variant for Moore lanes | OPEN: still four variants (`hotplug.rs:231-258`) | H | O designs, S edits |
| O12 | the resident-cycle seam test | OPEN, can start test-local | H | S |
| O13 | phase / activation rule | DECISION, unchanged | — | O |
| O14 | freeze-only plasticity | MERGED-INTO **K1** (= HL-O2 + the pyramid's migration + N1) | — | — |
| O17, O18, O19 | Experiment B, diagonal ownership, Experiment E | OPEN, after O12 | Y/L/H | — |
| N1 | `syllogize` writes bits 50..52 as `0b111` and 53..63 as 0; `execute` writes W and 59..63 as 0. A derived conclusion inherits none of its premises' hydration, witness or maturation; under the pyramid reading every syllogism would read as Astonishment | OPEN (#1548 entry, `isa::contracts::{FORWARD, SYLLOGIZE}`) | part of K1 | O |

### Thread 4 — event detection

| id | item | status now | fruit | agent |
|---|---|---|---|---|
| O15 | Experiment A (detection quality) | DECISION, unchanged | H | — |
| O20 | Cascade Stroke 1 drops unresolved candidates | OPEN in this ledger; **verify** against ndarray `main` (a Stroke-1 change was worked on in an earlier session) | L to verify | Haiku harvest-scout over ndarray `src/hpc/cascade*` |
| O21 | `meta_prefilter` returns row indices, not a mask | OPEN (still 2 code hits) | L | S |

### Thread 5 — reasoning: hydration, ablation, Gadamer, intention

| id | item | status now | fruit | agent |
|---|---|---|---|---|
| O22 | nothing computes `CounterfactualVerdict` | OPEN, verified: only `revision.rs` and `kanban.rs` tests construct `Necessary`/`Dispensable`. Now gates G6 *and* D-SAG-CF-1 W6 | L (one-claim producer) | S |
| O23 | TruthOnly → verdict | DECISION, **now with a recommendation from two independent sources**: D-SAG-CF-1 F8 requires TruthOnly to stay distinguishable from Inert and Necessary → a fourth reading, not a remap | — | O |
| O26 | Intentional → (Do, Causes) | OPEN; the "pin" at `causal_bridge.rs:450` asserts the defect **as correct**, so a fix must flip a passing test. Target now has a home: Rubicon (an intention is a commitment phase; R3 "a goal is never revised by counting") | L (re-pin as documented defect) / DECISION (target) | Haiku spec-editor for the re-pin text; O for the target |
| O27 | `first_divergence` unused in production; no cone | OPEN | H | S |
| O28 | 64 cut scenarios as planes need a population-cache descriptor | OPEN, **new evidence**: D-HOLO H6 shows symbolic interval closure is the cheapest arm in all 24 cells; a cut set that is an interval needs no plane at all | H | O |
| O29, O30, O31 | horizon identity revision, horizon persistence, eligibility masks | OPEN, unchanged | Y/H/H | — |
| O32 | deductive closure over hydrated relations | OPEN, **candidate home**: `relalg::normalize` fixpoint + D-RA-4 bounded recursion | Y | O |
| O33, O34 | teleology evidence classes; `learning::scm` unconnected | OPEN; O33 joins K4 | Y | O |
| O35 | recipe schedule depth 1 | MEASURED in #1495 (`knowledge/nal-recipe-audit/schedule_depth.md`); no action proposed | — | — |
| N9 | Rubikon R1–R4 tests (goal revision refused in `CognitiveWork`; agreement as `HorizonFusion`) | OPEN (#1548 entry §3) | L for R1 | S |
| N13 | deepnsm-v2 `answered_questions` returns a Pearl `CausalMask` for syntactic S/P/O completion | OPEN, marked BLOCKING in `d-oracle-fidelity-matrix-v1.md:68` | L | S |

### Thread 6 — the pyramid, residue and frontier (this session's)

| id | item | status now | fruit | agent |
|---|---|---|---|---|
| N11 | promote Maturation3 beyond test-local | DECISION = **K1**; D-SAG-3 asks the same question (`sag-wisdom-frontier-ona-hyperon-v1.md:180`) | Y | O |
| N12 | production feedback of resolved relations into `hydrate` | OPEN, needs K2's root id | H after K2 | O |
| N14 | Haiku cards registered only for sessions started after #1552; a broad 24-pattern brief returned counts without hits | process note | — | brief one question per card run |

## 5. The dots, re-drawn

**K1 — one decision about the upper half of CE64 replaces five.** HL-O1 /
O10 (46..49: opcode or activation), HL-O2 / O14 / N11 (50..52: plasticity,
Staunen or maturation), HL-O3 (53..58: W), N1 (derived edges zero their
witness and epistemic bits and set plasticity to `0b111`) and O4 (replay of
words written under an older reading) are one question: *who declares what
bits 40..63 mean for a class?* The plateau's U1 answers it the same way V3
facets and D-GSO-7a already work: the ClassView declares a reading; the ISA
reading becomes `isa-v2`; a pyramid reading is a second declaration; an
undeclared read is refused; replay goes through a version gate
(`I-LEGACY-API-FEATURE-GATED`). The test-local `M3Reading::Experimental` in
`maturation_ladder.rs` is that mechanism in miniature: it refuses every
existing `EdgeProvenance`. Why now: N1 makes the cost of not deciding
concrete — the moment any class declares the pyramid reading, every
`syllogize` output is born as Astonishment.

**K2 — the provenance identity exists four times; the work is to make it one.**
`nars::belief::Stamp(u64)` (disjoint → revise, overlap → choice), `nal`
`Evidence { stamp: &[u64] }` (refuse overlap and no-stamp), contract
`revision`'s `independent_roots` / `inherited_roots` masks (Gadamer), and the
maturation probe's `roots: u64` + derivation lineage all answer "which
independent sources does this rest on". What is still missing is the same
in all of them: `replay_step` uses none (O25), a CE64 derived edge carries
none (N1), and nothing maps a root to the split-tunnel log record (the
original D2). D-HL-3 is the probe that decides the record shape (U6: the
stamp lives in the delta record, not in the CE64 word). Fixing O25 with the
NAL policy is the smallest step that makes two of the four agree.

**K3 — the ablation verdict is one producer away.** G6 makes
`is_acceptable()` the commit gate; nothing computes its input (O22). Pearl
reactions, `first_divergence` (O27) and the pyramid's ablation impact are
three readouts of the same cut. One function from (cut, two traces) to
`CounterfactualVerdict` closes O22, gives D-SAG-CF-1 W6 its admission
policy, and needs O23 decided first only for the TruthOnly case.

**K4 — intention is not cause, and now it has a place to go.** O26
(`Intentional → Causes`), O33 (no teleology evidence class), D6 (purpose is
never derived), the Rubikon spec R3 (a goal is never revised by counting) and
`rubicon-volition-v1.md:87` (revising a belief can leave a commitment active)
point the same way: an intention maps to a commitment in the kanban phase
machine, with its own evidence class, never to an `EdgeType::Do / Causes`.

**K5 — residue, frontier and calibration are one loop.** D-RESIDUE-FP-0
proves "computation alone cannot lower the residue". D-SAG-3 asks whether a
stage-0 anchor exposes a frontier of consequences, and D-SAG-3.1 asks for
incremental truth maintenance checked against a full-recompute oracle after
every delta — the fixed-point test generalized to deletions. D-HL-4 (Narsese
run in parallel) is the external oracle for the same loop. All three already
keep the fences the pyramid set (no production 50..52, no 59..63 as desire,
no loop as independent support).

## 6. Fruit, in order

**Low (hours, no operator):**
1. N3 + N4 board hygiene — Haiku quote-harvester pulls #1494–#1504 PR
   bodies, the orchestrator writes the arc entries, spec-editor adds the two
   STATUS rows.
2. O22-min — one-claim producer: LoadBearing → Necessary, Inert →
   Dispensable, TruthOnly → NotRun; first end-to-end `is_acceptable()`.
3. O26 re-pin — rewrite the `causal_bridge.rs:450` assertion as a documented
   defect (FAILS IF the mapping is fixed) without changing behaviour.
4. N9 R1 — a goal revision attempted in `CognitiveWork` is refused.
5. N13, N10, O21 — each a small Sonnet edit behind a guarded-executor gate.
6. O20 verify — Haiku harvest over ndarray cascade.

**High (one PR with a probe, or one answer):**
7. O25 fix (K2 step 1) — `replay_step` refuses overlapping evidence the way
   `nal::Revision` does; the pinned test flips with a recorded reason.
8. D-HL-3 — per-delta evidential base on D-RPF-P5's duplicate fixture.
9. O27 + K3 — divergence cone feeding the verdict producer.
10. N6, N8 — exact revision at 255, argmax fold for `Choice`.
11. O11 → O12 — Moore `SlabReading`, then the resident-cycle test.

**Heavy (design, cross-repo or open science):**
12. K1 — the per-class reading of bits 40..63 (absorbs O4, O10, O14, N1, N11).
13. N7 — widen R2IL `RANGE` past 255 in OGAR.
14. O32 on relalg, O29, O33/O34 (K4), D-HOLO H11/H12.

## 7. Decisions only the operator can make

| id | question | suggestion |
|---|---|---|
| K1 | Declare bits 40..63 per class through the ClassView (U1)? | yes; it turns five conflicts into declarations and keeps `learn` correct under `isa-v2` |
| O25 policy | On overlapping evidence in replay: refuse (NAL) or choice (`BeliefArena`)? | refuse in replay (a replay that silently picks hides the overlap) |
| O23 | TruthOnly: fourth reading or remap? | fourth reading (D-SAG-CF-1 F8 requires it distinguishable) |
| O26 | What does "A intends B" map to? | a Rubikon commitment with its own evidence class, never Causes |
| O13 | phase / activation rule | unchanged; needs your rule |
| O15 | go for Experiment A? | unchanged |
| O7 | Is "no model identifier in committed artifacts" workspace-wide (then lance-graph's CLAUDE.md should say so), or per repo? | make it workspace-wide and write it into lance-graph's CLAUDE.md, so commits and harness trailers stop disagreeing |
