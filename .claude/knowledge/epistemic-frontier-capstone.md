# Epistemic frontier capstone — how CE64, the alpha channel, the 64 thought layers, NAL and the front ends connect

**READ BY:** any session touching CausalEdge64 bits 40..63, `contract::alpha` /
`alpha_tunnel` / `alpha_focus` / `spog_tenants`, the thought lanes (D-HL), the
givens layer, `lance-graph-nal`, `lance-graph-relalg`, `lance-graph-metta`,
the Hyperon / Atomese / Narsese front ends, the 34 recipes, the rung ladder,
or anything that says "witness", "provenance", "evidence", "G" or "NAL-k".
Read it before proposing a type, a field or a reference: most of what such a
proposal needs already exists and is listed below with its file.

**Status:** CURRENT-CONTRACT where a file and line are named, MEASURED where
a probe is named, DECISION where a date and BASIS are named, OPEN where it
says so. Nothing here authorizes a code change; each change has its own PR.
**Date:** 2026-10-17. **Entry:** `.claude/board/entries/2026-10-17-epistemic-frontier-clicks.md`.

---

## 0. The whole picture in one tree

```text
IMMUTABLE ADDRESS SPACE (the frame)
│  NodeGuid = address = distinguished name = HHTL Cartesian position
│  (canonical_node.rs; a selection is the 16-byte key with n leading bytes as aperture)
│  The baked spine: &[NodeRow], 512-byte rows, read-only for everyone
│  (reference-frame-vs-motion.md: calibrated, immutable per version, never an overlay target)
│
├── ONE ALLOCATION over the spine (alpha.rs AlphaAllocation)
│     allocate = every address the spine has, ZERO rows
│     ordinal  = base position, never claim order
│
├── 64 THOUGHT LAYERS = the epistemic frontier   (ce64-hypothesis-layers-v1.md, O5 decided 2026-10-14)
│   │  each layer = one parallel reality, one writer, sparse write, merge on read
│   │
│   ├── layer 0  GIVENS — written only by hydration (pearl::hydrate), never overwritten
│   │            a hypothesis offering another value is REFUSED and RECORDED as a contradiction
│   │            one LazyLock table per (classid, sealed version), never persisted   (givens_layer_probe)
│   │
│   ├── layers 1..63  HYPOTHESES — any strategy, any NAL level, any front end
│   │     ├── saccade: AlphaOverlay claim order (AlphaStamp: cycle, seq, rung, visits)
│   │     ├── conclusions: sparse slot deltas keyed by NodeGuid (delta_granularity_probe)
│   │     └── rung = implied level of a claim, can change within one thought (AlphaStamp.rung);
│   │           the 10 static rung lanes (alpha_tunnel.rs, LEVELS = 10) are deprecated (2026-10-20)
│   │
│   ├── PRESENCE BITS instead of references   (coverage_orientation_probe)
│   │     per row: 64-bit written-by word  → "which realities stand on this identity"
│   │     per thought: 64k-bit mask        → "where did this reality go"
│   │     every count = popcount(selection AND aperture); no key read, no foreign table
│   │
│   └── READ = merge on read (identity row + overlays, higher thought wins a shared lane;
│              the givens are never overridden)              (identity_session_probe)
│
├── G (the graph of SPOG) = classid >> 16, part of the ADDRESS   (spog_tenants.rs graph_of)
│     two graphs crossed = mask AND over one allocation (alpha_focus.rs); different
│     allocations are refused (FocusError::DifferentAllocations)
│
└── CE64, the reasoning word ON an edge (causal_edge::layout, one layout, D3)
      0..23  S/P/O   palette256 codes only
      24..31 F       frequency (evidence)
      32..39 C       confidence (evidence)
      40..42 Pearl   the causal question asked
      43..45 orientation (D-CGW-1)
      46..49 activation i4: +7 excitation … −8 inhibition     (DECISION 2026-10-17, see §4)
      50..52 novelty → wisdom (Staunen 1 … wisdom 7), moved by 46..49    (O2 still open)
      53..58 W = the thought layer the state belongs to (0 givens, 1..63 hypotheses)
      59..63 EpistemicState5, ONE field: Topology2 × Certification3 (Open … Causes)

FRONT ENDS (label views)                      ONE BACK END
  lance-graph-nal   (Narsese, NAL ops)  ┐
  lance-graph-relalg (relational alg.)  ├──► R2IL FoldDialect ──► mask-risc ──► ndarray::simd
  lance-graph-metta (MeTTa / Hyperon)   │       one program per fold; a call does not parse,
  lance-graph-cog   (Atomese; planned)  ┘       lower or allocate

ORACLES (outside the build, results only)
  OpenNARS for Research, ONA (nal-recipe-audit/oracles.md) · Hyperon interpreter, PeTTa (D-ASM-4) · MORK (D-ASM-10)
```

---

## 1. The address space (the frame)

| piece | what it is | file | status |
|---|---|---|---|
| NodeGuid | 16 bytes: classid(4) + HHTL tiers + V3 4+12 facet; address, name and Cartesian position in one | `lance-graph-contract/src/canonical_node.rs` | CURRENT-CONTRACT |
| aperture | a selection is the key with its first n bytes cared; on the sealed order it is one contiguous range found by two `partition_point`s | `lance-graph-nal/src/lib.rs` (`Aperture`, `bound_aperture`) | VERIFIED-IN-CODE |
| baked spine | `&[NodeRow]`, read-only; the reference frame | `alpha.rs` (`AlphaAllocation::over`) | CURRENT-CONTRACT |
| frame vs motion | calibrated reference data (coverage, frequency, PoS, LUTs) is never written by an overlay; a changed reference value is a new bake | `.claude/knowledge/reference-frame-vs-motion.md` | DECISION 2026-10-03 |
| Morton dense tile | for a dense class tile the row IS the HEEL code; no trie, no rank | plan § Dense tile; `contract::morton8x8` | MEASURED (558 ns point read at T = 64) |

**Rule:** a value's position is its address. Nothing is reached by reading
an index out of another lane (no gather, no foreign key). The only
gather-like operation is rank: popcount of the presence bits before a
position in its own presence mask.

---

## 2. The alpha channel and the split tunnel

| term | meaning in code | file |
|---|---|---|
| alpha channel | a second table at the same addresses as the spine, materialised only where attention went; discardable whole ("ephemer daneben, verwerfbar") | `alpha.rs` module doc |
| allocate vs claim | allocate = the whole address space, zero rows; claim = one row at one address | `alpha.rs` (`AlphaAllocation`, `AlphaOverlay::claim`) |
| not attended | an unclaimed address reads `None`; it NEVER falls back to the base row | `alpha.rs` (`get`, test `an_unclaimed_address_reads_as_not_attended`) |
| saccade | the claim order: `AlphaStamp.seq`; the trajectory is the index | `alpha.rs` (`AlphaStamp`, `scanpath`) |
| regression | a revisit keeps the first stamp and counts `visits`; the most diagnostic event | `alpha.rs` (`regressions`) |
| blind spot | allocated but never attended; only nameable because the address space is known | `alpha.rs` (`unattended`) |
| split tunnel (memory) | read side = the baked spine, shared by all lanes, no lock; write side = each lane's own overlay; parallel == sequential, byte for byte | `alpha_tunnel.rs` |
| split tunnel (persistence) | write = sparse append, one Lance version per batch, never updated in place; read = fold over resident records; Lance read once, at replay | `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md` (MEASURED in HubSPO-rs) |
| merge | lanes walked in index order; the first claim keeps its stamp, later claims add `visits`; `seq` reassigned globally | `alpha_tunnel.rs` (`merge`) |
| cross of two axes | rung × graph as a mask AND, nothing stored; "unlooked" = a graph no rung reached | `alpha_focus.rs` |
| mask algebra | `AlphaMask`: population over base ordinals; and / or / xor / and_not / not, in-place forms, one named materializer | `alpha.rs` |

**What the alpha overlay is not:** it carries no payload, so it is not the
persistence half (the split-tunnel doc says so). And it is not a reference
store: a changed calibrated value is a new bake, never an overlay write.

---

## 3. The 64 thought layers (the epistemic frontier)

All of this is in `.claude/plans/ce64-hypothesis-layers-v1.md`. Every number
below was printed by its probe on a v4 (avx512f) build.

| piece | decision / measurement | plan section |
|---|---|---|
| budget | a ceiling of 64 thoughts, not a requirement (like a cap on git branches); a lane's identity is its own type, not a rung and not a recipe id. ⊘ "BESIDE the 10 rung lanes" corrected 2026-10-20: rung is an implied level that can change within a thought; the 10 static rung lanes are deprecated | Pre-registration (2026-10-14, O5; corrected 2026-10-20) |
| lane = treatment | each lane runs one thought on the same fixtures against the same read-only base; external engines fill their lanes from golden outputs, their derivation order is their saccade | Pre-registration |
| placement | a thought's delta goes into a table it owns, or into one 16-byte slot of a shared table; the 65th table is refused | Placement and the first probe |
| delta granularity | slot deltas (32-bit mask + changed 16-byte slots) are the default; crossover to row images unstable between k = 8 and 31 | Delta granularity |
| addressing | identity trie per class with overlays per row (M); dense tiles address the row by the Morton code (D) | Identity table; Dense tile |
| presence bits | the coverage is a 64k × 64 bit matrix held two ways: per-row written-by word and per-thought mask; both kept, each count query routed to its faster form | Coverage orientation |
| givens | layer 0 = hydration; never overwritten; a different offered value is refused and recorded as a contradiction; bits 59..63 written as one `EpistemicState5`, certification capped by the proofs that ran | Givens layer |
| sessions | a session materializes a prefix range temporarily (105 ns/row one thought, 138 ns/row all 64); all 64 over a class at once would be 2 GiB, so sessions stay temporary | Identity table |
| hypotheses H1..H6 | base never changes; reserve cost flat in lanes; parallel == sequential; hydration marks; NAL evidential base vs Gadamer gate; exact codes vs OpenNARS | Pre-registration |

Measured nanosecond figures (v4): perspective count on one identity 1.1 ns;
aperture counts ≤ 0.40 ns per row; hydration 34 ns per row; hypothesis write
with the given check 73 ns per offered cell; point read 776 ns (identity
trie, T = 64) and 16–60 ns with few thoughts on a dense tile.

### Why presence bits replace references

A reference (`mailbox_ref`, `spo_fact_ref`, a foreign key, a pointer into a
fact table) has to be followed to learn anything, and following it is a key
read. A presence bit is already the answer: "this reality stands on this
identity". Provenance, G and evidence therefore all enter as layers:

| what a reference would carry | what carries it instead |
|---|---|
| which thought holds a state | the lane itself: a thought runs in one of 64 isolated epistemic boundaries, and its materialization is that lane's sparse storage layer (alpha channel, split tunnel, sparse write, merge on read). The 64 lanes in storage are the thoughts that inhabit them; W and the written-by word name the lane, they point nowhere |
| which sources a belief rests on | a source bitset per delta, inside the lane (D-HL-3): presence bits over sources, not a reference. Independence for revision is decided by these bits, never by the lane number: one observation copied into two lanes counts once, two independent observations in one lane count twice |
| which graph G a quad belongs to | `classid >> 16` in the address; a tenant mask |
| the evidence behind a value | the givens layer and the hypothesis layers, compared by mask; independence decided by the source bitsets, not by lane membership |
| which facts two ontologies share | the AND of their masks over one allocation |
| who disagrees with the givens | the contradiction records (row, thought, lanes, offered value) |

---

## 4. CausalEdge64: what each field means and who owns it

The layout is fixed by `crates/causal-edge/src/layout.rs` (`_LAYOUT_COVERAGE`
covers all 64 bits once). One layout, every reader migrated to it (D3,
`entries/2026-10-17-ce64-one-layout-moore-synapse-membrane.md`).

| bits | field | meaning | written by | status |
|---|---|---|---|---|
| 0..23 | S, P, O | palette256 codes only. Never a block number, a row index or any other non-palette value | the domain's palette | CURRENT-CONTRACT (ndarray #366 violated it and was closed) |
| 24..31 | F | frequency | evidence (revision folds) | CURRENT-CONTRACT |
| 32..39 | C | confidence | evidence | CURRENT-CONTRACT |
| 40..42 | Pearl | the causal question asked (`CausalMask`: O = 0b001, P = 0b010, S = 0b100) | the asker | CURRENT-CONTRACT; Pearl asks, it does not certify |
| 43..45 | orientation | `Orientation`: `Core` + the seven dative-map functions | the reading | CURRENT-CONTRACT (D-CGW-1) |
| 46..49 | activation i4 | +7 excitation … 0 neutral … −8 inhibition: the signed reaction, never the operation | the running thought | DECISION 2026-10-17; code still reads it as the inference mantissa (`isa.rs`), see §8 |
| 50..52 | novelty → wisdom | Staunen 1 … wisdom 7, climbed one step at a time, moved by 46..49 | the maturation | OPEN (O2): `learn()` also reads these bits as plasticity |
| 53..58 | W | the thought layer this state belongs to: 0 givens, 1..63 hypotheses; the breadcrumb to what changed the state. Hot path only: after a Lance seal the value is arbitrary (DECISION 2026-10-20, §8 row 4) | the merge / materialization | DECISION 2026-10-17; three competing readings exist, see §8 |
| 59..63 | EpistemicState5 | ONE field, never split: Topology2 × Certification3 (Open, Associated, Related, Supports, CausalCandidate, Causes) | hydration; then only measured results | CURRENT-CONTRACT (`epistemic_state5.rs`) |

**The operation lives in the running thought, the result lives in 59..63,
and the breadcrumb to what changed it lives in W.** Writing the operation
back into the edge (as the opcode reading of 46..49 does) gives the same four
bits two meanings and tells a reader nothing the thought did not already
know.

---

## 5. Four ladders that must not be merged

| ladder | what it measures | where | size | per |
|---|---|---|---|---|
| rung 0..9 | depth of processing / attention inside a thought (Maslow-shaped `RungLevel` Surface..Transcendent, `RungElevator` never below base) | `rung_schedule.rs` (`LEVELS = 10`, `MAX_RUNG = 9`), `recipe_loci.rs`, `alpha_tunnel.rs` | 10 | recipe dispatch, attention |
| novelty → wisdom | how far one discovery has matured | CE64 50..52 (`edge.rs` `novelty_wisdom`), `maturation_ladder.rs` (test-local, 8 stages) | 8 | edge |
| NAL-1..9 | which grammar and inference rules a reasoner can execute; cumulative capability, not nine passes | `.arxiv/2026-08_nal1_to_nal9_is_a_capability_ladder_not_nine_cycles.md`, `nal-recipe-audit/nal_inventory.tsv` | 9 | thought |
| Pearl rung | association / intervention / counterfactual question | CE64 40..42 (`CausalMask`) | 3 bits | edge |

Forbidden identifications (from the Wang distillation and the code docs):
a rung is not a NAL level; NAL-9 is not rung 9; the 10 tunnel rungs are not
widened to match NAL; maturation is not NAL; a Pearl rung is not NAL.

What DOES connect them: maturity (50..52, pushed by 46..49) is the gear that
may open a higher-level thought on that discovery. A thought is not capped
at any NAL level; a higher thought reads lower thoughts' layers as its
objects (NAL-5 makes statements into terms, NAL-9 makes processing into
events). 64 lanes hold all nine levels with room to spare.

---

## 6. NAL-1..9 and the strategies that fill the lanes

`nal_inventory.tsv` lists 115 capabilities with source, truth function,
oracle fixture and falsifier:

| level | rows | strategies (examples) | executes here today |
|---|---|---|---|
| NAL-1 | 15 | deduction, abduction, induction, exemplification, conversion, revision, choice, backward inference | revision (matches OpenNARS), choice, question, threshold (`lance-graph-nal`) |
| NAL-2 | 10 | similarity, comparison, analogy, resemblance, instance/property | copula `<->` stored; rules not yet |
| NAL-3 | 10 | intersection, union, difference, composition, decomposition | blocked: compound terms (ledger R10) |
| NAL-4 | 7 | product, image | blocked: R10 |
| NAL-5 | 17 | statement as term, implication, equivalence, detachment, contraposition | copulas `==>`, `<=>` stored; rules not yet |
| NAL-6 | 9 | variable kinds, substitution, introduction, elimination | blocked: R10 / variables |
| NAL-7 | 13 | events, temporal relations, temporal induction, update vs revision, anticipation | `temporal.rs` version-range reads exist; not tested as NAL-7 |
| NAL-8 | 12 | operation, goal, desire, goal derivation, decision, planning | none |
| NAL-9 | 22 | self concept, `^want`, `^wonder`, `^believe`, `^doubt`, `^consider`, feelings, internal experience trace | none; the recorded saccade is steps 1–2 of the six-step test only |

**A thought = (strategy, front end) compiled once into one program, running
in one lane.** Several strategies on one question become parallel realities;
where they agree and where they diverge is read from the written-by words and
the contradiction records. Agreement across engines (ours, OpenNARS, ONA) is
itself a signal (H6).

**NAL-9 needs, per Wang:** (1) record that a step occurred, (2) retrieve its
trace, (3) a self-model of the step, (4) compare predicted with actual
reasoning, (5) an authorized change to later processing, (6) a measurement
that the change helped. The substrate provides 1–4: the saccade, the
hypothesis layer (prediction), the givens and later hydration (actual), the
contradiction records and written-by words (comparison). Steps 5–6 are what
a NAL-9 thought must show, against a control run without it.

The 34 recipes (`recipe_kernels.rs`, `recipe_dispatch.rs`) already stack:
a deeper recipe reads the loci a shallower one placed (`recipe_loci.rs`
`carried_awareness`). Per the NAL audit none of them implements a NAL
operation yet; their closest resemblances are recorded in
`nal-recipe-audit/README.md`.

---

## 7. Front ends and the one back end

| crate | front end for | state | plan |
|---|---|---|---|
| `lance-graph-nal` | NAL / Narsese | Question, TruthAtLeast, Choice, Revision over folds; global core methods, no classid check; W4 truth laws, W5 syllogisms, W6 Narsese text open | `nal-polyfill-v1.md` (D-NALP) |
| `lance-graph-relalg` | relational algebra | prototype: proven rewrites to a fixpoint, lowering via `FoldDialect`; no μ, no hop yet | `relalg-v1.md` (D-RA) |
| `lance-graph-metta` | MeTTa / Hyperon atoms | probe only (D-MTP-1 `join_probe`); `Space` adapter is D-MTP-2 | `metta-polyfill-v1.md` (D-MTP) |
| `lance-graph-cog` | Atomese / OpenCog bridge | not created; planned as our Apache-2.0 API with a separate AGPL `atomspace-lance` adapter; mining seam D-ASM-7 | `metta-datalog-assimilation-v1.md` (D-ASM) |

Target topology: every front end lowers to R2IL `FoldDialect` → mask-risc →
`ndarray::simd`. Current paths differ per crate: `lance-graph-nal` and
`lance-graph-relalg` depend on `lance-graph-r2il` and lower through it;
`lance-graph-metta` is a probe that lowers by hand through Quack into
mask-risc, with no R2IL dependency yet; `lance-graph-cog` does not exist.
Build against the current path of a crate, not the target. Atoms extract
into classes and SoA (head + arity + positions; one plane per position, row =
instance); the IR describes, the masking algebra executes and never sees
atoms or symbols (entry 2026-10-17 D13). Text is a label resolved once
through the codebook; a rule or query holds coordinates, never strings.

---

## 8. Open, with the conflict each one carries

| # | open point | the conflict | where it is tracked |
|---|---|---|---|
| 1 | 46..49 | DECISION 2026-10-17 says activation; `isa.rs` executes opcodes from these bits (+1, +2, −1, +4, +5) and `ce64-semantic-upper-half.md` § Open said "stay the inference mantissa; activation moves to a row field (D-CGW-9)". Resolution proposed in the plateau (U1): the opcode reading becomes the declared "isa-v2" reading behind a version gate | plateau §4 O1; D-HL-1 |
| 2 | W (53..58) | three readings: corpus root handle (`witness_table.rs`, unwired), the Moore membrane locality answers (`MooreNars16` omits W, entry D5/D6), and the thought layer (DECISION 2026-10-17). `forward` / `syllogize` return W = 0 (PROBE-W-PRESERVE) | plateau §4 O3 (G6); D-HL-1 |
| 3 | 50..52 | novelty → wisdom (`wire`) and per-plane plasticity (`learn`) on the same bits | plateau §4 O2 |
| 4 | ~~"CE64 is not storage" vs "the result lives in 59..63"~~ **DECIDED 2026-10-20** | Both hold, in different places. CE64 is an ISA contract, not a storage contract: in the hot path, inside a live thought, the edge is thought-aware (W names the lane, 59..63 carries the result). Lowered into a sealed Lance version, the bits are kept bit-exact and keep their ISA meaning (like BF16 on disk); only the in-RAM reference becomes arbitrary: slot numbers recycle, so a sealed W means nothing. The thinking styles used and the reasoning replay live in the thinking, or are linked to the kanban lane's versions; never in sealed CE64 bits. (Before: "needs one sentence from the user".) | entry 2026-10-20-ce64-isa-not-storage; entry 2026-10-17-ce64-one-layout D1 |
| 5 | promotion to `EPIPHANIES.md` | the file is generated from `epiphanies-canon.json`, whose rows must resolve to archive lines; a post-baseline Eureka has no route in | entry 2026-10-17-ce64-one-layout § OPEN |
| 6 | NAL-9 steps 5–6 | not built; needs a pre-registered run with a control | D-HL-8 / D-NALP |
| 7 | compound terms | NAL-3..6 blocked | ledger R10 |

---

## 9. Forbidden shapes (each one has cost a session)

| shape | why | rule source |
|---|---|---|
| a foreign key, `x[y[i]]`, an index read from another lane | a key read instead of a fold; two orders of magnitude slower | `.claude/session` rules; entry D6 |
| a reference table behind W (`mailbox_ref`, `spo_fact_ref`) | same as above | this capstone §3 |
| splitting 59..63 or naming 61..63 alone | it is one field | `epistemic_state5.rs` |
| non-palette values in S/P/O | breaks every palette reader | ndarray #366 |
| writing the operation into the result edge | two meanings for one field; no information gained | §4 |
| an overlay falling back to the base row | a plausible value in place of an absent one | `alpha.rs` |
| an overlay writing calibrated reference data | frame vs motion | `reference-frame-vs-motion.md` |
| a rung = NAL level mapping; widening the tunnel to 9 or 64 rungs to match NAL | NAL is a capability ladder | Wang distillation |
| a second algebra or interpreter beside mask-risc / Quack | the fold path is the only executor | lane-fold capstone |
| strings as addresses or inside rules | labels resolve once through the codebook | session rules |
| G wired as a foreign key between ontologies | G is an address prefix; crossing is a mask AND | `spog_tenants.rs`, `alpha_focus.rs` |

---

## 10. Reading order for a new session

1. This file.
2. `.claude/plans/ce64-hypothesis-layers-v1.md` (whole) and
   `.claude/knowledge/ce64-reasoning-substrate-plateau.md` §4–§5.
3. `crates/lance-graph-contract/src/alpha.rs`, `alpha_tunnel.rs`, `alpha_focus.rs`.
4. `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md`,
   `.claude/knowledge/reference-frame-vs-motion.md`.
5. `.claude/board/entries/2026-10-17-ce64-one-layout-moore-synapse-membrane.md`
   and `2026-10-17-epistemic-frontier-clicks.md`.
6. `.arxiv/2026-08_nal1_to_nal9_is_a_capability_ladder_not_nine_cycles.md`,
   `.arxiv/undated_nal9_mental_operations_are_executable_events_not_a_ninth_op.md`,
   `.claude/knowledge/nal-recipe-audit/README.md`.
7. `crates/lance-graph-nal/src/{lib,nal}.rs`; `.claude/plans/metta-datalog-assimilation-v1.md`.
8. `docs/architecture/ce64-semantic-upper-half.md`; `crates/causal-edge/src/layout.rs`.
