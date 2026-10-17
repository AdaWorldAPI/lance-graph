# Cartesian Epistemic Self-Model: Presence, CE64, Datalog and Saccadic Understanding

**Status:** ARCHITECTURE / RESEARCH THESIS (2026-10-17). This document separates `CURRENT-CONTRACT`, `MEASURED`, `DECISION`, `PROPOSED` and `OPEN`. It is documentation only: no new runtime, CE64 layout, opcode, authority, ontology, store, or engine is approved here.

**Thesis:** `lance-graph` can use its own Cartesian-addressed substrate to represent both external concepts and its own executable methods. A thought-first selection reads existing values by address, a presence plane describes where each thought has written, CE64 supplies a declared epistemic reading, and provenance plus independent execution decide what has actually been earned. A demand-scoped relational closure can expose the spaghetti of dependencies. Attention, represented by an ephemeral alpha trace, makes this structure observable without equating activity with evidence.

**Do not read "holographic" as a visual halo.** In this design it names the *Cartesian-addressing inversion*: the program names its operand tables, the coordinate names their row, and a read does not traverse a parent tree or gather an address from another lane. A Morton/HHTL order may still provide locality. It is not a claim about optical holography or information-theoretic interference.

## 0. The architectural picture

```text
                    SEALED CARTESIAN REFERENCE FRAME
  NodeGuid / HHTL / classid / Morton tile / declared reading / version
             base is immutable for a sealed reference version
                                 |
                 +---------------+-----------------+
                 |                                 |
          LEXICAL + WORLD                    SOFTWARE SELF-MODEL
       COCA / WordNet / Bible             symbols / methods / compiled IR
       DeepNSM-v2 / SPO events            dependency / test / receipts
                 |                                 |
                 +---------------+-----------------+
                                 |
                    ONE ADDRESSABLE POPULATION
                  named tables, same row coordinate
                                 |
                  64 THOUGHT REALITIES (0..63)
             0: hydrated givens; 1..63: hypotheses
            per-thought presence P[t][a] / per-row W[a]
                                 |
         +-----------------------+------------------------+
         |                       |                        |
  ALPHA: ATTENTION        RELATIONAL ALGEBRA       EPISTEMIC READING
  transient saccades      masks / Quack / R2IL    CE64 59..63 + witness
  visits / blind spots    proposed Datalog μ      not a proof by itself
         |                       |                        |
         +-----------------------+------------------------+
                                 |
                REPLAY / ABLATION / INDEPENDENT ORACLE
             predict -> inspect -> compare -> revise claims
                                 |
               SELF-MODEL UPDATED ONLY IF EARNED BY EVIDENCE
```

This is a *composition hypothesis*. The pieces below have unequal implementation status. In particular a general recursive Datalog engine, autonomous verified self-improvement, all front-end adapters, and the complete CE64 upper-bit migration are **not** shipped by this diagram.

## 1. Five distinct things, no lossy collapse

| Question | Surface | What it can answer | What it cannot assert |
|---|---|---|---|
| **Where?** | `NodeGuid`, class, Cartesian row coordinate, sealed codebook | Deterministic identity/address | Semantic truth or authorization |
| **Was there?** | presence bits, per-row written-by word | Which thought occupied which location | Truth, confidence, independence |
| **Why looked?** | `AlphaStamp`, `AlphaOverlay`, `AlphaFocus`, scheduler receipt | Attention order, visits, missed aperture | Proof of existence or belief strength |
| **What is claimed?** | declared CE64 reading, NAL truth, typed propositions | Typed state under a versioned codebook | That a claim was validated |
| **Why believe it?** | source-root ledger, derivation support, temporal and independent oracle receipts | Grounds for evidence revision and certification | Automatic intervention permission |

These distinctions are invariants, not decorative layers. Presence agreement, equal values, semantic agreement, shared evidential roots, and certified causation are **five different relations**.

### 1.1 Address is not a reference chase

Canonical thought-first expression (see D-HOLO-H14c):

```text
Thought(t)[Table(a)[i] : Table(b)[i]]
```

- `t` is the thought/execution perspective, whose program resolves `a` and `b` once.
- `i` is the same Cartesian row coordinate for both operands.
- Each table's own presence bit says whether the operand exists at `i`.
- No `table[a][table[b][i]]`, foreign-lane gather, string lookup in the fold, trie walk to find an operand, or row-first 512-byte reassembly.
- A logical address is **not** a Lance `_rowid`, physical fragment offset, Arrow batch position, or permission grant. Coordinate-to-storage mapping has a versioned safety boundary.
- Different class allocations cannot be AND-ed merely because they use equal ordinals. `AlphaFocus` refuses different allocations.

### 1.2 Address ≠ semantic distance

`deepnsm-v2::vocab::PaletteVocab` gives a frequency-ranked 256 × 256 `(basin, identity)` word tile (`WordId` is `u16`). Consecutive addresses proxy *frequency rank*, not semantic closeness. The trained Cam96 codebook provides a separate semantic-similarity reading. COCA counts and PoS readings live beside the word identity; WordNet synsets and language-use functions require explicit mappings and provenance.

A lexical form may support multiple senses, grammatical readings, or interpretations. The same address can flare for multiple reasons without silently conflating those reasons. A Haspelmath semantic map is a graph of grammatical-function adjacencies, not a pre-existing Euclidean 12 × 12 grammar ISA; any quantized projection must be evaluated against held-out mappings.

## 2. The epistemic occupancy duality (CORE)

For one dense 256 × 256 class tile, define:

```text
M[a,t] ∈ {0,1}     a ∈ [0,65,536), t ∈ [0,64)
R[a] = Σ_t M[a,t]·2^t
P[t][a] = M[a,t]
```

`R[a]` is one **64-bit written-by word per identity**: who occupies address `a`.
`P[t]` is one **65,536-bit presence mask per thought**: where thought `t` has written.

They are transposes of the *same logical occupancy matrix*, not two unrelated ontologies.

- Per orientation per class: `65,536 × 64 bits = 512 KiB`; both orientations occupy `1 MiB`, excluding metadata and values.
- Per-thought mask: `65,536 bits = 8 KiB`.
- At an address: `popcount(R[a] & selected_thoughts)` counts selected present perspectives.
- Across addresses: `popcount(P[t1] & P[t2] & aperture)` counts shared occupied coordinates.
- `popcount(P[t1] ^ P[t2])` is **Hamming distance of coverage**, not semantic or epistemic disagreement.
- `OR` over thought masks yields the union of their **occupancy**.
- An empty bit indicates absence *in that selected surface*, never empirical falsity or a globally unoccupied ontology.

For multiple value lanes, track **per-(thought,lane)** presence where required; the aggregate written-by word alone cannot answer whether a particular value lane is populated. Do not infer value-level identity or equality from thought-level occupancy.

**MEASURED:** `coverage_orientation_probe` tested both representations on AVX-512 with a brute-force oracle and disable runs. Perspective count on one address was ~1.1 ns; aperture counts reached ≤0.40 ns/address in the faster orientation. The 2026-10-15 correction to early-exit Q-TH reversed some performance winners; retain both orientations, route by workload and density. See `../plans/ce64-hypothesis-layers-v1.md` § Coverage orientation.

**MEASURED:** D-HOLO-H14c showed population `AND`/`XOR` ~0.9–1.2 ns per 64-address word on its recorded machine, no payload reads or row assembly for population operators. Point projection read the requested tables directly; terminal materialization remained explicit. These are **probe-specific**, not universal performance promises.

## 3. One fixed CE64 contract, with declarations and known conflicts

CE64 is an **ISA/ALU-style 64-bit contract**, not an extra store of self-model objects and not an alternative ontology. Its physical fields have a single canonical layout, while declared typed readings determine permissible interpretation.

| Bits | Field | Contract / status |
|---|---|---|
| 0..23 | S/P/O | Three 8-bit palette codes; not foreign row IDs |
| 24..31 | F | Frequency/evidence register |
| 32..39 | C | Confidence/evidence register |
| 40..42 | Pearl | Which causal question was posed, **not** proof of causation |
| 43..45 | Orientation | Canonical orientation reading |
| 46..49 | i4 | **DECISION** 2026-10-17: excitation/inhibition; **OPEN migration** from inference-opcode readers |
| 50..52 | maturation/plasticity | **OPEN**: novelty-to-wisdom vs `learn()` plasticity |
| 53..58 | W | **DECISION**: thought-layer breadcrumb 0..63; **OPEN migration** from old corpus-root reading and writer paths |
| 59..63 | `EpistemicState5` | One declared field, `Topology2 × Certification3`; implemented reader contract |

`EpistemicState5` is a product with `Topology2` = Direct / IndirectKnown / IndirectUnknown / Unknown and `Certification3` = Open / Associated / Related / Supports / CausalCandidate / Causes. **24 meaningful, eight reserved raw codes.** The five-bit word classifies the claim; observation provenance is **not encoded in it**. `raw5 == 0` is *Direct × Open*, **not** a generic unknown. `Epi5Declarations::project_state5` enforces declared `(classid, rail, generation)` and rejects untrusted legacy readings or reserved values.

The **operation** is in the compiled running thought; the resulting relation has an epistemic reading and a thought breadcrumb. Merely writing a high certification code does not satisfy the external proof obligation. A lane cannot certify itself by emitting its own `Causes` bits.

**OPEN, do not paper over:** `causal-edge::layout/isa` still documents opcode semantics at 46..49; W has legacy producers and a known `forward`/`syllogize` preservation issue; bits 50..52 are contested; CE64 as a control word vs persisted edge-state needs an explicit transition-ownership law; ndarray has legacy incompatible CE64 readers. These are migration gates, not permission to invent CE128, new opcodes, or a second layout.

## 4. Attention is a projection, not knowledge

The alpha channel is an ephemeral, discardable surface over the immutable reference spine:

- Allocate the whole address domain logically; initially materialize **zero** attention rows.
- `claim(address)` records that attention landed, `AlphaStamp.seq` orders saccades, `visits` counts regressions.
- An unclaimed address returns *not attended*; **never** fall back to a base value to fabricate a visit.
- `AlphaFocus` intersects rung and graph/tenant masks over the **same allocation**. Rung (10 processing levels), NAL capability, CE64 maturation, and Pearl questions are different axes.
- The attention trace can be discarded without invalidating stored evidence or certified results.

**The visualizer starts blank, but the sealed address space need not be empty.** A flare is a view of activity at an address, not an assertion that a new fact was created.

Proposed observability projections (no new persistent schema required as a starting assumption):

1. **Lexical flare:** COCA/WordNet/DeepNSM-v2 resolves a token or sense to a declared address.
2. **Interpretive flare:** another method considers an alternative meaning or grammatical function.
3. **Deductive flare:** Quack/R2IL or a future Datalog fold derives an eligible consequence.
4. **Epistemic flare:** declared CE64 certification or NAL evidence changed *with an eligible witness*.
5. **Self-model flare:** a thought inspects a program, method, compiler refusal, or previous execution receipt.
6. **Counterfactual flare:** a detached alternative world changes a support set, not the immutable givens.

Do not visually portray attention lines as causal edges. Show separate layers for address, occupancy, declared relation, proof provenance and saccade trajectory. Inactive is not unknown; unknown is not false; false is not an unseen row.

## 5. Relational closure as demand-scoped explanation, not universal truth

**PROPOSED:** A Datalog-style rule/value front end names reusable recursive relations over the existing typed class/SoA planes, lowers lawful fragments through `lance-graph-relalg`/Quack/R2IL into mask-RISC, and reuses the canonical execution path. This is *not* a claim that recursive Datalog with negation, joins, mutable provenance and fixed-point execution is currently shipped as one engine. Current `relalg` remains limited (no general μ / hop).

Example **illustrative syntax**:

```prolog
depends(A,B) :- calls(A,B).
depends(A,C) :- calls(A,B), depends(B,C).
```

For a named finite extensional fact set, positive Datalog provides a least-fixed-point closure. Its results are valid **relative to those facts and rules**, not a guarantee of worldly truth. A source-derived `calls` edge is not sufficient to prove behavior or a causal influence.

Candidate demand flow:

```text
Attention(A, horizon, aperture)
  -> select declared class / relation / rule handles
  -> compose coverage and permission masks BEFORE payload reads
  -> demand only the reachable relation subproblem
  -> fold matching planes at their common coordinate
  -> preserve proof witnesses / unresolved insufficiency
  -> return an interpretation or typed refusal
```

**Strict boundaries:**
- No gather (`x[y[i]]`) or foreign lane read disguised as a join. A proven preimage mapping is a separately accounted operation, not a free same-coordinate read.
- No strings/atoms in the hot fold. Resolve labels at the typed front end; Atomese/MeTTa head+arity+positions lower to classes/SoA first.
- Negation of a missing fact is not licensed without a **coverage receipt** proving an adequate, current and complete observation window and supported predicate set.
- Hyperon nondeterministic branch multiplicity must be preserved by a parity oracle; a set mask is not automatically a faithful representation of a bag of bindings.
- No second graph evaluator or hidden fallback. Unsupported shapes get a typed refusal.

## 6. Recursion: the graph can model its own implementation

Treat software itself as a source corpus, with typed and versioned distinctions between:

```text
Declared source -> Extracted AST/SPO -> Compiled method / refusal
  -> actual execution receipt -> independently verified effect
  -> hypothesis about implementation -> authorized proposal (optional)
```

A candidate self-model is `H_v = Project(G_v)` for executable program version `v`.
A hypothesis is useful if it makes a **precommitted, falsifiable prediction** about `G_v` or a controlled version `G_{v+1}` that survives independent execution.

**SAG lineage and caution:** PR #1509 demonstrated a one-concept, one-hop SoftwareGraph based on Quack, but those `awareness.rs`/`method.rs` surfaces were subsequently **removed from main by #1524**. The capstone in `../plans/sag-capstone-alignment-v1.md` is a research alignment, not evidence that the earlier runtime remains shipped. Build on current D-CMR/actual code only.

Reflection can recurse over **addresses of methods and receipts**, not clone the entire world into a fresh ontology at each depth. Proposed ladder of discriminating abilities:

1. Find own source/method identity and version.
2. Associate compiled IR with its declared method and measured execution.
3. Predict an observable behavior before running it.
4. Compare the observation with the prediction, including silence only when observable.
5. Ablate one method/assumption with independent source-root accounting.
6. Propose an alternative implementation or attention policy.
7. Repeat the prediction under a holdout test and show improved results at equal budget.

A recorded reflection chain is not consciousness, independent evidence, or automatic permission to modify code. No recursive self-certification: an independent oracle decides whether a behavioral claim was verified.

## 7. Counterfactuals: what an empty flare cannot tell us

For a result `Q` with derivation provenance `Q = (A∧B)∨C`, occupancy alone cannot tell the difference between:

- all three sources merely present;
- A and B jointly necessary;
- C independently sufficient;
- a duplicated C copied into multiple thought layers.

`P[t1] & P[t2]` proves only *shared coverage*. The causal/derivational question needs an explicit provenance expression and a detached replay or independent oracle. Removing A alone may leave Q; removing A and C may not.

**Falsifier:** use one shared physical root copied into multiple thought worlds and two genuinely independent roots in one world. Any procedure using only thought IDs to infer evidential independence fails.

**Temporal boundary:** pin source revision, class/reading generation, dataset version, decision horizon and relevant codebook. Compare `W0` (factual) with `WX` (ablation) without leaking future evidence. A stale Lance coordinate map or an unauthorized read must refuse rather than silently use a plausible value.

**No causal inflation:** a Datalog dependency edge is not necessarily a Pearl causal edge; strong co-occurrence is not intervention certification; self-model predictions are not empirical truth until checked.

## 8. Measurements, scaling and the inconvenient result

| Surface | Existing probe result | Limitation |
|---|---|---|
| Presence operations | ~0.9–1.2 ns per 64-address word on the tested AVX-512 machine | Counts occupancy, not value equality, meaning or proofs |
| Per-row perspective count | ~1.1 ns on its measured reference grid | 64-lane budget, specific hardware |
| Dual occupancy orientation | 512 KiB per orientation for 64 × 65,536 | Metadata, roots, values and additional value-lane masks excluded |
| Point read | Thought-first tests passed equality + no-row-materialization gates | Work linear in selected lanes; hot/cold locality matters |
| C-lazy dense positional values | At 64 thoughts / 31 lanes, 1% uniform writes: ~1,850 MiB RSS | Page-dense waste; sparse comparator ~25 MiB with slower lookups |
| Lance 13 masked I/O | Clustered sparse reads can cut bytes; scattered sparse reads sometimes do not | Projection/row filtering does not guarantee page read savings |

These numbers are **documented measurements from October 14–15**, not rerun by this doc. See D-HOLO-H14c for setup, full grids, corrected timer bias, and negative results.

**Scaling hypothesis:** Keep presence projections cheap and composable, but investigate a residency-aware value layout so 64 hypotheses do not materialize a dense 2 GiB table merely to inspect a sparse attentional neighborhood. Memory, versioned coordinate mapping, data locality, and proof-maintenance cost may dominate bitmap ALU throughput.

## 9. Experiments that could justify the large claim

Status of all tests below: **PROPOSED**, not passed. Build no new executor or register merely to make them pass.

| Gate | Minimal fixture / oracle | Kill condition |
|---|---|---|
| E0 Address | Same logical coordinate across compaction/rewrite; pinned version and oracle | Silent value misattribution or stale map accepted |
| E1 Occupancy | 64×65,536 seeded matrix, per-row/per-thought transpose and scalar oracle | Any popcount or aperture disagreement |
| E2 Absence | Zero value, missing overlay, unvisited address, unseen-but-existing fact, observed-false fact | Any state conflation |
| E3 Reading | All 32 CE64 upper codes under declared/legacy/unknown generations | Reserved or untrusted state accepted; code zero treated as generic unknown |
| E4 Evidence | Duplicated source copied into worlds vs independent sources | Duplicate support counted twice |
| E5 Rule parity | Existing Quack/R2IL masks vs independently executed positive Datalog over finite fixture | Missing/excess answer; unseen negative silently certified |
| E6 Bags | MeTTa nondeterministic 2×2 branch against Hyperon/PeTTa | Four answers collapsed to two by a set mask |
| E7 Nonlocal | Same-address read vs unsupported foreign-lane preimage | Foreign gather presented as zero-copy |
| E8 Ablation | `Q=(A&B)|C`, single and joint cuts, fixed horizon and independent provenance oracle | Wrong survival / history leakage |
| E9 Self-model | Predict effect of a withheld source-code edit before independently executing it | Retrospective prediction or source AST treated as behavioral proof |
| E10 Attention | FIFO, confidence priority, entropy, calibrated EIG and Moore/Staunen under same compute budget | Adaptive policy fails held-out baseline or uses feedback leakage |
| E11 Cost | Sparsity × clustering × k × T, cold/hot, both occupancy orientations and storage arms | No benefit under realistic workloads; unjustified performance claim |
| E12 Authorization | Propose change, then revoke relevant role/policy at commit | A proposed method mutates state without a fresh authorization decision |

Pre-register baseline, random seeds, source revision, fixtures, oracles, workload, counters, and acceptance thresholds before a run. Include disable/falsifier variants that must fail. Preserve negative results.

## 10. What success would actually mean

The strongest **falsifiable** claim is not "we built a mind" or "64 lanes prove AGI".

It is:

> With a fixed evidence corpus and fixed execution budget, a graph-backed reasoner can use its own inspectable execution/provenance traces to identify a mistaken interpretation, choose a better method, and achieve a **measurable held-out gain in correct, independently witnessed predictions** without fabricating evidence or changing the canonical ISA.

The growth path is:
1. **Epistemic microscope:** vocabulary and conceptual loci flare with separate attention/evidence/sense layers.
2. **Counterfactual laboratory:** 64 isolated treatments compare method choice, provenance and prediction.
3. **Falsifiable software self-model:** the graph predicts, observes and revises claims about its implementation.
4. **Cumulative method discovery:** candidate algebraic equivalences become reusable only after independent semantic and cost validation.

This could transform a knowledge graph from a static catalog of relationships into an experimental instrument for examining *how relationships are inferred*. The major novelty would be **cheap, address-aligned observability composed with rigorous provenance and independent error correction**, not a purportedly magical interpretation of one bitfield.

## 11. Source-of-truth reading map

All paths are repository-relative. The notes are intentionally not copied into this architecture; they own their current contracts, dates, and open issues.

- `../knowledge/epistemic-frontier-capstone.md`: authoritative architectural inventory, statuses, conflicts and source locations.
- `../plans/ce64-hypothesis-layers-v1.md`: 64 thought sessions, dual presence orientations, givens, measured probes.
- `../plans/d-holo-h14c-thought-first-v1.md`: thought-first reads, Hamming population, memory negative result, Lance masked I/O.
- `../../crates/lance-graph-contract/src/alpha.rs`, `alpha_tunnel.rs`, `alpha_focus.rs`: attention, saccades and tenant/rung projection.
- `../../crates/lance-graph-contract/src/epistemic_state5.rs`: five-bit typed declarations, certification and topology product.
- `../../crates/causal-edge/src/layout.rs`: physical CE64 layout; **check conflicts before implementation**.
- `../board/entries/2026-10-17-epistemic-frontier-clicks.md`: operator decisions and open migrations.
- `../board/entries/2026-10-17-ce64-one-layout-moore-synapse-membrane.md`: CE64 reader catalog and Moore-local open questions.
- `../plans/metta-datalog-assimilation-v1-council-v3.md`: **DO NOT BUILD** banner for forbidden foreign-lane gather; follow H14c/H14d instead.
- `../../crates/lance-graph-quack/examples/wired_reasoning_probe.rs`: finite Boolean Datalog-shaped property checks, not full recursive Datalog.
- `../../crates/lance-graph-relalg/src/ir.rs`, `../../crates/lance-graph-nal/src/lib.rs`: current typed IR and NAL lowering entry points.
- `../knowledge/software-awareness-provenance-parity.md`: removed SAG first build, authority separation and independent verification.
- `../knowledge/sag-v2-observability-causality-and-evaluation-addendum.md`: coverage receipts, blind self-prediction, negative observations.
- `../plans/sag-capstone-alignment-v1.md`: consolidated research vocabulary, including inert `ProposedMethod`.
- `../../crates/deepnsm-v2/src/vocab.rs`, `coca.rs`, `codebook.rs`, `examples/bible_wave.rs`: vocabulary/semantic/corpus inputs; distinguish actual code from planned full-stack wiring.

**Change rule:** Update this document by distinguishing source-backed `CURRENT-CONTRACT` or `MEASURED` from new `PROPOSED` ideas. Never silently "resolve" an upper-bit interpretation conflict, mutate a user-authorized ISA, or infer that historical code remains on `main`.

