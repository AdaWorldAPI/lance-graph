# QUESTION BRIEF — coresearch: wiring thinking onto CausalEdge64 and the mask algebra (O1–O5)

## 1. The question
The operator wants all thinking wired to the `CausalEdge64` register: its
SPOFC fields (S, P, O, F, C at bits 0..39) and what the operator calls its
"ALU ISA" half (bits 40..63). Where the register is the wrong home, thinking
should migrate onto the Moore masking algebra and the EWA sandwich.

The inventory (`inventory.md`, alongside this brief) shows this is not yet
possible. Five questions are open:

- **O1. Candidate identity and ranking.** How does a population of scored
  candidates (chess: one score per legal move) keep the identity of each
  score and support ranking?
  - The mask algebra has membership only: no ORDER BY, no position terminal
    (no masked argmax), integer lanes only.
  - `ThoughtCtx.candidates: Vec<f32>` carries neither identity nor order.
  - The chess census measured that every move-narrowing recipe loses move
    identity: TCP, CUR, ETD, TCF, IDR, and CDT's convergent branch.
- **O2. A home for the thinking scalars.** free_energy, sd (dispersion),
  dissonance, temperature and rung have no CE64 bits.
  - Candidates on record: `MetaWord` (a sibling u32 column holding
    thinking 6 | awareness 4 | nars_f 8 | nars_c 8 | free_e 6), or a side
    table beside the register. D-CE64-CYCLE-0 measured that the evidence
    table and attention state are needed and live outside the register.
  - Bits are contested: 50..52 conflict with `learn`; 61..63 are now
    Certification3.
- **O3. Does CE64 get a named instruction set (ALU/ISA)?** The operations
  exist as functions but form no ISA: `pack`, setters, `forward`, `learn`,
  `syllogize`, `NarsTables::revise`/`deduce`, `pearl::reason`/`revise`/`hydrate`.
  Three conflicts:
  - `forward` and every constructor zero W (53..58) and bits 59..63;
  - the mantissa (46..49) has four readings (instruction, provenance,
    polarity, energy), and Counterfactual is encoded as both −1 and −6;
  - the mask and plasticity bit orders disagree.
- **O4. Contradiction on CE64.** There is no `contradicts(a, b)` and no
  demotion policy (pearl only raises certification). Recipes CR, CDI and CWS
  need contradiction to be a fact at register level.
- **O5. EWA as a trust propagator.** Σ' = M Σ Mᵀ is certified numerically in
  `jc`. Its use for trust is PROPOSED (`mul-ewa-trust-propagation-v1.md`), and
  its own falsifier F-MEP-0b says the suspicion score inverts under
  M = √trust·I. No recipe uses EWA.

**Why now:** D-CHESS-NARS-0 (stockfish-rs plan) needs recipes that act on
chess positions and are learned across games. The only learnable consequence
today is meta-state, and no recipe can narrow the move list.

## 1b. Premise audit result: the questions are re-asked (this governs)

The premise audit returned **PREMISE-WRONG** for the overall framing. Its
findings:

- **Three kinds of state.** "Thinking" in the code is three kinds of state
  with different cardinality and lifetime:
  - (E) per-edge belief: one `u64` CE64 per edge;
  - (T) per-thought / per-cycle scalars: the `ThoughtCtx` sd, free_energy,
    dissonance, temperature, confidence and rung;
  - (P) populations: `candidates`, `beliefs`.
- **Four "mask/EWA" artifacts, not one:**
  - the Moore 4×4 plasticity probe;
  - the mask-risc bit-plane vocabulary ("names no ISA", "no ORDER BY", A1
    forbids selection vectors);
  - the ndarray kernels;
  - EWA covariance propagation.
- **The lane-fold anchor** forbids routing through cognitive layers or the
  shader driver unless an explicit exception is made.

So the council answers these questions instead of O1–O5 as first written:

- **Q0. Carriers.** For each kind of state (E, T, P), which carrier matches
  its signature? Is "wire all thinking to CE64" right for E only? Options
  include "the property belongs elsewhere".
- **Q1a.** How is a candidate's identity carried? (An ordinal address on the
  population, not register state?)
- **Q1b.** Which carrier owns ranking:
  - a lane-fold-style permutation;
  - a semiring (`blasgraph` over CSR, "valued → semirings",
    PLAN-INVENTORY-2026-09-07.md:308);
  - a new position terminal;
  - or the kernel?
- **Q2.** Where do the per-thought scalars live? A cycle-context or side
  table? Are any of them genuinely per-row (`free_e`, `nars_c` in `MetaWord`)?
- **Q3a.** What closed operation set over a CE64 forms its instruction set,
  and which operations must preserve W (53..58) and bits 59..63?
- **Q3b.** Does any CE64 field encode an opcode? The mantissa's "instruction"
  reading is one of four.
- **Q4.** Split into four parts:
  - (a) a binary `contradicts(a, b)` predicate over S/P/O plus F;
  - (b) where a contradiction is stored (a pair relation);
  - (c) the scan over the belief population (mask `GroupReduce`?);
  - (d) the demotion policy.
- **Q5.** Does multi-hop trust belong to NARS deduction chains or semirings,
  with EWA purely attentional? Or is there a bounded monotone
  trust → step-transform mapping that makes EWA usable for trust?

H3 (confidence delta → C byte) is not this council's question. It moves to
the 5+3 with a prerequisite: name which edge is the subject of a recipe's
confidence effect.

## 2. Anchors (frozen; never re-litigate)
- **CE64 v2 layout** (`causal-edge/src/layout.rs:89-219`; all 64 bits
  const-asserted once). Bits 59..63 are `EpistemicState5`, "semantically
  opaque without a declared reading" (`edge.rs:1022`; `epistemic_state5.rs:43`).
- **I-VSA-IDENTITIES:** never XOR or bundle content registers.
- **Masks and magnitude are separate.** "a mask bit is never a weight"
  (mask-risc A2, `crates/lance-graph-mask-risc/src/lib.rs:76-80`;
  `E-TOPOLOGY-MASKS-MAGNITUDE-COMPOSE-NEVER-COLLAPSE-1`). mask-risc has "no
  ORDER BY" (`lib.rs:101-103`).
- **EWA:** "EWA ranks WHERE to look, never WHAT is true". It may not mint
  evidence. Do not store EWA covariance in CE64 (lineage addendum STOP 5).
- **Lane-fold capstone** (`.grok/board/CLAUDE_LANE_FOLD_CAPSTONE.md:18`):
  "Do not route this through cognitive layers, the shader driver, or ontology
  bridges."
- **Global-Sudoku plan:** no new CE64 bit allocation from that plan; no
  `Vec<Instruction>` recipes; no global Moore relaxation engine.
- **ogar-loco** `DOMAIN_FLOOR = 0x90` is a stored-byte ABI; the core
  `0x00..0x8F` is universal forever.
- **Zero-copy law; lance family upstream; AdaWorldAPI forks; no float/LLM on
  the hot path.**
- **Falsifiability rule (lance-graph CLAUDE.md).** Every guard needs both a
  can-fire test and a can-stay-silent test.

## 3. Internal surface
**lance-graph `e1bd42c`:**
- `crates/causal-edge/src/{edge.rs,layout.rs,pearl.rs,syllogism.rs,tables.rs,epistemic_state5.rs,network.rs}`
- `crates/lance-graph-planner/src/{pearl.rs,chain_replay.rs,chain_counterfactual.rs,chain_admission.rs,cache/nars_engine.rs,dismech_candidates.rs}`
- `crates/lance-graph-contract/src/{recipe_kernels.rs,recipes.rs,recipe_dispatch.rs,cognitive_shader.rs(MetaWord),revision.rs,sigma_propagation.rs}`
- `crates/lance-graph-mask-risc/src/{lib.rs,ir.rs}`
- `crates/cognitive-shader-driver/src/{driver.rs,mailbox_soa.rs}`
- examples `moore_plasticity_probe.rs`, `moore_nars_gomoku_probe.rs`, `staunen_scheduler_probe.rs`, `ce64_*_probe.rs`
- `crates/jc/src/ewa_sandwich*.rs`
- board entries `.claude/board/entries/2026-10-0[678]-*`
- `ISSUES.md`, `TECH_DEBT.md` (TD-THOUGHTCTX-IS-A-LOSSY-PROJECTION)

**OGAR `fca2380`:** `crates/ogar-loco/src/{lib.rs,vocabulary.rs,interpret.rs,nars.rs}`.

**ndarray:** `src/simd_masking_ops.rs`.

**stockfish-rs:** `examples/recipe_census.rs`, `docs/RECIPE-CENSUS-CHESS-V1.md`,
`docs/NARS-RECIPE-LOCO-PALETTE-INVENTORY-V1.md`, `src/policy.rs`.

## 4. External domains (seed terms)
- **Literature:**
  - NARS / Non-Axiomatic Logic (Wang), revision and contradiction handling;
  - register-based truth maintenance; belief revision (AGM) and contraction;
  - bitmap/bit-sliced ranking and top-k over bit-planes (bit-sliced index
    top-k, O'Neil & Quass);
  - "selection vectors"; learned move ordering;
  - EWA splatting (Zwicker 2001) and covariance propagation for uncertainty
    (unscented / Kalman trust);
  - Moore neighbourhood cellular automata;
  - tagged / typed-register ISAs (e.g. LISP machines, capability registers).
- **Systems:**
  - DuckDB selection vectors and top-N;
  - Roaring and bit-sliced index ranking;
  - PostgreSQL TID bitmaps;
  - Datalog engines with stratified negation (Soufflé);
  - truth-maintenance systems (JTMS/ATMS, e.g. de Kleer);
  - OpenNARS / ONA (OpenNARS-for-Applications) memory and contradiction;
  - Stockfish move picker (staged generation, scored move list with identity).
- **Concepts:**
  - formal concept analysis (attribute masks);
  - lattice / Boolean algebra (membership vs order);
  - paraconsistent logic (contradiction without explosion);
  - Belnap four-valued logic;
  - typed instruction sets.

Read first: `.claude/knowledge/research-council-semantics-papers-2026-06.md`,
and any knowledge doc on NARS, mask-risc or EWA (scouts find them).

## 5. What counts as an answer
For each of O1–O5:
- the options, each with a mechanism, evidence and the inside surface it
  would touch;
- which anchors each option would strain;
- one falsifiable probe that discriminates between the options.

A mapping table is enough; a design is out of scope. **"The property belongs
elsewhere" is a legitimate answer.** For example, the candidate population is
not register state, so "wire all thinking to CE64" may be the wrong framing
for it.

## 6. Budget
At most 10 items per scout. Literature: prefer foundational and recent
(2015–2026). Read-in-full or section-read for any mechanism claim.
