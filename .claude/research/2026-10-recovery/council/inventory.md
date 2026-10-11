# NARS-RECIPE-LOCO-PALETTE-INVENTORY-V1 — the 34 recipes and the 256-slot palette, routed

> Status: INVENTORY (read from source, not measured unless a row says so).
> Gate for D-CHESS-NARS-0b and for wiring thinking onto `CausalEdge64`.
> Sources, all on current `main` on 2026-10-08: lance-graph `e1bd42c`
> (`recipe_kernels.rs`, `recipes.rs`, `recipe_dispatch.rs`, `causal-edge`,
> `lance-graph-mask-risc`, `jc`, `sigma_propagation.rs`, board entries),
> OGAR `fca2380` (`ogar-loco`), ndarray `main` (`simd_masking_ops.rs`).
> The chess census column is the measured result of
> `examples/recipe_census.rs` (`docs/RECIPE-CENSUS-CHESS-V1.md`).

## What this answers

For every one of the 34 NARS recipes and every one of the 256 palette slots,
it answers four questions:

1. Can the operation live on the `CausalEdge64` register today?
2. Does it belong on the mask algebra instead? That means the Moore-neighbourhood
   fold and the `lance-graph-mask-risc` / `ndarray::simd` bit-plane algebra. The
   EWA sandwich is the uncertainty propagator alongside them.
3. Is the open question research-shaped? Those go to `/coresearch`.
4. Is it concrete enough to specify and harden? Those go to `/5plus3`.

## Headline findings

1. **Only one recipe fits CE64 today: #7 ASC**, a read and saturating update of
   the C byte (bits 32..39). 21 recipes operate on the candidate list, which is
   a population and not a register. 10 depend on thinking scalars with no CE64
   bits: free energy, sd, dissonance, temperature, rung. 3 are no-op
   Demonstrations.
2. **The thinking layer and the register share no data in production.** No
   code builds a `ThoughtCtx` from a CE64 or writes a kernel result into one.
   The board already records the cause: `E-THE-RECIPE-SURFACE-IS-CAUSALLY-BLIND-1`
   ("`lance-graph-contract` … depends on `causal-edge`? **No dependency.**").
3. **CE64 has no ALU or ISA.** The register operations exist as separate
   functions: `pack`, setters, `forward`, `learn`, `syllogize`,
   `NarsTables::revise`, and `pearl::{reason, revise, hydrate}`. Nothing names
   them as one instruction set, and the phrase "ALU ISA" appears nowhere in
   code or board entries for CE64.
4. **None of the 256 palette slots reads or writes CE64.** ogar-loco has zero
   hits. lance-graph-ogar mentions it in two doc comments only.
5. **No recipe propagates anything across hops, so none has an EWA
   counterpart.** The EWA sandwich (Σ' = M Σ Mᵀ, certified in `jc`) has no
   production caller. Its trust use is a proposal, and that proposal's own
   falsifier says the suspicion score comes out inverted (F-MEP-0b).
6. **The mask algebra can express the membership half of most population
   recipes, but not the ordering half.** It cannot express ranking:
   - mask-risc lanes are integer only;
   - mask-risc has "no ORDER BY";
   - no terminal returns a position (no masked argmax).

   The chess census's identity finding is the same gap seen from the other
   side.
7. **Two vocabularies mint the same 34 bytes `0x90..=0xB1` with different
   specs.** `ogar_loco::nars::NarsVocabulary` gives Datapath recipes arity 2;
   `lance_graph_ogar::recipe_vocab::RecipeVocabulary` gives every recipe arity 1
   plus an attention-focus value codebook.

## The 34 recipes

Column key:

- **Chess**: the measured chess census.
  - app = applicable
  - nwy = not wired yet
  - red = redundant
  - n/a = not applicable
  - unr = unreachable
  - U-id = unsafe, loses move identity
  - U-sem = unsafe, wrong effect for chess
- **Gomoku**: the lance-graph D-MOORE-NARS-0 census, a different domain.
  Disagreement is expected; it is shown so that neither census is mistaken
  for a property of the kernel.
- **CE64**:
  - NATIVE: fits existing register operations.
  - NEEDS-OP: needs an operation or field that does not exist.
  - NOT-REG: a population operation.
  - Every `delta_conf` (14 recipes) could land on the C byte.
- **Mask**: the nearest mask-risc / ndarray counterpart; "none" = no counterpart.
- **Route**: this inventory's routing decision.
  - CE64: wire to the register now.
  - MASK: migrate the population half onto mask algebra.
  - O1..O5: open, goes to `/coresearch`.
  - H1..H7: concrete, goes to `/5plus3`.
  - retire: not for chess.

| id | FnIndex | code | Chess | Gomoku | CE64 | Mask counterpart | Route |
|---|---|---|---|---|---|---|---|
| 1 | 0x90 | RTE | red | redundant | NEEDS-OP (FE, rung have no bits) | none | retire for chess (iterative deepening is search's); O2 |
| 2 | 0x91 | HTD | U-id | not wired | NOT-REG | `Pred::GeI32` membership masks; the reorder has none | MASK (membership) + O1 (order) |
| 3 | 0x92 | SMAD | app | redundant | NOT-REG, Δc → C | `MaskedMaxI32`, `MaskedMinI32` | MASK + H3 |
| 4 | 0x93 | RCR | U-id | realized | NOT-REG | none (order op) | O1; concept maps to the CE64 abduction mantissa |
| 5 | 0x94 | TCP | U-id | realized | NOT-REG, Δc → C | mean via `MaskedSumI32`/`Count`, then `Pred::GeI32` → `Keep` | MASK + O1 (identity, integer lane) |
| 6 | 0x95 | TR | red | realized | NOT-REG | none (a mask bit is never a weight) | retire (`ExplorationPolicy` owns it) |
| 7 | 0x96 | ASC | nwy | realized | **NATIVE** (C byte) | none | **CE64** + H3 |
| 8 | 0x97 | CAS | app | redundant | NOT-REG | none (value quantisation) | keep as kernel; no CE64, no mask |
| 9 | 0x98 | IRS | n/a | unreachable | NOT-REG | none | retire |
| 10 | 0x99 | MCP | app | chooser | NEEDS-OP (FE has no bits) | none | O2 + H3 |
| 11 | 0x9A | CR | app | realized | NEEDS-OP (`contradicts`) + population scan | `GroupReduce` by topic, `MinI32`/`MaxI32`; `EvidenceMask` | O4 |
| 12 | 0x9B | TCA | U-id | not wired | NOT-REG | none (`mask_shift_morton` is a lattice shift, not a list rotation) | retire for chess |
| 13 | 0x9C | CDT | U-id | chooser | NOT-REG, Δc → C | argmax needs a position terminal | O1 |
| 14 | 0x9D | MCT | n/a | unreachable | NOT-REG | `MaskedSumI32` + `Count` | retire for chess |
| 15 | 0x9E | LSI | app | unreachable | NOT-REG (writes sd) | `GroupPowerSumsI32` | MASK + O2 (sd home) |
| 16 | 0x9F | PSO | U-id | chooser | NOT-REG | none (no ORDER BY) | retire (move ordering owns it) |
| 17 | 0xA0 | CDI | U-sem | unsafe | mixed: mint is NATIVE, append is a population op, dissonance has no bits | none | O4 |
| 18 | 0xA1 | CWS | U-sem | chooser | NOT-REG (argmax, topic `u32::MAX` has no SPO) | argmax needs a position terminal | O1 + O4 |
| 19 | 0xA2 | ARE | unr | unreachable | NOT-REG (Demonstration) | `MaskOp::Xor` | H4 |
| 20 | 0xA3 | TCF | U-id | realized | NOT-REG, Δc → C | none (no median, no ORDER BY) | O1 |
| 21 | 0xA4 | SSR | app | redundant | NEEDS-OP (FE has no bits) | none | O2 + H3 |
| 22 | 0xA5 | ETD | U-id | not wired | NOT-REG | none | O1 |
| 23 | 0xA6 | AMP | app | chooser | NEEDS-OP (FE, rung have no bits) | none | O2 |
| 24 | 0xA7 | ZCF | unr | unreachable | NOT-REG (Demonstration) | `MaskOp::Xor` | H4 |
| 25 | 0xA8 | HPM | U-sem | realized | NOT-REG | argmin needs a position terminal | retire (semantics) |
| 26 | 0xA9 | CUR | U-id | realized | NOT-REG, Δc → C | loop of mean → `GeI32` → `Keep`, ends at `Count == 1` | MASK + O1 |
| 27 | 0xAA | MPC | n/a | redundant | NOT-REG | mean; per-bit majority is ternlog `MAJ3` (3 inputs only) | retire for chess |
| 28 | 0xAB | SSAM | nwy | not wired | NEEDS-OP (sd has no bits) | none | O2; real analogy is D-CHESS-NARS-0b's job |
| 29 | 0xAC | IDR | U-id | unreachable | NOT-REG | masked argmax with ordinal tie-break (does not exist) | O1 |
| 30 | 0xAD | SPP | nwy | redundant | NOT-REG, Δc → C | `MaskedSumI32`, `Count`, `MaskedMax/MinI32` | MASK + O2 (real paths = NNUE vs search = dissonance) |
| 31 | 0xAE | ICR | nwy | realized | NEEDS-OP (FE) | none | O3 (real path is `pearl::reason`, mask = SPO) + H6 |
| 32 | 0xAF | SDD | U-sem | unreachable | NOT-REG, Δc → C | mean as MCT | retire (semantics) |
| 33 | 0xB0 | DTMF | app | chooser | NEEDS-OP (temperature, sd have no bits) | none (`GateState` is a scalar) | O2 |
| 34 | 0xB1 | HKF | unr | unreachable | NOT-REG (Demonstration) | `Ternlog` XOR3 | H4 |

Route tally:

| route | recipes |
|---|---|
| CE64 now | 7 ASC |
| MASK (membership half) | 2, 3, 5, 15, 26, 30 |
| retire for chess | 1, 6, 9, 12, 14, 16, 25, 27, 32 |
| keep as kernel | 8 |
| O1 (candidate identity and ranking) | 2, 4, 5, 13, 18, 20, 22, 26, 29 |
| O2 (thinking scalars with no register home) | 1, 10, 15, 21, 23, 28, 30, 33 |
| O3 (CE64 instruction set) | 31 |
| O4 (contradiction on CE64) | 11, 17, 18 |
| H3 (C-byte sink for `delta_conf`) | the 14 `moves_confidence` recipes; 3, 7, 10, 21 named above |
| H4 (Demonstrations) | 19, 24, 34 |
| H6 (Counterfactual encoding) | 31 |

Some recipes carry two routes; every recipe carries at least one.

## The 256-slot palette

`FnIndex(u8)` indexes a per-scope codebook. `VocabularyTable { specs: [FnSpec; 256] }`
is composed in two halves by `compose` (`vocabulary.rs:635-660`):

- **`0x01..=0x8F`, the shared core.** Minted only by ogar-loco and identical in
  every vocabulary; "the universal ABI forever" (`lib.rs:337-349`). A
  vocabulary cannot influence it.
- **`0x90..=0xFF`, vocabulary-local.** Selected by the node's classid.
  `DOMAIN_FLOOR = 0x90` is a compile-time-pinned stored-byte ABI constant
  (`lib.rs:353-361`).
- **`0x00`** is NOP, the zero fallback, composed as refused.

Engine column: ENG = `Interpreter` executes it; REF = refused
(`UnhandledControlFlow`); DIA = handed to a `Dialect`. ogar-loco ships no
production dialect.

### Shared core

| band | slots | named | contents |
|---|---|---|---|
| control `0x01–0x1F` | 31 | 14 | IF, IF_ELSE, REPEAT, REPEAT_UNTIL, WHILE (**ENG**); FOREVER, FOR_EACH, FOR_RANGE, WAIT, WAIT_UNTIL, STOP, BREAK, CONTINUE, RETURN (REF); `0x0F–0x1F` reserved (17) |
| logic `0x20–0x2F` | 16 | 7 | AND, OR, NOT, TRUE, FALSE, NULL, TERNARY (DIA); `0x27–0x2F` reserved (9) |
| comparison `0x30–0x3F` | 16 | 6 | EQ, NEQ, LT, LTE, GT, GTE (DIA); `0x36–0x3F` reserved (10) |
| math `0x40–0x5F` | 32 | 30 | ADD SUB MUL DIV POW MOD NUMBER ABS NEG ROUND FLOOR CEIL SQRT LN LOG10 EXP_E EXP_10 SIN COS TAN ASIN ACOS ATAN ATAN2 RANDOM_INT RANDOM_FLOAT CONSTRAIN NUMBER_PROPERTY CONSTANT ON_LIST (DIA); `0x5E–0x5F` reserved (2) |
| text `0x60–0x6F` | 16 | 16 | TEXT JOIN LENGTH CHAR_AT INDEX_OF IS_EMPTY SUBSTRING CHANGE_CASE TRIM CONTAINS APPEND PRINT PROMPT COUNT REPLACE REVERSE (DIA) |
| list `0x70–0x7F` | 16 | 16 | LIST_EMPTY LIST_WITH LIST_REPEAT LIST_LENGTH LIST_IS_EMPTY LIST_INDEX_OF LIST_GET LIST_SET LIST_INSERT LIST_ADD LIST_DELETE LIST_DELETE_ALL LIST_SUBLIST LIST_SPLIT LIST_SORT LIST_CONTAINS (DIA) |
| var / proc / mask `0x80–0x8F` | 16 | 7 | VAR_GET VAR_SET VAR_CHANGE PROC_DEF PROC_CALL PROC_ARG (DIA); **TERNLOG `0x86`**: pops 3 masks, value byte = 8-bit truth table, no executor in ogar-loco (doc points at ndarray `mask_ternlog`); `0x87–0x8B` retracted band (5), `0x8C–0x8F` reserved (4) |

Census, pinned by `vocabulary.rs:1434`:
- 96 named core functions;
- 5 of those are deliberately refused (variadic: RETURN, NUMBER_PROPERTY, PROMPT, LIST_WITH, PROC_CALL);
- 47 reserved core slots.

### Domain half (`0x90–0xFF`), with the recipe vocabulary plugged in

| range | count | contents |
|---|---|---|
| `0x90–0xB1` | 34 | the NARS recipes, `op_of(id) = 0x90 + id − 1` (table above) |
| `0xB2–0xFF` | 78 | unminted, composes `FnSpec::REFUSED` |

Other vocabularies occupy `0x90+` under their own classids: `ogar-dismech`
predicates (`0x90 "causes"` …), `ogar-r2il`, `ogar-ro`, `blockly-rs`, and
MedCare `loco_vocab`. Those are other palettes and are not inventoried here.

Total: 1 NOP + 96 named + 47 reserved + 34 recipes + 78 unminted = **256**.

### Palette vs CE64 and the mask algebra

| question | answer |
|---|---|
| slots that read or write CE64 | **none** |
| slots that carry NARS truth | none in ogar-loco (`nars.rs` holds bucket, arity and name only); recipe kernels carry `ThoughtCtx.confidence` / `beliefs` |
| mask-algebra slots | TERNLOG `0x86` (combinator) and the logic band, as booleans; the 9 "Datapath = masking op" recipe slots are labelled mask ops by `nars.rs`, but 6 of their kernels are not mask operations (H5) |
| slots the engine itself executes | 5 (IF, IF_ELSE, REPEAT, REPEAT_UNTIL, WHILE) |

## Hardening cases

### Open → `/coresearch`

- **O1. Candidate identity and ranking.**
  - *Question:* how does a population of scored candidates keep the move each
    score belongs to, and how does it rank? The mask algebra has membership
    but no ordering: no ORDER BY, no position terminal, integer lanes only.
    `ThoughtCtx.candidates` has neither identity nor order.
  - *Options on record:*
    - kernels return a selection mask or permutation over input indices (chess census);
    - a masked argmax with an ordinal tie-break, plus the score as a lane (DAV round 2 entry);
    - stay in the kernel.
  - *Blocks:* 9 recipes, including every move-narrowing one.
- **O2. Thinking scalars with no register home.**
  - *Which scalars:* free energy, sd, dissonance, temperature, rung.
  - *Candidate homes:*
    - `MetaWord` (thinking 6, nars_f 8, nars_c 8, free_e 6), a sibling column;
    - a side table beside the register (D-CE64-CYCLE-0 measured that the
      evidence table and attention state are needed, and neither is in the register);
    - new bits. These are contested: bits 50..52 conflict with `learn`, and
      bits 61..63 are now `Certification3`.
  - *Blocks:* 8 recipes.
- **O3. Does CE64 get a named instruction set?** The register operations exist
  but are not one ISA, and three of them conflict:
  - `forward` and every constructor destroy W and bits 59..63;
  - the mantissa has four readings;
  - the mask and plasticity bit orders disagree.

  Wiring "all thinking" to the register first needs an answer to what a
  register instruction is.
- **O4. Contradiction on CE64.**
  - No `contradicts(a, b)` operation exists.
  - No demotion policy exists (`pearl` only raises certification).
  - Recipes 11, 17 and 18 need a contradiction to be a register-level fact,
    not a scan over a `Vec`.
- **O5. EWA as a trust propagator.**
  - No recipe uses EWA today.
  - The proposal (`mul-ewa-trust-propagation-v1.md`, PROPOSED) has an open
    inversion falsifier (F-MEP-0b) and a hop-count domination falsifier (F-MEP-0c).
  - Kept on the list because "wire all thinking" will reach uncertainty
    propagation, and EWA is the only multi-hop propagator in the stack.

### Concrete → `/5plus3`

- **H1.** Reconcile the two vocabularies minting `0x90..=0xB1` (arity 2/1/1 vs 1
  everywhere, value codebook vs none) into one spec.
- **H2.** PROC_DEF (`0x83`) lies outside `CONTROL_BAND`. It is therefore handed
  to the dialect, contradicting the doc that says it is refused
  (`interpret.rs:157,178,308-321`). Untested, read from source.
- **H3.** The `delta_conf` → C-byte sink. Define the saturating C update that 14
  recipes' confidence effects would land on, with ASC as the first NATIVE
  wiring.
- **H4.** The three Demonstrations, ARE, ZCF and HKF: wire them or delete them
  (the contract's own open decision).
- **H5.** `nars.rs` labels the 9 Datapath recipes "masking ops". The kernels
  for TCA, MCT, MPC, SDD, HPM and SSAM are not mask operations as written.
  Fix the label or the kernels.
- **H6.** Counterfactual has two mantissa encodings, −1 (`network.rs:209`) and
  −6 (`chain_counterfactual.rs:253`). Pick one.
- **H7.** CE64 doc drift: `lib.rs:9-15` and `edge.rs:138-150` still show the v1
  layout, with temporal in bits 52..63.

## What this does not decide

- No recipe is wired, migrated or deleted here.
- Every route above is a proposal for the councils.
- The chess and gomoku censuses use different classes and often place a
  recipe differently. RCR, TCP and CUR, for example, are "realized" in gomoku
  and identity-unsafe in chess. That is a property of the domains, not an
  error in either, and no agreement count is claimed because the class sets
  do not map one-to-one.
- No change is filed against lance-graph or OGAR. The H items name upstream
  code; whether to change it is an operator decision.
