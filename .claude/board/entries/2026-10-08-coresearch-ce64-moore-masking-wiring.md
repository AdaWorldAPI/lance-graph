# 2026-10-08 — coresearch: wiring thinking onto CausalEdge64, Moore and the masking ops

**Status:** OPEN. Exploration map only; the council ratifies nothing.
PROBE-CF-MANTISSA and PROBE-W-PRESERVE are MEASURED and TEST-PINNED
(`crates/causal-edge/tests/ce64_op_contract.rs`, 8 tests; `cargo test -p
causal-edge --test ce64_op_contract`). Both predicted defects reproduce:
a −6 weight computes exactly the Synthesis average (128, 128) and is
re-stamped −6; `forward` returns W = 0 and 59..63 = 0 even when both operands
carry them. `learn` and the single-field setters preserve both. Each pinned
defect was disable-verified: adding a Counterfactual arm to `forward`, or
carrying W/Epi5 through it, turns exactly one test red.

Also MEASURED and TEST-PINNED, each disable-verified red:

- **PROBE-PREFILTER-MASK** (`cognitive-shader-driver/examples/meta_prefilter_mask_probe.rs`,
  `test = true`, 3 tests). `MetaFilter` lowers to a row MASK with only
  `ternary_match_u32_to_mask_under`, `mask_or_assign` and `mask_set_range`:
  a threshold `x >= k` on an n-bit packed field is a disjoint union of at most
  n + 1 ternary patterns, so no field is extracted. Release run: 400 random
  filters x 4099 rows identical to `meta_prefilter`; 393 admit under 1/3 of
  rows. Exhaustive per-field check over every (k, x). Disables: dropping the
  equality pattern (2 red); ungating the clauses (1 red). The shipped
  `meta_prefilter` is not changed yet.
- **PROBE-MOORE-PLANES** (2 tests added to `moore_plasticity_probe.rs`). The
  Moore SCHEDULE (which lanes have an on-grid neighbour per direction) equals
  `grid & shift(grid, -d)` built from `mask_shift_morton` on the D-MORTON-0
  reading (4 x 4 grid = codes 0..16 of one word), cross-checked against
  `Morton8x8::checked_offset`. Direction set sizes 9/12/9/12/12/9/12/9. The
  palette fold stays a LUT fold (value plane, A2). Can-fire arm: a swapped x
  axis is caught. Disables: swapped y axis (1 red); dropped grid clamp (1 red).
- **PROBE-CHAIN-CONF / PROBE-STAMP-GATE** (`lance-graph-planner/tests/chain_confidence.rs`,
  5 tests). The prediction was only partly right:
  - 5 deduction hops (f = c = 200) through `replay_step`: confidence
    200 -> 224 -> 237 -> 237 -> 237 -> 237. It rises and then saturates; it
    does not rise at every hop. `forward`'s own deduction falls at every hop
    on the same chain (reference arm).
  - Revising an edge with itself raises confidence (no evidential-base check
    on this path).
  - NEW: a weight with ZERO confidence still raises confidence, 128 -> 137.
    `NarsTables::revise` reads confidence through 16 bins and bin 0 carries
    evidence. `forward` gives 0 for the same weight (silent arm).
  - Disable (keep `forward`'s truth instead of the revised one): all 3 pinned
    tests flip, both reference arms stay green.

PROBE-ARGMAX-TIES has not run (needs a stockfish-rs score dump).

**Source:** stockfish-rs `docs/NARS-RECIPE-LOCO-PALETTE-INVENTORY-V1.md`
(stockfish-rs PR #21), open items O1–O5, re-asked after a premise audit as
Q0–Q5. Operator priority (2026-10-08): CE64 + Moore + masking first, with
cognitive-shader-driver reasoning kept in the loop on the newest masking ops.
Read alongside: `.arxiv/*` (17 notes) and `.grok/` (imported in #1348).

## The re-asked questions

"Thinking" is three kinds of state with different keys: **E** per-edge belief,
**T** per-thought scalars, **P** populations (candidates, beliefs). O1–O5 had
filed all three under "wire to CE64".

- Q0 carriers per kind · Q1a candidate identity · Q1b who owns ranking
- Q2 home of per-thought scalars · Q3a closed CE64 op set, what must survive
- Q3b does a CE64 field encode an opcode · Q4a–d contradiction predicate,
  storage, scan, demotion · Q5 multi-hop trust

## What the council found (one line each; grades as given by the scouts)

1. **Beside, not inside.** All three co-architects and every outside system
   read agree: scores sit beside an identity, never inside it; ranking is a
   separate operator; no new admissible meaning fits CE64 bits 53..63.
   `.arxiv/1608.07879` H2 says the same for responsibility ("not another CE64
   field").
2. **The carriers already exist.** E → `CausalEdge64`; T → `ThoughtCtx`
   (`recipe_kernels.rs:47`); P → `MailboxSoA.meta` / mask-risc masks
   (VERIFIED). OGAR D-META64: "CE64 is the edge". What is missing is wiring.
3. **`ThoughtCtx.candidates: Vec<f32>` drops identity.** stockfish-rs already
   carries `(MoveCode u16, frozen_score i32)` (`policy.rs:37-205`).
4. **`forward` reads its opcode from the operand.** It dispatches on the WEIGHT
   edge's mantissa (`edge.rs:~686`); −6 (Counterfactual) has no arm and runs
   the Synthesis average, then re-stamps −6 (VERIFIED by S3). Letter of
   D-PEARL-IO-0 holds (it governs bits 40..42); its spirit does not.
5. **`pack`, `forward`, `syllogize` silently zero W 53..58 and 59..63**;
   `learn` and the single-field setters preserve them (VERIFIED).
6. **`chain_replay::replay_step` revises on every hop**, so confidence rises
   along a chain (`chain_replay.rs:178`), with no stamp-disjointness guard
   (`BeliefArena` has one, `belief.rs:180-196`). `NarsTables::deduce` stores
   `c_out = f_out`, ignoring c1 and c2 (`tables.rs:61-71,126`) (VERIFIED).
7. **The shader driver's prefilter is a selection vector.**
   `BindSpace::meta_prefilter` (`cognitive-shader-driver/src/bindspace.rs:399`)
   loops row by row and returns a dense `Vec<u32>` of row indices. mask-risc A1
   forbids selection vectors, and the driver's `src/` calls none of ndarray's
   masking ops (closed search: `grep` over `src/` and `examples/`, only the
   crossword example uses them).
8. **Moore runs on its own bit loop.** `moore_plasticity_probe.rs` steps a
   `Register128` with hand-rolled neighbour masks and `trailing_zeros`, not on
   ndarray `mask_shift_morton` (VERIFIED). `.arxiv/2607.21273` H1: a Moore
   tenant is a predictor whose residual is `popcount(predicted XOR observed)`,
   which updates local LUT plasticity and must never rank actions.
9. **Contradiction is absorbed by revision** (Wang 1993, READ-IN-FULL; ONA,
   SECTION-READ); its pair carriers already exist twice
   (`Locus::Contradiction`, `unresolved_tension`). Demotion stays out:
   `pearl::revise` is raise-only and test-pinned.

## Exploration map

| idea | verdict | why |
|---|---|---|
| CE64 op-set hardening (finding 4, 5) | **PROBE** — PROBE-CF-MANTISSA, PROBE-W-PRESERVE | operator priority; runnable on shipped code |
| Hop confidence + stamp gate (finding 6) | **PROBE** — PROBE-CHAIN-CONF, PROBE-STAMP-GATE | predicted kills |
| Prefilter as a mask (finding 7) | **PROBE** — PROBE-PREFILTER-MASK | keeps the shader driver in the loop on `*_to_mask_under` |
| Moore on bit-planes (finding 8) | **PROBE** — PROBE-MOORE-PLANES | path invariance, `.arxiv/2608.00546` |
| Argmax set inside mask-risc (MaskedMaxI32 → EqI32 → Keep) | **PROBE** — PROBE-ARGMAX-TIES | yields a set; order stays a u16 permutation outside |
| `candidates` as `(u16, i32)` | **PROBE** after PROBE-ARGMAX-TIES | contract signature change |
| opcode as a `forward` argument | **PARK** until PROBE-CF-MANTISSA | call-site cost M |
| basin evidence side table home (D-CE64-CYCLE-0) | **PARK** | load-bearing, home is an operator decision |
| BSI top-k (Rinfret 2001) | **PARK** | a slice bit is a weight 2^k: breaks A2 unless a value plane read by a Terminal |
| covariance intersection for EWA | **PARK** | no fusion site exists |
| selection vectors (DuckDB) | **SKIP** | mask-risc A1 |
| entrenchment demotion | **SKIP** | D-PEARL-IO-0 raise-only (F6/F7) |
| Jøsang discount as a second truth calculus | **SKIP** | conflicts with NARS; fix deduce instead |

## Pre-registered probes (operator priority order)

Each has a can-fire arm and a can-stay-silent arm.

1. **PROBE-CF-MANTISSA** (`crates/causal-edge/tests/mantissa_dispatch.rs`).
   Running (f=255,c=255), weight (f=0,c=0) with mantissa −6 vs +5 (Synthesis).
   PASS: outputs differ. FAIL (predicted): identical. Can-fire: +1 (Deduction)
   differs from +5. Silent: +5 vs +5 identical.
2. **PROBE-W-PRESERVE** (`crates/causal-edge/tests/reading_preservation.rs`).
   Edge with W=0b101010, bits 59..63=0b10101 through each op. PASS: preserved
   or visibly stripped. FAIL (predicted for `pack`, `forward`): silent zero.
   Silent arm: `learn` and setters preserve.
3. **PROBE-CHAIN-CONF** (planner test). Five Deduction hops, f=c=200. PASS:
   c never increases. FAIL (predicted): c rises. Can-fire: a per-hop deduce
   reference falls monotonically.
4. **PROBE-STAMP-GATE**. `replay_step(E,E)` must not raise c. Can-fire:
   `BeliefArena` refuses a shared stamp; silent: disjoint stamps raise c.
5. **PROBE-PREFILTER-MASK** (cognitive-shader-driver). For random MetaWord
   columns and filters, a mask built from ndarray `*_to_mask_under` /
   `ternary_match_u32_to_mask_under` must set exactly the rows
   `meta_prefilter` returns. Anti-vacuity: filters that admit < 1/3 of rows.
   Silent: `MetaFilter::ALL` sets every in-range row.
6. **PROBE-MOORE-PLANES**. The `Register128` Moore step and a bit-plane step
   over `mask_shift_morton` give identical registers on the existing
   `moore_plasticity_probe` fixture. Can-fire: a wrong direction table breaks it.
7. **PROBE-ARGMAX-TIES**. MaskedMaxI32 → EqI32 → Keep, lowest set row, equals
   stockfish-rs `ordered_by(..)[0]` on ≥200 dumped positions; popcount equals
   the tie count. Anti-vacuity: fixture must contain ties.

## NOT searched

The normative CHERI spec; ATMS primary sources (secondhand only); Jøsang 2013
(403); the information/truth bilattice orderings; OGAR consumers outside the
local checkouts; ladybug-rs.

## Linked 5+3 (same session)

H1, H2, H5, H7 went to `/5plus3`. Its premise audit split H6 (the mantissa
carries both the rule that made an edge and the rung-3 tag); the iron-rule
savant found its committed fix writes bits 61..63 through the deprecated
`with_reasoning_band`. H6 therefore joins Q3b here and is not fixed there.
