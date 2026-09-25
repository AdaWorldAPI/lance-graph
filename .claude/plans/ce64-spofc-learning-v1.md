# ce64-spofc-learning-v1 — CE64 as the mantissa that carries SPOFC evidence between cycles

**Status:** PROPOSAL + first leg shipped (draft PR, stacked on #1293). Legs B–E are
blocked on the dependencies named in § Missing dependencies; none of them is
substituted here.

## Overview

CE64 was meant to keep what a cycle learned for the next cycle. In the live
dispatch path that role is dormant. #1293 made stage [3] collect each row's
supporting relationships into one SPOFC record (support count, best partner,
predicate union, `TruthU8`). This plan carries that record the rest of the way:
into the emitted CE64, back into the store, and into the next cycle's revision —
without recounting evidence the store already holds.

## Checklist

- [x] **D-CSL-A. Emit CE64 from SPOFC** (this PR): palette-space S/O, SPOFC truth,
      Pearl mask as the inverse of `edge_to_layer_mask`, topology by how the
      relation was found. Disable-verified tests.
- [x] **D-CSL-A2. Characterise `learn`** (this PR, `tests/ce64_recount_probe.rs`): it
      pools a repeated observation as fresh evidence; the driver's emission
      plasticity (`ALL_FROZEN`) stops it moving S/P/O.
- [ ] **D-CSL-B. Evidence identity.** Decide what tells "same evidence again" from "a
      new witness" when a stored edge is revised. Blocked: see § Missing, item 2.
- [ ] **D-CSL-C. Write-back through the owner path.** Blocked on W4a (item 1).
- [ ] **D-CSL-D. Cross-cycle revision.** After B and C: revise the stored edge with the
      cycle's SPOFC observation; the probe in A' flips deliberately.
- [ ] **D-CSL-E. Completions ("Sudoku autocomplete").** One-way probe only, per
      `mul-ewa-trust-propagation-v1.md` §0b; inferred edges marked by topology.
- [ ] **D-CSL-F. Support entropy.** Shannon entropy over a candidate's support
      distribution, as a spread signal beside `m`. New code; nothing like it exists.

## Verified facts this rests on

CE64 layout (default `causal-edge-v2-layout`, `causal-edge/src/layout.rs`, const-asserted
to cover all 64 bits once):

| bits | field |
|---|---|
| 0–23 | S, P, O — three palette256 indices |
| 24–31 / 32–39 | NARS frequency / confidence, u8 |
| 40–42 | Pearl causal mask |
| 46–49 | inference mantissa (signed i4) |
| 50–52 | plasticity (`0` = `ALL_FROZEN`, `plasticity.rs:17`) |
| 53–58 | W-slot: witness **corpus** root handle, 0..=63 |
| 59–60 | `CausalTopology`: Direct 0, IndirectKnownIntermediates 1, IndirectUnknownIntermediates 2, Unknown 3 |
| 61–63 | `ReasoningBand` |

- **`pack` defaults bits 59–60 to 0 = `Direct`.** Every edge emitted before this
  PR claimed a direct relation, including content-similarity partners. The two
  bits share their ordinals with `TrustTexture`; the last writer wins.
- **The 24-bit S/P/O can carry the target.** The emission simply never wrote it:
  it packed a row id into palette space (`s = row%256`, `o = (row/4)%256`).
- **`predicates & 0b111` was not the inverse of `edge_to_layer_mask`**
  (`p64-bridge/src/lib.rs:61`): plane 2 is SUPPORTS, but Pearl bit 2 reads back as
  CONTRADICTS. A supporting relation round-tripped as a contradiction.
- **`learn` (`edge.rs:731`) pools unconditionally**: `c = ws/(ws+1)` with
  `ws = w_self + w_obs`, no source identity. Revising twice with one observation
  counts it twice (pinned in A').
- **Nothing in `run` writes an edge back.** `persist_cycle`
  (`engine_bridge.rs:784`) writes only `emitted_edges[0]`, has test-only callers,
  and `ShaderDriver.bindspace` is `Arc<BindSpace>`. The only stored-edge read that
  crosses cycles is the cascade query `backing.edge(row).s_idx()`.
- **`tables.revise(...)` is computed and discarded** in stage [3]
  (`_revised_truth`).
- **Confidence convention matches arm-discovery; frequency does not.** SPOFC
  `c = m·255/(m+k)` is `evidence_confidence_u8`, shared with `arm_to_truth_u8`.
  SPOFC `f` is the best resonance; arm-discovery's `f` is a conditional ratio
  `cooccur/antecedent`. There is no antecedent count in a dispatch, so the ratio
  is not available; the difference is recorded, not papered over.

## The "Sudoku" framing, stated against the code

- **8× support = 8× evidence.** In SPOFC terms a row supported eight ways has
  `m = 8`, `c = 226/255`. That is the evidence a completion is constrained by.
  It is only honest if the eight are independent; the pre-pass credits each
  content pair to both rows, and cascade hits through one shared palette target
  are counted per row. Independence is item B.
- **Bits 59–60 as the grid folding itself.** Topology is the one place a CE64
  says how it was obtained: `Direct` (observed plane edge), `IndirectKnown`
  (a completion whose intermediates are named — its derivation), `IndirectUnknown`
  (a fill with the path unknown), `Unknown` (similarity, no causal path). An
  inferred completion must land as `Indirect*`, never as `Direct`, so a later
  cycle cannot read it back as an observation. This PR is the first writer of
  bits 59–60 in the live path (Direct / Unknown only; nothing is inferred yet).
- **Oberflächenspannung / EWA.** `jc::ewa_sandwich` (`crates/jc/src/ewa_sandwich.rs`,
  contract twin `sigma_propagation::ewa_sandwich`) is SPD covariance push-forward,
  `Σ' = M·Σ·Mᵀ`. It propagates uncertainty *shape* and counts no evidence. Its
  only epistemic use is the PROPOSED `mul-ewa-trust-propagation-v1.md`, which
  names the circularity: Σ → trust → gate → Σ. A fill by a minimisation principle
  is "plausible everywhere, grounded nowhere", so a completion from it must never
  raise confidence as if observed. The learned prior it could supply is the
  surface; the evidence stays the SPOFC count. The driver's "Kerbl EWA"
  (`alpha_front_to_back_composite`) is scalar alpha compositing and is unrelated.
- **Shared upstream.** Two supports that trace to one root are one witness
  (`lance-graph-contract/src/fusion.rs` anti-alchemy law; `shared_roots`,
  `inherited_roots`). `nars::belief::revise_at` pools only on disjoint, non-empty
  stamps and otherwise falls back to CHOICE; `admit_derived` admits derivations
  with an empty stamp so they never pool. That is the discipline item B needs;
  where its carrier lives in the dispatch path is the open question.

## Missing dependencies (named, not substituted)

1. **The W4a owner write path.** `COMPONENT-MAP.md` §6: `persist_cycle` /
   `dispatch_busdto` are BLOCKED→W4a, the batch writer pairing
   `cast(on_behalf = mailbox_owner())`. A write-back that bypasses it violates the
   V3 ownership rule. Item C waits for it.
2. **An evidence-identity carrier per stored edge.** `learn` has none. The W-slot
   (bits 53–58) is a witness *corpus root*, not a per-observation source set, and
   `le-contract.md` forbids new awareness semantics in CE64 bits. `nars::Stamp`
   exists in the planner, not in the driver's store. Where the stamp lives (a SoA
   lane beside the edge, per V3) is a design decision this plan does not make.
3. **A predicate palette.** P stays 0 because nothing assigns a palette index to a
   predicate. Using the p64 plane index would be inventing one.
4. **A production feed into the planes.** `update_planes` has no production caller;
   `edges_to_layered_rows` has none outside p64-bridge, and `convergence.rs`
   addresses `%64` where p64-bridge uses `/4`. Closing the topology loop waits on
   that being reconciled.

## Effects of leg A

Emitted edges change for every dispatch (S/O address space, confidence, Pearl
bits, topology). `cycle_fp` and top-k are untouched: the braid and ranking read
the candidates, not the edges.
