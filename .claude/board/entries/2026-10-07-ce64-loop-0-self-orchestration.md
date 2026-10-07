# 2026-10-07 — D-CE64-LOOP-0: a CE64 register can pick its next fold from local feedback; bits 40..63 census

**Status:** TEST-PINNED (`crates/lance-graph-planner/examples/self_orchestration_probe.rs`, `test = true`, 15 tests, run by the `member-tests` job). MEASURED for the numbers below (`cargo run --release -p lance-graph-planner --example self_orchestration_probe`, 4-core Xeon 2.8 GHz).

## DECISION

- **SCOPE:** a probe, not a production path. One `CausalEdge64` register `R` crosses cycles; each cycle selects a fold, runs an existing operator (`pearl::hydrate`; `pearl::reason` + `revise` under SO / PO / SPO-cut / SP), updates a posterior over five declared alternatives (direct, or via one of four candidate intermediates), and carries `R` forward.
- **H** is Shannon entropy over that posterior. **Activation** is the signed change, in sevenths, of the mass of the alternative the fold tested (or, for a Pearl fold, the certification rank it earned). Both live in a transient trace, **never in the register**.
- **Epi5 moves only through `revise` / `hydrate`.** Neither H nor activation is an input to them. A test pins that across a whole run only bits 40..42 and 59..63 differ from the start register; witness (17), mantissa and plasticity are untouched.
- **No phase variable.** The selectors read the posterior, `R`'s Epi5 and per-fold activation only; regimes are read off the trace afterwards.
- **BASIS:** the likelihood tables and the combined score `EIG + (1 − H/Hmax)·headroom + 0.25·activation(cut opened)` are **policy pins**, not measurements.

## Measured (one sealed fixture, cap 40)

| policy | folds | to goal | redundant | noise | Epi5 moves |
|---|---|---|---|---|---|
| round-robin | 40 | 10 | 28 | 3 | 3 |
| random (200 seeds) | 40 | median 14, 197/200 | — | — | — |
| entropy-only | 5 (stops) | never | 0 | 0 | 1 |
| activation-only | 11 (stops) | 8 | 0 | 1 | 3 |
| **combined** | **6 (stops)** | **3** | 0 | 0 | 2 |

Goal = H ≤ 0.5 bit **and** Epi5 = `IndirectKnown × Causes`. Entropy-only never reaches it because the Pearl folds carry no information about the mediator: certification needs a term entropy cannot supply.

Throughput (combined, release, episodes back to back): **~830 ns/fold**, ~0.67 allocations/fold, **~420k folds per 350 ms**. 1M folds in 350 ms would need ~350 ns/fold. The sweep runs many 6-fold episodes, not one long run. No SIMD is involved, and the figure includes selection.

## Census of bits 40..63 (production code, excludes tests/examples)

- **40..42 mask:** an instruction (`pearl::reason` selects its operator by it). The driver also writes `h.predicates & 7` there, a p64 *layer* mask in a different bit order (layer 0 CAUSES → O; content-match 0x01 → O).
- **43..45 direction:** no production producer measures it; `forward` copies the weight, everyone else writes 0.
- **46..49 mantissa:** four readings — instruction (`forward`, driver style), provenance (`syllogize` rule, −6 counterfactual terminal), polarity (`to_spo`), energy (`MailboxSoA::apply_edges`). "Counterfactual" has two encodings: −1 (`CausalNetwork::counterfactual`) and −6 (`counterfactual_replay`). `forward` normalisation is lossy (0 → Deduction). `nars_engine::from_causal_edge` maps protocol ordinals 5/6 to local Resemblance/Synthesis.
- **50..52 plasticity:** feedback state under `learn`. Defaults are opposite between producers (ALL_HOT vs ALL_FROZEN), and its S bit is bit 0 where the mask's is bit 2.
- **53..58 W:** routing key for `apply_edges`; no production producer sets it, and `pack` (so `forward`) zeroes it.
- **59..63 Epi5:** the only earned field (`revise`, `hydrate`); every constructor writes 0 = `Direct × Open`, an asserted fact.
- **Must survive R[k] → R[k+1]:** 59..63 (earned), 53..58 (addressing), 50..52 (if `learn` runs), the −6 / syllogism provenance tags in 46..49. 43..45 and the instruction echoes are recomputable.

Not canonized: the alternative readings above stay as found; this entry records them.

## Gates

11 disable runs, each red on its named falsifier: witness counter, entropy promotes, activation promotes, contradiction read as settled, zero-gain folds run, no fixed point, pseudo-replication, transition from activation, combined = round-robin, selection reads the cycle index, wall-clock seed.

## OPEN

- Only one fixture; the ranking of policies on a second, harder population is unmeasured.
- The likelihood tables are invented; nothing calibrates them against outcome rates.
- `forward` and every constructor destroy W and Epi5 (census §8); a register carried through `forward` would lose exactly what this loop needs to survive.
- The allocation source (~0.67/fold) is not identified.
- No runtime caller: nothing in production runs this loop.
