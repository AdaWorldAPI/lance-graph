# A10 epistemic admission and NNUE compiler-heuristic feasibility (two independent findings, 2026-11-02)

**Status:** research findings. Nothing implemented; no CE64 change; /5plus3 deferred. Operator asked for two independent findings.

## Finding 1 — A10 epistemic writer / witness

### Answer to the question asked
Can the existing 53..58 witness contract plus the publication frame distinguish an honestly stamped base fact from an unwritten zero, without altering CE64?
- Inside the CE64 word: NO.
  - A hydrated base fact is `W = 0` (givens layer) and `raw5 = 0` (Direct × Open). An edge nobody stamped carries the same bits.
  - A sealed W is arbitrary by decision ("CE64 is an ISA contract, not a storage contract; slot numbers recycle", capstone §8 row 4, DECIDED 2026-10-20).
  - W is a session-local lane index: owner = the merge/materialization step; lifetime = one session (hot path); no generation; validity ends at seal. It is not a witness identity and must not be promoted to one.
- With existing owners OUTSIDE the word: YES, and that is where the contract says it belongs. `epistemic_state5.rs:36-39`: "observed vs asserted belongs to witness/provenance, never to one of these 32 cells."
  - **Hot path:** presence. The per-row 64-bit written-by word names the lanes that wrote the row. An unclaimed alpha address reads `None` and never falls back to a base value (`alpha.rs`). The givens probe already relies on this ("a row carrying only givens is never read past the written-word check").
  - **Sealed:** a row exists only if it was written. The hazard is a writer that builds a composed edge without stamping 59..63. The canonical writer `with_epistemic_raw5(u8)` accepts any code (causal-edge is zero-dependency and cannot check the codebook), so nothing forces a writer to state what it stamped.
- Conclusion: NO CE64 change is justified. Reserved Certification3 codes 6/7 stay unused. `raw5 = 0` stays valid Direct × Open.

### The six concerns, their existing owners, and the gaps

| concern | existing owner | state |
|---|---|---|
| source assertion / direct observation | revision `EncounterEvidence.independent_roots`; `SupportLedger` receipt (source, at, strength); givens = thought layer 0, written only by hydration | contract types exist; givens is a probe (`givens_layer_probe`, measured 34 ns/row) |
| mechanical / logical derivation with rule + premise identity | `inherited_roots`, `BasisView.ancestry_derived_roots`; planner `nars::belief::Stamp` (source set; conservative modulo-64 folding) | premise ROOTS carried; RULE identity is not carried anywhere (tarski register Q4: route identity varies with HashMap order, unresolved) |
| independent roots / duplicate-support protection | Stamp overlap gate (`p_e2_stamp_gate_probe`, test-local); D-HL-3 per-delta evidential base (planned) | GAP in production: `Opcode::Revision.truth_code` has no stamp, and duplicated evidence raises c (P-E2 measures it) |
| maturation (Maslov) | bits 50..52; "maturation never writes F/C, W or 59..63; certification never rises because a rival was ruled out" | test-local (`maturation_ladder.rs`); 50..52 reading still contested (O2) |
| certification of relational claims | 59..63 written by hydration (capped by the proofs that ran) and by measured results only; Gadamer writer (D-CTX-7 probe) | GAP: no production writer of 59..63 from evidence (ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE) |
| publication validity / frame-qualified witness identity | `BindStore::check_route` / `RouteVersion` (D-TGL-PUB); `SupportReceipt.at: DatasetVersion` | `at` is a storage revision, not an epistemic view (named gap in `causal_audit.rs`). Witness identity = (classid, sealed version / route version, NodeGuid row), with the thought lane meaningful only inside a session |

### Recommendation (mandatory = verification of the required evidence, not passage through every stage)

**One writer-side admission step, in the contract, in front of `with_epistemic_raw5`.**
- A write of 59..63 must state its admission basis as a typed value, one of:
  - `Given` (hydration, layer 0);
  - `Observed { roots }`;
  - `Derived { rule, premises, roots = union(premise roots) }`;
  - `Measured { test }`.
- The certification it may claim is capped by that basis. A derivation inherits roots and earns no new one.
- Readers keep the presence check: no written-by bit / unclaimed means `None`, never Direct × Open.
- Enforcement is a census ratchet in the style of D-DT64-AUTH: every `with_epistemic_raw5` call site goes through the admission step or is a listed LAB exception.
- No new authority: the basis types already exist (`EncounterEvidence`, `SupportLedger`, `Stamp`). What is new is that the writer must present one.

**Open gaps the protocol cannot paper over:**
- Rule identity of a derivation is not carried anywhere (tarski Q4).
- Production revision has no stamp gate (P-E2).
- `SupportReceipt.at` is a storage revision, not an epistemic view.
- W's meaning is still contested (O3, and PROBE-W-PRESERVE).

### Pre-registered falsifiers (not run; runnable today = R, needs the admission step = N)

| # | falsifier | fixture | KILL | TWIN | status |
|---|---|---|---|---|---|
| F1 | stamped Direct × Open vs unwritten zero | two rows, both `raw5 = 0`: one hydrated (written-by bit 0 set), one never claimed | the reader gives both the same answer | two stamped rows read alike | R, on the givens / thought-layers probes |
| F2 | observed/given vs mechanically derived sudoku fact | the D-GSO epistemic sudoku: a given cell vs a cell derived by elimination | the derived fact carries an independent root, or certification above the givens' | two givens stay independent | N |
| F3 | two derivations sharing one premise root | P-E2 `shared_premise_is_not_pooled_but_disjoint_roots_are` | on the production revision path, the shared premise raises c (expected to fire: production has no gate) | disjoint roots still pool | R, test-local gate; production arm N |
| F4 | retraction of a supporting premise | `SupportLedger` with one source, `withdraw_source`, then recompute | certification survives the withdrawal of its only support | withdrawing an unrelated source changes nothing | N for certification; the ledger half is R |
| F5 | a valid formal derivation earns no causal certification | NAL deduction / transitive closure over Related premises | the derived edge asserts Supports or above | a measured intervention still earns Causes | N |
| F6 | Maslov maturation does not raise Epi5 | `maturation_ladder.rs`: advance to Falsification / Integration | 59..63 changes on a maturation step | a measured result still moves 59..63 | R |

## Finding 2 — NNUE as a compiler heuristic

**Verdict: FEASIBLE AFTER two prerequisites. Not justified as a compiler integration today.**

**What exists.**
- No NNUE accumulator over weighted features exists in any workspace repo (closed search of `*.rs` in lance-graph, ndarray, brainyblaze/xlog, adaworldapi/xlog, OGAR).
  - `bgz-tensor/examples/nnue_palette_cosine.rs` checks palette fidelity of NNUE columns.
  - The real i16 HalfKA accumulator is in stockfish-rs (GPL-3.0, not checked out; never copy it into Apache repos).
  - The eaw-6, eaw-8 and Yuan notes are unexecuted research.
- No compiler integration exists in xlog-logic, egglog-r2il-probe or codegraph.

**Features readable by `RelId` today, with no strings or ASTs:**
- cardinality and access heat;
- an observed join-selectivity EMA;
- `estimate_join_cardinality` per candidate pairing;
- `DeltaRecomputeStats` and codegraph `RefreshStats` (change counts);
- `CostPredictionRecord`;
- canonical WCOJ slot roles.

Absent: adornment features, degree histograms, RIR/SSA features, a hardware descriptor, measured CPU cost per candidate, effect or ambiguity features, and any corpus.

**The one meaningful probe: XLOG's triangle inner-pairing choice.**
- The candidate set is legal by construction: 3 pairings, or 2 for a 4-cycle.
- The current choice is a hardcoded argmin over `estimate_join_cardinality` (`optimizer.rs:3395-3404`). Its estimates have no provenance and "may all collapse to the same fallback ratio" (`:1586-1593`), and then `SharedY` wins by evaluation order.
- A plug-in seam exists beside it: the `WcojVariableOrderingModel` trait (`wcoj_var_ordering.rs:78`).
- Label: the exact inner join size, Σ deg_A·deg_B, as a fold over degree counts. It is hardware-independent and computable here; there is no GPU, so no measured-ns labels.
- Instances: seeded skewed graphs, with whole families held out.
- Arms:
  - the argmin;
  - the heat-aware formula;
  - **an exact-degree arm (non-learned)**;
  - random;
  - an NNUE-style accumulator.
- Metric: regret on the held-out families.

**Accumulator.**
- Feature = (slot role, statistic kind, log2 bucket), resolved by `RelId`.
- Score `h_c = b + Σ W[f,c]` (i32).
- A statistic crossing a bucket touches at most 2 × candidates weights; a change within a bucket touches none.
- No predicate strings, `RirNode` re-walks, plan clones or row decoding on the hot path.

**Boundary.** The score is its own newtype, with no conversion into any authority:
- legality: the optimizer's shape and safety floor, and the promoter's fallback;
- rewrite admission: `laws::admit`;
- equivalence: egglog proof plus the oracle, and capsule recheck;
- amortization: `CostEvidence::pays_for_itself`;
- possible calls: codegraph's `Possible` qualifier;
- certification: `epistemic_state5`.

No code owner for purity was found in these crates.

**Pre-registered falsifiers.**

| # | falsifier | KILL | TWIN |
|---|---|---|---|
| F1 | illegal but top-scored candidate | it is emitted | a legal top-scored candidate is emitted |
| F2 | one bucket change | it touches more than 2 × candidates weights | a within-bucket change touches 0; incremental update = full refresh, byte for byte |
| F3 | no gain | held-out regret is not below the argmin, or not below the exact-degree arm with its cost counted → PARK | beats random on the training families |
| F4 | permutation of atom order, load order and `RelId` | the canonical choice changes | a changed degree distribution can change the choice |
| F5 | compile-fail: score coerces into cost evidence or an Epi5 code | it compiles | the score works in its own ranking call |

**The decisive risk.** With 3 candidates the exact cost is a cheap fold. If that fold fits the planning budget, F3 kills the learned ranker. A learned ranker pays only where the exact cost is expensive and its inputs change in small deltas.

## Closeout
STATUS: measured (research) | OUTCOME: A10 — no CE64 change justified; the stamped-vs-unwritten distinction belongs to presence (hot path) plus a writer-side admission step (sealed), and the six concerns map to existing owners with four named gaps. NNUE — feasible only after a labelled corpus and an exact-degree arm; one minimal probe named | OPEN: admission step design; rule identity of derivations; production stamp gate; certification writer; whether a learned ranker beats the exact fold
