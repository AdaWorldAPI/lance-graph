# Consolidated ledger — after the rebase to main (2026-10-11)

Supersedes the open-point lists in `RECOVERY-A-to-G.md` (V1–V17, G1–G8, F.2),
`NEUTRINO-RECOVERY.md` (A–F + addendum), the resident-cycle and CE64-composition
turns, and `GADAMER-ABLATION-RESEARCH.md` (A–G). Those files stay as evidence;
this one is the single list.

## 0. State after the rebase

| repo | branch was | now | how |
|---|---|---|---|
| lance-graph | c4be11f3, 123 behind, 0 ahead | 5700a21b (#1491) | fast-forward; nothing of this session was unmerged |
| ndarray | 5cc83b4, 17 behind, 0 ahead | a6541db (#357) | fast-forward |
| OGAR | eed23d1, 34 behind, 0 ahead | 149a3a6 (#356) | fast-forward |

No history was rewritten. Of the files behind this ledger's open points, only
`canonical_node.rs` (V2/V3 tail un-gated, #1490) and quack (range intersection,
#1487; procedures plan, #1476/#1484) changed. The Moore tenants, revision, Pearl,
chain replay, causal-edge, `lab.rs` and `causal_bridge.rs` are unchanged, so
every finding below still reads against current code.

## 1. What merged since, and which point it touches

| merged | what it is | touches | effect |
|---|---|---|---|
| #1465, #1474, #1490 (D-HPR-2/7/8/6a) | readings come from the plug; reading-only plugs; V2/V3 tail un-gated | G5, G6, resident cycle | **unblocks** G5/G6 (they waited on #1465), but shows the real blocker: `SlabReading` has only Facet96 / Register128 / RegisterI4x32 / RegisterI8x16 (`hotplug.rs:231-258`). No variant can declare a Moore lane. |
| #1466 + dir-sim #1482/#1485 | split-tunnel persistence; dir-sim's history as a Lance log, one batch per append | resident cycle step 4, Gadamer horizon persistence, witness | the pattern now has a second consumer; one write = one sparse SET per lane is the write-back contract |
| #1489 D-PHOTO-0 | 64 resident populations: bit-sliced planes win on every regime but one (0.03–0.05 ms per query over 1M addresses); missing a **population-cache descriptor** | 64 ablation scenarios | 64 cut scenarios = 64 populations; planes are the right carrier; the scope tuple it asks for is exactly what a cut scenario needs (universe, snapshot, expression = cut set) |
| #1476/#1478/#1479/#1484 D-SQLP | SQL procedures as R2IL fold calls over mask-risc; W1–W3 authorized, W4+ gated on a benchmark | Gadamer as a procedure, F.2 query shape | candidate home for "revise" as a named procedure (see dot D4); not authorized beyond W3 |
| #1491 | one-subject rule: a delegated actor (RFC 8693 `act`) authorizes as the principal only and holds no roles of its own | moving focus | a focus is an actor, never a subject: it cannot write for several owners in one request (dot D7) |
| #1471 synergy ledger | classid/hotplug/glove ledger (another session) | dedup | no overlap with this ledger except the hotplug reading path; `contract-slab-declaration-has-no-tail` confirms the SlabReading gap from its side |
| ndarray #356/#357 D-KEY16 | contiguous fast path for the 16-byte key match | identity = HHTL address | neighbour identity is computed by offset, not matched; low relevance, noted only |
| OGAR #350 | OGIT tables as u16 lanes | legality veto | `Allowed (src, verb, dst)` as a pre-op veto stays a candidate; ids are row ordinals over a moving tree (their own ledger: `ogar-p23`), so they are not stable addresses |

## 2. Closed ends (with the reason they are closed)

| id | item | how it closed |
|---|---|---|
| C1 | "13,000 LOC at risk" | all 14,767 + 1,352 lines are merged; no untracked or unpushed work in any repo |
| C2 | where CE64 bits 0..39 live | SPO = the HHTL key as Morton Cartesian address (neighbour = `checked_offset`); F/C = `Nars16x8`; witness = (location, slot); operation = the thought's recipe |
| C3 | witness meaning conflict | resolved: the 6-bit witness is a handle to the transition record, and for a Moore tenant that record is keyed (row = location, lane = slot); `lift_ce64`'s common witness is the location half |
| C4 | #1465 dependency | merged; G5/G6 no longer wait on it |
| C5 | #1427 "merged without measurements" | wrong: it measured correlation, and D-MOORE-OBSERVABLE-FIRST-0 measured decision soundness |
| C6 | ring-bucket Mexican hat (arm E) | negative result, kept: wrong at every bucket count |
| C7 | R-MHB-1 / signed window / tile64k | proposal obsolete; ndarray additions reverted (V5, V11, V12) |
| C8 | CORDIC, `vml` as SIMD, f32 recurrence | negative results kept (D-PHT-1) |
| C9 | Gadamer anti-laundering | VERIFIED: Echo / ClosedCycle mint nothing (`gadamer_isa_revision`); covers falsifiers T4 |
| C10 | genuine horizon expansion from observation | VERIFIED (D-CTX-6); covers T5 |
| C11 | replay determinism | VERIFIED (chain_replay tests, Pearl F9, Alpha byte identity); covers T9 |
| C12 | overdetermination keeps certification | VERIFIED once (Pearl F7); T7 half-covered |
| C13 | observation cannot certify `Causes` | VERIFIED (`certification.rs:306-327`, F1) |
| C14 | V14 duplicate commits, V17 force push | inert / not repeated; no action |

## 3. Excluded (decided not to pursue, with the reason)

| id | item | why excluded |
|---|---|---|
| X1 | `perturbation-sim` as an event detector | it is a power-grid outage simulator; its f64 `RollingFloor` mirrors ndarray `Cascade`, a third floor would be duplication |
| X2 | 64 lanes on the AlphaTunnel | its 10 lanes are recipe rungs; widening redefines rungs. Keep five axes apart: rung, CE64 bit, SIMD lane, cut-scenario ordinal, overlay owner |
| X3 | 30 new recipes to fill 64 slots | no recipe touches epistemic state, chains, Pearl or horizons; inventing them fills a number, not a gap |
| X4 | KM3NeT as evidence | analogy only; nothing was read from KM3NeT, and no trigger mechanism here is shown equivalent |
| X5 | "7 ns per committed event" | durability is per batch (~1.7 ms); 7 ns can only be a resident apply |
| X6 | new statistics library | jc + `lab.rs` cover it; the work is moving `lab.rs`, not writing another |
| X7 | equating OGIT row ids with Palette256 indices | different id spaces; a mapping would have to be declared |

## 4. Open ends — one list, deduplicated

Status words: OPEN = work to do; DECISION = blocked on an operator answer;
MERGED-INTO = a former item now covered by another. Fruit: L = hours, no
operator; H = a PR with a probe, or one answer; Y = heavy (cross-repo,
contract change, or open science).

### Thread 1 — audit and hygiene (from V1–V17, G1–G8)

| id | was | item | status | fruit |
|---|---|---|---|---|
| O1 | G1, V5, V6, V8, V9, V10, #1427 label | append-only board supersession: storno the obsolete claims, mark tenants 18–20 "authorized experiment, not ratified", correct #1427 | OPEN | L |
| O2 | V2, G5, Codex P2s on #1408 | Moore tenant hygiene, **split in two**: (a) no reading needed: Epi5 through `project_state5`, writes through `tenant_update`, Energy4 renamed or documented as the inference mantissa (now: *activation i4* per your layout), a version marker for the `Full` offset; (b) the declared reading → MERGED-INTO O11 | OPEN (a) | L (a) |
| O3 | V1, G6 | Register128: drop or gate `RelativeOffset`; keep value readings only if an authority declares them | OPEN | H |
| O4 | V3, G7, V15 | CE64 replay compatibility (total replay + admission check, or a replay version gate); tighten the ±1 LSB tolerance in the same PR | DECISION | Y |
| O5 | V7, G3 | move `lab.rs` to jc with the sign-flip test; reproduce all 13 verdicts exactly | OPEN | H |
| O6 | V7, G4 | fix the five `lab.rs` defects (t-quantile above 30 df, `higher_is_better`, multiplicity, tolerance, doc), re-run, list changed verdicts | OPEN, after O5 | H |
| O7 | V13 | model-identifier trailer vs CLAUDE.md rule | DECISION | — |

### Thread 2 — learning seam

| id | was | item | status | fruit |
|---|---|---|---|---|
| O8 | F.2, G2 | the HashMap learner vs the same learner on quack `GroupAddr::Pair` + SumI32: 100 % decision agreement, flat allocation, byte-identical replay; arm B′ (CE64 u8 carrier) predicted to diverge; arm C3 echo with and without stamp dedup | OPEN, unblocked | L |

### Thread 3 — resident Moore × CE64 cycle

| id | was | item | status | fruit |
|---|---|---|---|---|
| O9 | continuity law steps 1–8 | operand composed from resident state (C2) | design settled | — |
| O10 | opcode-in-edge conflict | shipped `forward` reads its opcode from the weight edge's bits 46..49; your contract puts the operation in the thought and reads 46..49 as activation. The resident cycle must pass the op from the recipe and pin "changing 46..49 does not change the op" (it fails today, by design) | OPEN (test documents the gap) | L |
| O11 | V2/G5 reading half, `Nars16x8` question | a `SlabReading` variant for Moore lanes (`Nars16x8` as frequency + confidence; `MooreNars16`; phase pairs), declared through the plug | OPEN — contract change | H |
| O12 | minimal seam | one planner test: compose → op from recipe → write back only the live fields as sparse SET records → reload → next cycle; falsifiers: dropped write-back, cleared plasticity bit (confidence < 0.9, disagreeing index), reload drift, plasticity from F/C alone, hidden argmax, undeclared reading refused | OPEN, can start test-local before O11 | H |
| O13 | phase rule | the Palette256 × NARS16 phase interaction and the rule for which candidate becomes active, written as a read of stored state | DECISION (needs your rule) | — |
| O14 | freeze-only plasticity | `learn` only freezes; does the hypothesis need thawing (a new op)? | DECISION | — |

### Thread 4 — event detection (neutrino reading)

| id | was | item | status | fruit |
|---|---|---|---|---|
| O15 | Experiment A | detection quality (TPR at fixed FPR) of coincidence / + floor / + Mexican hat / both, on synthetic tracks with background, distractors and a drifting rate | DECISION (your go) | H |
| O16 | RollingFloor drift rule | untested in the drifting-background role | MERGED-INTO O15 | — |
| O17 | Experiment B | moving focus vs Moore8 / 4 windows / strict 4 | OPEN, after O12 | Y |
| O18 | diagonal ownership | each diagonal falls in two N/E/S/W windows; one window must own it | OPEN, design | L |
| O19 | Experiment E | runtime rows: LUT, popcount, ISA op, resident event, committed batch | OPEN, after O12 | H |
| O20 | Cascade Stroke 1 recall | drops unresolved candidates silently | OPEN (ndarray, isolate) | H |
| O21 | prefilter mask not wired | `meta_prefilter` still returns a row-index vector | OPEN | L |

### Thread 5 — reasoning: hydration, Pearl, ablation, Gadamer, teleology

| id | was | item | status | fruit |
|---|---|---|---|---|
| O22 | Gadamer G | **nothing computes `CounterfactualVerdict`**, so `is_acceptable()` is unreachable | OPEN | L for LoadBearing → Necessary and Inert → Dispensable on a one-claim chain |
| O23 | Gadamer G decision 1 | TruthOnly (truth moved, verdict did not): Dispensable hides overdetermination, Necessary overclaims; keep NotRun or add a fourth reading | DECISION | — |
| O24 | Gadamer G decision 2, P7 | claim ↔ chain-step map (root ablation is degenerate) | MERGED-INTO D2 below | Y |
| O25 | deduction mints support | `replay_step` revises at every hop with no evidential-base check (confidence 200 → 237; self-revision raises it) | OPEN: pin as a test first, fix with D2 | L (pin) / H (fix) |
| O26 | teleology defect | `causal_bridge.rs:190` maps Intentional to (Do, Causes) | OPEN: pin first; what it should map to is a DECISION | L (pin) |
| O27 | first divergence + cone | `first_divergence` exists and is unused; no cone | OPEN | H |
| O28 | joint cuts, 64 scenarios | `Edit` has one cut; 64 scenarios as bit-sliced planes need D-PHOTO-0's population-cache descriptor | OPEN | H |
| O29 | horizon identity revision (T8) | question / language / grammar / codebook / lens are copied unchanged; what would justify changing them is unspecified | OPEN — falsifier first, then a contract | Y |
| O30 | horizon persistence | revised horizons are stored nowhere | OPEN, via split tunnel | H |
| O31 | eligibility from epistemic masks | the affordance law exists only in a doc comment and the D-CTX-7 probe | OPEN | H |
| O32 | deductive closure | none, over hydrated relations | OPEN | Y |
| O33 | teleology evidence classes | intent / function / means–end / rationalisation have no evidence type | OPEN | Y |
| O34 | Pearl do() on a graph | `planner::pearl` uses trial arms; `learning::scm` has a real SCM, unconnected | OPEN | Y |
| O35 | recipe schedule depth 1 | `requires()`/`writes()` cover 8 scalar fields; ordering never materializes | OPEN (finding, no action proposed) | Y |

## 5. The dots to connect, and why

**D1 — The resident cycle is three merged pieces plus one variant.**
Composition (C2) + split-tunnel write-back (#1466: one SET per changed lane) +
plug readings (#1465/#1474) give everything the cycle needs except a
`SlabReading` variant that can declare a Moore lane (O11). The tenant hygiene
item (O2a) and the opcode-in-edge test (O10) ride along. Why it matters: it is
the only way "NARS16 × 8 is resident state" stops being a test-held payload, which
is exactly what #1406's equivalence relied on.

**D2 — One provenance identity closes four open points at once.** These look
unrelated and are the same question, "which independent source does this
evidence come from":

| where | symptom |
|---|---|
| CE64 witness (C3) | the handle should resolve to a transition record |
| split-tunnel log | every record already carries `(version, owner, op, row, lane)` |
| `replay_step` (O25) | no check, so deduction re-counts the same evidence |
| Gadamer (O24) | `independent_roots` vs `inherited_roots` need a stable root id, and claims need to point at the steps that support them |
| F.2 arm C3 (O8) | echo inflation is the same defect measured statistically |

If the root id of a piece of evidence is the coordinate of its log record, then
the witness resolves to it, `replay_step` can refuse a root already in the
running base, Gadamer roots become those ids, and the claim ↔ step map is "the
records this claim's chain consumed". No new registry: the log already exists.
This is the keystone; it is heavy only because it touches four places, not
because any one is hard.

**D3 — Ablation already has its readouts; it lacks the joints.**
Pearl gives three reactions, the crossword probe four (incl. SilentlyNecessary),
`first_divergence` the location. Joining them: cut → two traces →
`first_divergence` → per-step XOR = cone (O27) → reaction → verdict (O22). For
many cuts, D-PHOTO-0 says carry each scenario's reachability as a bit-sliced
plane, scoped by a population-cache descriptor (O28). Only Boolean reachability
is cheap that way; each scenario's truth replay is a separate run.

**D4 — Gadamer revision already compiles to folds.** `quack/tests/spofc_cycle.rs`
shows its kind decision is a scalar table over seven `Any` folds and its result
three `Keep` outputs. D-SQLP makes R2IL procedures the API over mask-risc. So
"revise" is a natural procedure candidate. Not authorized beyond D-SQLP W3; record
as a candidate, do nothing now.

**D5 — Every gate here has the same abstain contract.** The certified
decide-or-abstain of D-MOORE-OBSERVABLE-FIRST-0, Gadamer's Suspend /
GroundingRequest, quack's refusals, and `CounterfactualVerdict::NotRun` are one
pattern: decide only with a certificate, otherwise say so. The trigger stages
(coincidence, veto, contrast) belong in front of it as cheap filters that
choose which events earn a CE64 op. This is why O15 (detection quality) matters:
a filter that is fast but not measured for recall is a Stroke-1 risk (O20).

**D6 — Purpose is never derived.** The dark-room doctrine (`.arxiv/2607.21273`:
prediction may update belief, never purpose), the Moore-as-predictor note
(X20 in the coresearch crosswalk) and the `causal_bridge` defect (O26) are one
contract: purpose claims need their own evidence class (O33) and must never be
produced by prediction or by causal certification.

**D7 — A moving focus is an actor, not an owner.** #1491's one-subject rule plus
split-tunnel rule 2 (one writer per log) mean a focus crossing owner boundaries
issues one write per owner, each under that owner's authority. Experiment B
(O17) must count those writes; "one focus, many owners, one commit" is ruled out.

## 6. What I aim for, in order

**Low (hours, no operator needed):**
1. **O22-min + O25-pin + O26-pin** — three planner/learning tests. One maps
   LoadBearing → Necessary and Inert → Dispensable on a one-claim chain and
   reaches `is_acceptable()` end to end for the first time; TruthOnly stays
   NotRun. Two pin existing defects (deduction mints support; intent becomes
   cause) so their fixes later have a failing test to turn green.
2. **O8 (F.2)** — the learning seam; no new primitive.
3. **O1 + O2a** — board supersession and the reading-free tenant hygiene,
   answering the two Codex P2s.

**High (a PR with a probe, or one answer):**
4. **O11 + O12** — the Moore `SlabReading` variant through the plug, then the
   resident-cycle test (D1). Can start test-local while the variant is reviewed.
5. **O27 + O28** — divergence cone and multi-cut scenarios as planes (D3).
6. **O5 → O6** — `lab.rs` into jc, then its defect fixes.
7. **O15** — Experiment A, once you say go.

**Heavy (needs design or open science):**
8. **D2** — the provenance identity, fixing O24, O25, C3's handle and O8-C3
   together.
9. **O29** — horizon-identity revision: write the T8 falsifier first.
10. **O33 / O34** — teleology evidence classes; connecting `learning::scm`.
11. **O4** — replay compatibility, after your decision.

## 7. Decisions only you can make

| id | question |
|---|---|
| O4 | replay: accept everything again with validation at admission, or stay strict behind a version gate? |
| O7 | model-identifier trailer on future commits? |
| O13 | the phase rule and the activation rule |
| O14 | is freeze-only plasticity acceptable, or is thawing part of the hypothesis? |
| O15 | go for Experiment A? |
| O23 | how a TruthOnly cut maps to the counterfactual verdict |
| O26 | what "A intends B" should map to instead of Causes |
